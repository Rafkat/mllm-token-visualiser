import subprocess
import tempfile
from pathlib import Path
from typing import Any

import librosa
import PIL.Image as PImage

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Message, Text
from mllm_tokens.report import TokenReport


class MiniCPMo45Adapter(ModelAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool,
        kv_cache_dtype: str,
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        minicpm_messages, image_sources = self._to_minicpm_messages(
            normalized_messages, add_generation_prompt
        )

        inputs = self.processor(
            **minicpm_messages, max_slice_nums=1, return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        total_tokens = int(attention_mask.sum().item())
        text_tokens = self._count_content_text_tokens(messages)
        audio_tokens = sum(
            (end - start).item() for start, end in inputs.audio_bounds[0]
        )

        image_tokens, video_tokens = self._count_image_video_tokens(
            image_sources, inputs
        )

        template_tokens = (
            total_tokens - text_tokens - image_tokens - audio_tokens - video_tokens
        )

        kv_bytes_per_tokens = self._kv_cache_bytes_per_token(kv_cache_dtype)

        return TokenReport(
            model_id=self.model_id,
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
            template_tokens=template_tokens,
            token_id_bytes=(input_ids.numel() * input_ids.element_size()),
            kv_cache_bytes=total_tokens * kv_bytes_per_tokens,
            kv_cache_bytes_per_token=kv_bytes_per_tokens,
        )

    def _to_minicpm_messages(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: bool = False,
        omni: bool = True,
        use_tts_template: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        images = []
        audios = []
        image_sources = []

        for message in messages:
            content = message["content"]
            cur_msgs = []
            for c in content:
                if c["type"] == "image":
                    if not Path(c["image"]).is_file():
                        raise FileNotFoundError(f"Image not found: {c['image']}")

                    image = PImage.open(c["image"])
                    images.append(image)
                    cur_msgs.append("<image>./</image>")
                    image_sources.append("image")
                elif c["type"] == "audio":
                    if not Path(c["audio"]).is_file():
                        raise FileNotFoundError(f"Audio not found: {c['audio']}")
                    audio, _ = librosa.load(c["audio"], sr=16000, mono=True)
                    audios.append(audio)
                    cur_msgs.append("<audio>./</audio>")
                    use_tts_template = True
                elif c["type"] == "text":
                    cur_msgs.append(c["text"])
                elif c["type"] == "video":
                    if not Path(c["video"]).is_file():
                        raise FileNotFoundError(f"Video not found: {c['video']}")

                    video_segments = self.load_video_frames(c["video"])
                    cur_msgs.append("<image>./</image>" * len(video_segments))
                    images.extend(video_segments)
                    image_sources.extend(["video"] * len(video_segments))
                else:
                    raise ValueError(f"Unknown message type: {c['type']}")

            message["content"] = "\n".join(cur_msgs) if omni else "".join(cur_msgs)

        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

        processor_kwargs = {
            "text": text,
            "images": images if images else None,
            "audios": audios if audios else None,
        }

        return processor_kwargs, image_sources

    def _count_content_text_tokens(self, messages: list[Message]) -> int:
        tokenizer = self.processor.tokenizer
        count = 0

        for message in messages:
            for item in message.content:
                if isinstance(item, Text):
                    count += len(tokenizer.encode(item.text, add_special_tokens=False))

        return count

    def _kv_cache_bytes_per_token(
        self,
        dtype: str,
    ) -> int:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype {dtype}")

        text_config = self.config.get_text_config()

        num_layers = text_config.num_hidden_layers
        num_attention_heads = text_config.num_attention_heads

        num_kv_heads = getattr(text_config, "num_key_value_heads", num_attention_heads)

        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // num_attention_heads
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]

    @staticmethod
    def load_video_frames(
        video_path: str | Path,
        *,
        fps: float = 1.0,
    ) -> list[PImage.Image]:
        video_path = Path(video_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_pattern = Path(temp_dir) / "frame_%06d.jpg"

            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                "-q:v",
                "2",
                str(output_pattern),
            ]

            subprocess.run(command, check=True)

            frames = []
            for frame_path in sorted(Path(temp_dir).glob("frame_*.jpg")):
                with PImage.open(frame_path) as image:
                    frames.append(image.convert("RGB").copy())

        return frames

    @staticmethod
    def _count_image_video_tokens(image_sources, inputs) -> tuple[int, int]:
        image_bounds = inputs.image_bound[0]

        if len(image_bounds) != len(image_sources):
            raise ValueError(
                "Cannot match visual bounds with input images: "
                f"processor returned {len(image_bounds)} bounds, "
                f"but {len(image_sources)} visual inputs were provided. "
                "The processor may have split an image into several slices."
            )

        image_tokens = 0
        video_tokens = 0
        for image_source, (start, end) in zip(image_sources, image_bounds, strict=True):
            if image_source == "image":
                image_tokens += (end - start).item()
            else:
                video_tokens += (end - start).item()
        return image_tokens, video_tokens
