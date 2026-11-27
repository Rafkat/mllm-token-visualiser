import subprocess
import tempfile
from pathlib import Path
from typing import Any

import librosa
import PIL.Image as PImage
import torch

from mllm_tokens.adapters.base import MediaSource, ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport, TokenSegment

VIDEO_SAMPLE_FPS = 1.0


class MiniCPMo45Adapter(ModelAdapter):
    ASSUMPTIONS = (
        "Images are processed with max_slice_nums=1 (a single slice per "
        "image); the model's default slicing of high-resolution images can "
        "produce more tokens.",
        f"Videos are sampled with ffmpeg at {VIDEO_SAMPLE_FPS:g} fps and each "
        "frame is processed as one image.",
    )

    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        minicpm_messages, visual_sources, audio_sources = self._to_minicpm_messages(
            normalized_messages, add_generation_prompt
        )

        inputs = self.processor(
            **minicpm_messages, max_slice_nums=1, return_tensors="pt"
        )

        segments = self._sort_segments(
            self._build_text_segments(messages),
            self._build_bound_segments(
                visual_sources,
                inputs.image_bound[0],
                bound_name="visual",
            ),
            self._build_bound_segments(
                audio_sources,
                inputs.audio_bounds[0],
                bound_name="audio",
            ),
        )

        input_ids = inputs["input_ids"]
        bounded = [
            *zip(visual_sources, inputs.image_bound[0], strict=False),
            *zip(audio_sources, inputs.audio_bounds[0], strict=False),
        ]

        return self._build_report(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            segments=segments,
            kv_cache_dtype=kv_cache_dtype,
            messages=messages,
            media_masks=self._bound_masks(input_ids, bounded),
            media_properties=self._bound_properties(visual_sources),
            add_generation_prompt=add_generation_prompt,
        )

    def _to_minicpm_messages(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: bool = False,
        omni: bool = True,
        use_tts_template: bool = False,
    ) -> tuple[dict[str, Any], list[MediaSource], list[MediaSource]]:
        images = []
        audios = []
        visual_sources = []
        audio_sources = []

        for message_index, message in enumerate(messages):
            content = message["content"]
            cur_msgs = []
            for content_index, c in enumerate(content):
                source_kwargs = {
                    "message_index": message_index,
                    "content_index": content_index,
                    "role": message["role"],
                }

                if c["type"] == "image":
                    if not Path(c["image"]).is_file():
                        raise FileNotFoundError(f"Image not found: {c['image']}")

                    image = PImage.open(c["image"])
                    images.append(image)
                    cur_msgs.append("<image>./</image>")
                    visual_sources.append(
                        MediaSource(**source_kwargs, modality="image")
                    )
                elif c["type"] == "audio":
                    if not Path(c["audio"]).is_file():
                        raise FileNotFoundError(f"Audio not found: {c['audio']}")
                    audio, _ = librosa.load(c["audio"], sr=16000, mono=True)
                    audios.append(audio)
                    cur_msgs.append("<audio>./</audio>")
                    audio_sources.append(MediaSource(**source_kwargs, modality="audio"))
                    use_tts_template = True
                elif c["type"] == "text":
                    cur_msgs.append(c["text"])
                elif c["type"] == "video":
                    if not Path(c["video"]).is_file():
                        raise FileNotFoundError(f"Video not found: {c['video']}")

                    video_segments = self.load_video_frames(c["video"])
                    cur_msgs.append("<image>./</image>" * len(video_segments))
                    images.extend(video_segments)
                    visual_sources.extend(
                        MediaSource(**source_kwargs, modality="video")
                        for _ in video_segments
                    )
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

        return processor_kwargs, visual_sources, audio_sources

    @staticmethod
    def _bound_masks(
        input_ids: Any, bounded: list[tuple[MediaSource, Any]]
    ) -> dict[str, torch.Tensor]:
        length = input_ids.shape[-1]
        masks: dict[str, torch.Tensor] = {}

        for source, (start, end) in bounded:
            mask = masks.setdefault(
                source.modality, torch.zeros(length, dtype=torch.bool)
            )
            mask[int(start) : int(end)] = True

        return masks

    @staticmethod
    def _bound_properties(
        visual_sources: list[MediaSource],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        counts: dict[MediaSource, int] = {}
        for source in visual_sources:
            counts[source] = counts.get(source, 0) + 1

        properties = {}
        for source, count in counts.items():
            key = (source.message_index, source.content_index)
            if source.modality == "image":
                properties[key] = {"slices": count, "max_slice_nums": 1}
            else:
                properties[key] = {
                    "frames_sampled": count,
                    "sample_fps": VIDEO_SAMPLE_FPS,
                }

        return properties

    def _kv_cache_config(self) -> Any:
        return self.config.get_text_config()

    @staticmethod
    def load_video_frames(
        video_path: str | Path,
        *,
        fps: float = VIDEO_SAMPLE_FPS,
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
    def _build_bound_segments(
        sources: list[MediaSource],
        bounds: Any,
        *,
        bound_name: str,
    ) -> tuple[TokenSegment, ...]:
        if len(bounds) != len(sources):
            raise ValueError(
                f"Cannot match {bound_name} bounds with input items: "
                f"processor returned {len(bounds)} bounds, "
                f"but {len(sources)} sources were provided."
            )

        token_counts: dict[MediaSource, int] = {}
        for source, (start, end) in zip(sources, bounds, strict=True):
            token_counts[source] = token_counts.get(source, 0) + int(
                (end - start).item()
            )

        return tuple(
            TokenSegment(
                message_index=source.message_index,
                content_index=source.content_index,
                role=source.role,
                modality=source.modality,
                tokens=tokens,
            )
            for source, tokens in token_counts.items()
        )
