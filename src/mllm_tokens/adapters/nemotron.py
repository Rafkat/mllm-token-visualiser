from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Message, Text
from mllm_tokens.report import TokenReport


class NemotronAdapter(ModelAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        self.check_video_input(normalized_messages)

        (
            prompt_messages,
            images,
            audios,
            video,
            video_metadata,
        ) = self._prepare_processor_inputs(normalized_messages)

        prompt = self.processor.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }

        if images:
            processor_kwargs["images"] = images

        if audios:
            processor_kwargs["audio"] = audios

        if video is not None:
            processor_kwargs["videos"] = video

            if video_metadata is not None:
                processor_kwargs["videos_kwargs"] = {
                    "video_metadata": video_metadata,
                }

        inputs = self.processor(**processor_kwargs)
        print(self.processor.batch_decode(inputs.input_ids[0]))

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        total_tokens = int(attention_mask.sum().item())

        text_tokens = self._count_content_text_tokens(messages)

        image_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "image_token", "<image>"),
        )

        video_tokens = self._count_video_tokens(
            video=video,
            video_metadata=video_metadata,
        )

        audio_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "audio_token", "<so_embedding>"),
        )

        template_tokens = (
            total_tokens - text_tokens - image_tokens - video_tokens - audio_tokens
        )

        kv_bytes_per_token = self._kv_cache_bytes_per_token(kv_cache_dtype)

        return TokenReport(
            model_id=self.model_id,
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            template_tokens=template_tokens,
            token_id_bytes=(input_ids.numel() * input_ids.element_size()),
            kv_cache_bytes=total_tokens * kv_bytes_per_token,
            kv_cache_bytes_per_token=kv_bytes_per_token,
        )

    def _count_content_text_tokens(
        self,
        messages: list[Message],
    ) -> int:
        tokenizer = self.processor.tokenizer
        count = 0

        for message in messages:
            for item in message.content:
                if isinstance(item, Text):
                    count += len(
                        tokenizer.encode(
                            item.text,
                            add_special_tokens=False,
                        )
                    )

        return count

    def _count_placeholder_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token: str,
    ) -> int:
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)

        return int(((input_ids == token_id) & attention_mask).sum().item())

    def _count_video_tokens(
        self,
        *,
        video: np.ndarray | torch.Tensor | None,
        video_metadata: VideoMetadata | None,
    ) -> int:
        if video is None:
            return 0

        processor_kwargs: dict[str, Any] = {
            "text": ["<video>"],
            "videos": video,
            "return_tensors": "pt",
        }

        if video_metadata is not None:
            processor_kwargs["videos_kwargs"] = {
                "video_metadata": video_metadata,
            }

        video_inputs = self.processor(**processor_kwargs)

        return self._count_tokens_inside_image_blocks(
            input_ids=video_inputs["input_ids"],
            attention_mask=video_inputs["attention_mask"],
        )

    def _count_tokens_inside_image_blocks(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> int:
        tokenizer = self.processor.tokenizer

        start_token = getattr(
            self.processor,
            "image_start_token",
            "<img>",
        )
        end_token = getattr(
            self.processor,
            "image_end_token",
            "</img>",
        )

        start_ids = tokenizer.encode(
            start_token,
            add_special_tokens=False,
        )
        end_ids = tokenizer.encode(
            end_token,
            add_special_tokens=False,
        )

        if len(start_ids) != 1:
            raise RuntimeError(
                "Nemotron image start token is not atomic: "
                f"{start_token!r} -> {start_ids}"
            )

        if len(end_ids) != 1:
            raise RuntimeError(
                f"Nemotron image end token is not atomic: {end_token!r} -> {end_ids}"
            )

        start_id = start_ids[0]
        end_id = end_ids[0]

        active_ids = input_ids[0][attention_mask[0].bool()]

        count = 0
        inside_block = False

        for token_id in active_ids.tolist():
            if token_id == start_id:
                if inside_block:
                    raise RuntimeError("Nested <img> blocks detected")

                inside_block = True
                continue

            if token_id == end_id:
                if not inside_block:
                    raise RuntimeError("Unexpected </img> token")

                inside_block = False
                continue

            if inside_block:
                count += 1

        if inside_block:
            raise RuntimeError("Unclosed <img> block detected")

        return count

    def _kv_cache_bytes_per_token(
        self,
        dtype: str,
    ) -> int:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype: {dtype}")

        config = self.config.llm_config

        pattern = config.hybrid_override_pattern
        num_layers = pattern.count("*")
        num_attention_heads = config.num_attention_heads

        num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            num_attention_heads,
        )

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // num_attention_heads,
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]

    @staticmethod
    def load_video(
        video_path: str | Path,
        *,
        fps: float = 2.0,
        max_frames: int | None = 256,
    ) -> tuple[np.ndarray, SimpleNamespace]:
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        frames: list[np.ndarray] = []
        frame_indices: list[int] = []

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            source_fps = float(stream.average_rate or stream.base_rate)
            total_frames = int(stream.frames or 0)

            next_sample_time = 0.0
            sample_interval = 1.0 / fps

            for frame_index, frame in enumerate(container.decode(stream)):
                if frame.time is not None:
                    current_time = float(frame.time)
                else:
                    current_time = frame_index / source_fps

                if current_time + 1e-9 < next_sample_time:
                    continue

                frames.append(frame.to_ndarray(format="rgb24"))
                frame_indices.append(frame_index)

                next_sample_time += sample_interval

                if max_frames is not None and len(frames) >= max_frames:
                    break

        if not frames:
            raise RuntimeError(f"No frames extracted from video: {video_path}")

        video = np.stack(frames, axis=0)

        metadata = SimpleNamespace(
            fps=source_fps,
            frames_indices=frame_indices,
            total_num_frames=total_frames,
        )

        return video.transpose((0, 3, 1, 2)), metadata

    def _prepare_processor_inputs(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, str]],
        list[Any],
        list[Any],
        np.ndarray | torch.Tensor | None,
        VideoMetadata | None,
    ]:
        prompt_messages: list[dict[str, str]] = []

        images: list[Any] = []
        audios: list[Any] = []

        video: np.ndarray | torch.Tensor | None = None
        video_metadata: VideoMetadata | None = None

        image_token = getattr(
            self.processor,
            "image_token",
            "<image>",
        )
        video_token = getattr(
            self.processor,
            "video_token",
            "<video>",
        )
        audio_token = getattr(
            self.processor,
            "audio_token",
            "<so_embedding>",
        )

        for message in messages:
            content = message.get("content", [])

            if isinstance(content, str):
                prompt_messages.append(
                    {
                        "role": message["role"],
                        "content": content,
                    }
                )
                continue

            content_parts: list[str] = []

            for item in content:
                content_type = item.get("type")

                if content_type == "text":
                    content_parts.append(item["text"])

                elif content_type == "image":
                    image = item.get("image")

                    if image is None:
                        raise ValueError(
                            "Image content does not contain the 'image' field"
                        )

                    images.append(image)
                    content_parts.append(image_token)

                elif content_type == "audio":
                    audio = item.get("audio")

                    if audio is None:
                        raise ValueError(
                            "Audio content does not contain the 'audio' field"
                        )

                    audios.append(audio)
                    content_parts.append(audio_token)

                elif content_type == "video":
                    video_path = item.get("video")

                    if video_path is None:
                        raise ValueError(
                            "Video content does not contain the 'video' field"
                        )

                    video, video_metadata = self.load_video(video_path)

                    content_parts.append(video_token)

                else:
                    raise ValueError(f"Unsupported content type: {content_type!r}")

            prompt_messages.append(
                {
                    "role": message["role"],
                    "content": "\n".join(content_parts),
                }
            )

        return (
            prompt_messages,
            images,
            audios,
            video,
            video_metadata,
        )
