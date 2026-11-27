from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import torch

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport

VIDEO_SAMPLE_FPS = 2.0
VIDEO_MAX_FRAMES = 256


class NemotronAdapter(ModelAdapter):
    ASSUMPTIONS = (
        f"Videos are sampled with PyAV at {VIDEO_SAMPLE_FPS:g} fps, up to "
        f"{VIDEO_MAX_FRAMES} frames; only one video per request is supported.",
        "video_tokens counts video positions before Efficient Video Sampling "
        "(EVS). The model drops a fixed share of them after the vision encoder "
        "(config.video_pruning_rate); media[].properties['tokens_after_evs'] is "
        "the number the language model keeps. total_tokens and the KV-cache "
        "estimate are pre-EVS.",
        "KV cache counts only attention layers in hybrid_override_pattern; "
        "recurrent Mamba-2 state is not included.",
    )

    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        self._check_only_one_video_input(normalized_messages)

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

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        video_token_id = None
        warnings: tuple[str, ...] = ()
        if video is not None:
            input_ids, attention_mask, video_token_id, warnings = (
                self._normalize_video_placeholders(input_ids, attention_mask)
            )

        media_masks = self._nemotron_media_masks(
            input_ids, has_video=video is not None, video_token_id=video_token_id
        )
        video_tokens = (
            [int(media_masks["video"][attention_mask[0]].sum())]
            if "video" in media_masks
            else []
        )

        segments = self._sort_segments(
            self._build_text_segments(messages),
            self._build_placeholder_segments(
                messages,
                input_ids,
                attention_mask,
                modality="image",
                token=getattr(self.processor, "image_token", "<image>"),
            ),
            self._build_segments_from_counts(
                self._build_media_sources(messages, "video"),
                video_tokens,
                source_name="video token counts",
            ),
            self._build_placeholder_segments(
                messages,
                input_ids,
                attention_mask,
                modality="audio",
                token=getattr(self.processor, "audio_token", "<so_embedding>"),
            ),
        )

        return self._build_report(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segments=segments,
            kv_cache_dtype=kv_cache_dtype,
            messages=messages,
            media_masks=media_masks,
            media_properties=self._nemotron_media_properties(
                messages, video_metadata, video_tokens
            ),
            add_generation_prompt=add_generation_prompt,
            warnings=warnings,
        )

    def _normalize_video_placeholders(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int | None, tuple[str, ...]]:
        """Return input IDs with one position per video embedding.

        The model scatters video embeddings into positions equal to
        ``config.video_context_token_id``. When the tokenizer has an atomic
        ``<video>`` token, the processor output already has that layout. Some
        checkpoints' tokenizers lack the token, so ``<video>`` is encoded as
        text fragments (``<``, ``video``, ``>`` merged in varying ways) and the
        processor output never contains ``video_context_token_id``. In that
        case each video frame block is rebuilt with ``num_image_token``
        positions of ``video_context_token_id``, the layout the model's
        forward pass expects. Returns ``(input_ids, attention_mask,
        video_token_id, warnings)``; ``video_token_id`` is ``None`` when the
        layout could not be determined.
        """
        token = getattr(self.processor, "video_token", "<video>")
        token_ids = self.processor.tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            return input_ids, attention_mask, token_ids[0], ()

        fragmented = (
            f"The tokenizer has no single token for the video placeholder "
            f"{token!r} (it encodes to {len(token_ids)} tokens), so the "
            "processor output splits every frame placeholder into text "
            "fragments."
        )
        video_token_id = getattr(self.config, "video_context_token_id", None)
        tokens_per_frame = getattr(
            getattr(self.processor, "image_processor", None), "num_image_token", None
        )
        if not isinstance(video_token_id, int) or not isinstance(tokens_per_frame, int):
            return (
                input_ids,
                attention_mask,
                None,
                (
                    f"{fragmented} config.video_context_token_id or "
                    "image_processor.num_image_token is unavailable, so "
                    "video_tokens counts those fragments and may not match "
                    "the model's video context.",
                ),
            )

        if input_ids.shape[0] != 1:
            raise ValueError("Nemotron video analysis supports batch size 1 only")

        active_ids = input_ids[0][attention_mask[0]].tolist()
        image_token_id = self._placeholder_token_id("image", "<image>")
        try:
            rebuilt = self._rebuild_video_blocks(
                active_ids, image_token_id, video_token_id, tokens_per_frame
            )
        except RuntimeError as error:
            return (
                input_ids,
                attention_mask,
                None,
                (f"{fragmented} Frame blocks could not be rebuilt ({error}).",),
            )
        rebuilt_ids = torch.tensor([rebuilt], dtype=input_ids.dtype)
        rebuilt_mask = torch.ones_like(rebuilt_ids, dtype=torch.bool)

        return (
            rebuilt_ids,
            rebuilt_mask,
            video_token_id,
            (
                f"{fragmented} The model only matches video embeddings to "
                f"video_context_token_id ({video_token_id}), so the report "
                f"rebuilds each frame as {tokens_per_frame} positions of that "
                "ID, the layout the model expects. Running this processor "
                "output through the model unchanged would fail.",
            ),
        )

    def _rebuild_video_blocks(
        self,
        token_ids: list[int],
        image_token_id: int,
        video_token_id: int,
        tokens_per_frame: int,
    ) -> list[int]:
        """Replace the interior of each video ``<img>`` block.

        Image blocks are recognised by their ``image_token_id`` placeholders
        and kept as they are; any other block is a video frame.
        """
        interior = self._image_block_interior(token_ids)
        rebuilt: list[int] = []
        block: list[int] = []

        for token_id, inside in zip(token_ids, interior, strict=True):
            if inside:
                block.append(token_id)
                continue

            if block:
                is_image = image_token_id in block
                rebuilt += block if is_image else [video_token_id] * tokens_per_frame
                block = []

            rebuilt.append(token_id)

        return rebuilt

    def _nemotron_media_masks(
        self,
        input_ids: torch.Tensor,
        has_video: bool,
        video_token_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        image_mask = self._placeholder_mask(input_ids, "image", "<image>")
        masks = {
            "image": image_mask,
            "audio": self._placeholder_mask(input_ids, "audio", "<so_embedding>"),
        }

        if has_video and video_token_id is not None:
            masks["video"] = input_ids[0] == video_token_id
        elif has_video:
            # Fallback when the video token ID is unknown: video frames are
            # wrapped in the same <img>...</img> blocks as images; every
            # non-image position inside a block is video. A
            # malformed block structure leaves video unmapped (a warning),
            # rather than failing the analysis.
            try:
                interior = self._image_block_interior(input_ids[0].tolist())
            except RuntimeError:
                return masks
            masks["video"] = torch.tensor(interior, dtype=torch.bool) & ~image_mask

        return masks

    def _nemotron_media_properties(
        self,
        messages: list[Message],
        video_metadata: SimpleNamespace | None,
        video_tokens: Sequence[int] = (),
    ) -> dict[tuple[int, int], dict[str, Any]]:
        properties: dict[tuple[int, int], dict[str, Any]] = {}

        tokens_per_tile = getattr(
            getattr(self.processor, "image_processor", None), "num_image_token", None
        )
        if isinstance(tokens_per_tile, int):
            for source in self._build_media_sources(messages, "image"):
                properties[(source.message_index, source.content_index)] = {
                    "tokens_per_tile": tokens_per_tile
                }

        if video_metadata is not None:
            frames = len(video_metadata.frames_indices)
            video_properties: dict[str, Any] = {
                "frames_sampled": frames,
                "sample_fps": video_metadata.fps,
                "max_frames": VIDEO_MAX_FRAMES,
            }
            rate = getattr(self.config, "video_pruning_rate", None)
            if video_tokens and frames and isinstance(rate, (int, float)):
                video_properties["evs_pruning_rate"] = rate
                video_properties["tokens_after_evs"] = evs_retained_tokens(
                    video_tokens[0], frames, rate
                )
            for source in self._build_media_sources(messages, "video"):
                key = (source.message_index, source.content_index)
                properties[key] = video_properties

        return properties

    def _image_block_interior(self, token_ids: list[int]) -> list[bool]:
        """Mark positions strictly inside ``<img>...</img>`` blocks."""
        start_id, end_id = self._image_block_token_ids()

        interior = []
        inside_block = False

        for token_id in token_ids:
            if token_id == start_id:
                if inside_block:
                    raise RuntimeError("Nested <img> blocks detected")

                inside_block = True
                interior.append(False)
                continue

            if token_id == end_id:
                if not inside_block:
                    raise RuntimeError("Unexpected </img> token")

                inside_block = False
                interior.append(False)
                continue

            interior.append(inside_block)

        if inside_block:
            raise RuntimeError("Unclosed <img> block detected")

        return interior

    def _image_block_token_ids(self) -> tuple[int, int]:
        tokenizer = self.processor.tokenizer

        start_token = getattr(self.processor, "image_start_token", "<img>")
        end_token = getattr(self.processor, "image_end_token", "</img>")

        start_ids = tokenizer.encode(start_token, add_special_tokens=False)
        end_ids = tokenizer.encode(end_token, add_special_tokens=False)

        if len(start_ids) != 1:
            raise RuntimeError(
                "Nemotron image start token is not atomic: "
                f"{start_token!r} -> {start_ids}"
            )

        if len(end_ids) != 1:
            raise RuntimeError(
                f"Nemotron image end token is not atomic: {end_token!r} -> {end_ids}"
            )

        return start_ids[0], end_ids[0]

    def _kv_cache_config(self) -> Any:
        return self.config.llm_config

    @staticmethod
    def _num_kv_cache_layers(config: Any) -> int:
        # NemotronH mixes Mamba-2, attention, and MoE layers. Only attention
        # layers ("*" in hybrid_override_pattern) keep a KV cache; recurrent
        # Mamba state is not included in this estimate.
        return config.hybrid_override_pattern.count("*")

    @staticmethod
    def load_video(
        video_path: str | Path,
        *,
        fps: float = VIDEO_SAMPLE_FPS,
        max_frames: int | None = VIDEO_MAX_FRAMES,
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

        # The processor stamps frame j at j / metadata.fps, so fps must be the
        # rate frames were sampled at, not the source frame rate. Every frame
        # is kept when the source is slower than the requested rate.
        metadata = SimpleNamespace(
            fps=min(fps, source_fps) if source_fps else fps,
            source_fps=source_fps,
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
        SimpleNamespace | None,
    ]:
        prompt_messages: list[dict[str, str]] = []

        images: list[Any] = []
        audios: list[Any] = []

        video: np.ndarray | torch.Tensor | None = None
        video_metadata: SimpleNamespace | None = None

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

    @staticmethod
    def _check_only_one_video_input(normalized_messages: list[dict]) -> None:
        video_counter = 0
        for message in normalized_messages:
            for content in message["content"]:
                if content["type"] == "video":
                    video_counter += 1

        if video_counter > 1:
            raise ValueError(
                "This model currently supports only one video per request. "
                "The Nemotron processor does not correctly expand "
                "multiple <video> placeholders."
            )


def evs_retained_tokens(video_tokens: int, frames: int, pruning_rate: float) -> int:
    """Video tokens kept by Efficient Video Sampling.

    Mirrors ``EfficientVideoSampling.compute_retention_mask`` in the model's
    remote code: ``max(per_frame, int(frames * per_frame * (1 - rate)))``. The
    count is exact; which tokens are kept depends on frame content.
    """
    if pruning_rate <= 0 or frames <= 0:
        return video_tokens

    per_frame = video_tokens // frames
    return max(per_frame, int(frames * per_frame * (1 - pruning_rate)))
