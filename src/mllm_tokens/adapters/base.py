from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.report import TokenReport, TokenSegment

MediaModality = Literal["image", "video", "audio"]
SegmentModality = Literal["text", "image", "video", "audio"]


@dataclass(frozen=True, slots=True)
class MediaSource:
    message_index: int
    content_index: int
    role: str
    modality: MediaModality


class ModelAdapter(ABC):
    def __init__(self, model_id: str, processor: Any, config: Any) -> None:
        self.model_id = model_id
        self.processor = processor
        self.config = config

    @abstractmethod
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        """Preprocess the message and produce token statistics."""

    def _build_report(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        segments: Sequence[TokenSegment],
        kv_cache_dtype: str,
    ) -> TokenReport:
        """Aggregate per-segment counts into a report.

        ``template_tokens`` is every active position not attributed to a
        text or media segment.
        """
        total_tokens = int(attention_mask.sum().item())

        text_tokens = self._sum_segment_tokens(segments, "text")
        image_tokens = self._sum_segment_tokens(segments, "image")
        video_tokens = self._sum_segment_tokens(segments, "video")
        audio_tokens = self._sum_segment_tokens(segments, "audio")

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
            token_id_bytes=input_ids.numel() * input_ids.element_size(),
            kv_cache_bytes=total_tokens * kv_bytes_per_token,
            kv_cache_bytes_per_token=kv_bytes_per_token,
            segments=tuple(segments),
        )

    def _kv_cache_bytes_per_token(self, dtype: str) -> int:
        """Estimate decoder KV-cache bytes per active context position.

        ``2 x cached_layers x num_key_value_heads x head_dim x dtype_bytes``
        """
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype: {dtype}")

        config = self._kv_cache_config()
        num_layers = self._num_kv_cache_layers(config)
        num_attention_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // num_attention_heads
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]

    def _kv_cache_config(self) -> Any:
        """Return the decoder configuration that owns the KV cache."""
        return getattr(self.config, "text_config", self.config)

    @staticmethod
    def _num_kv_cache_layers(config: Any) -> int:
        """Return the number of decoder layers that keep keys and values."""
        return config.num_hidden_layers

    @staticmethod
    def _normalize_messages(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        result = []

        for message in messages:
            content = []

            for item in message.content:
                if isinstance(item, Text):
                    content.append(
                        {
                            "type": "text",
                            "text": item.text,
                        }
                    )
                elif isinstance(item, Image):
                    content.append(
                        {
                            "type": "image",
                            "image": str(item.path),
                        }
                    )
                elif isinstance(item, Video):
                    content.append(
                        {
                            "type": "video",
                            "video": str(item.path),
                        }
                    )
                elif isinstance(item, Audio):
                    content.append(
                        {
                            "type": "audio",
                            "audio": str(item.path),
                        }
                    )
                else:
                    raise ValueError("Unsupported input type.")

            result.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )

        return result

    @staticmethod
    def _check_audio_input(normalized_messages: list[dict]) -> None:
        for message in normalized_messages:
            for content in message["content"]:
                if content["type"] == "audio":
                    raise ValueError("Not supported input type: audio")

    def _build_text_segments(self, messages: list[Message]) -> tuple[TokenSegment, ...]:
        tokenizer = self.processor.tokenizer
        segments = []

        for message_index, message in enumerate(messages):
            for content_index, item in enumerate(message.content):
                if not isinstance(item, Text):
                    continue

                tokens = len(
                    tokenizer.encode(
                        item.text,
                        add_special_tokens=False,
                    )
                )

                segments.append(
                    TokenSegment(
                        message_index=message_index,
                        content_index=content_index,
                        role=message.role,
                        modality="text",
                        tokens=tokens,
                    )
                )

        return tuple(segments)

    @staticmethod
    def _build_media_sources(
        messages: list[Message],
        modality: MediaModality,
    ) -> tuple[MediaSource, ...]:
        media_types = {
            "image": Image,
            "video": Video,
            "audio": Audio,
        }

        if modality not in media_types:
            raise ValueError(f"Unsupported segment modality: {modality}")

        media_type = media_types[modality]
        return tuple(
            MediaSource(
                message_index=message_index,
                content_index=content_index,
                role=message.role,
                modality=modality,
            )
            for message_index, message in enumerate(messages)
            for content_index, item in enumerate(message.content)
            if isinstance(item, media_type)
        )

    @staticmethod
    def _build_segments_from_counts(
        sources: Sequence[MediaSource],
        token_counts: Sequence[int],
        *,
        source_name: str,
    ) -> tuple[TokenSegment, ...]:
        if len(token_counts) != len(sources):
            raise ValueError(
                f"Cannot match {source_name} with input items: "
                f"processor returned {len(token_counts)} values, "
                f"but {len(sources)} sources were provided."
            )

        return tuple(
            TokenSegment(
                message_index=source.message_index,
                content_index=source.content_index,
                role=source.role,
                modality=source.modality,
                tokens=tokens,
            )
            for source, tokens in zip(sources, token_counts, strict=True)
        )

    def _build_placeholder_segments(
        self,
        messages: list[Message],
        input_ids: Any,
        attention_mask: Any,
        *,
        modality: MediaModality,
        token: str,
    ) -> tuple[TokenSegment, ...]:
        sources = self._build_media_sources(messages, modality)
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
        matches = (input_ids == token_id) & attention_mask.bool()

        if matches.ndim != 2 or matches.shape[0] != 1:
            raise ValueError(
                "TokenSegment currently supports one processed conversation at a time."
            )

        run_lengths = []
        current_run = 0
        for is_match in matches[0].tolist():
            if is_match:
                current_run += 1
            elif current_run:
                run_lengths.append(current_run)
                current_run = 0

        if current_run:
            run_lengths.append(current_run)

        return self._build_segments_from_counts(
            sources,
            run_lengths,
            source_name=f"{modality} placeholder groups",
        )

    @staticmethod
    def _sort_segments(
        *segment_groups: Sequence[TokenSegment],
    ) -> tuple[TokenSegment, ...]:
        return tuple(
            sorted(
                (segment for group in segment_groups for segment in group),
                key=lambda segment: (segment.message_index, segment.content_index),
            )
        )

    @staticmethod
    def _sum_segment_tokens(
        segments: Sequence[TokenSegment],
        modality: SegmentModality,
    ) -> int:
        return sum(
            segment.tokens for segment in segments if segment.modality == modality
        )
