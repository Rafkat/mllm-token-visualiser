from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import torch

from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.media import probe_media
from mllm_tokens.report import MediaDetail, TokenReport, TokenSegment, TokenSpan
from mllm_tokens.spans import SpanAlignmentError, TextSource, build_spans

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

    # Model-family assumptions that apply to every report from this adapter.
    ASSUMPTIONS: tuple[str, ...] = ()

    COMMON_ASSUMPTIONS: tuple[str, ...] = (
        "text_tokens: each Text item is tokenized on its own without special "
        "tokens; merges with neighbouring template characters can shift a few "
        "tokens between text and template (see TokenSpan.exact).",
        "KV cache: 2 x cached layers x KV heads x head_dim x dtype bytes per "
        "active position. Excludes weights, activations, and framework "
        "overhead. Sliding-window attention, cross-layer KV sharing, and "
        "quantized caches are not modelled.",
    )

    def _build_report(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        segments: Sequence[TokenSegment],
        kv_cache_dtype: str,
        messages: Sequence[Message] | None = None,
        media_masks: Mapping[str, Any] | None = None,
        media_properties: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
        add_generation_prompt: bool | None = None,
        warnings: Sequence[str] = (),
    ) -> TokenReport:
        """Aggregate per-segment counts into a report.

        ``template_tokens`` is every active position not attributed to a
        text or media segment. When ``messages`` and ``media_masks`` (one
        boolean placeholder mask per media modality, over the unpadded row)
        are given, the report also carries per-position ``spans``. Span
        alignment failures are reported as warnings and never fail analysis.
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

        all_warnings = list(warnings)
        spans: tuple[TokenSpan, ...] = ()
        if messages is not None and media_masks is not None:
            try:
                spans, span_warnings = self._build_spans(
                    messages, input_ids, attention_mask, segments, media_masks
                )
                all_warnings.extend(span_warnings)
            except SpanAlignmentError as error:
                all_warnings.append(f"Token spans unavailable: {error}")

        media = (
            self._build_media_details(messages, segments, spans, media_properties or {})
            if messages is not None
            else ()
        )

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
            spans=spans,
            media=media,
            metadata=self._run_metadata(kv_cache_dtype, add_generation_prompt),
            assumptions=self.COMMON_ASSUMPTIONS + self.ASSUMPTIONS,
            warnings=tuple(all_warnings),
        )

    def _build_spans(
        self,
        messages: Sequence[Message],
        input_ids: Any,
        attention_mask: Any,
        segments: Sequence[TokenSegment],
        media_masks: Mapping[str, Any],
    ) -> tuple[tuple[TokenSpan, ...], tuple[str, ...]]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise SpanAlignmentError(
                "Token spans support one processed conversation at a time."
            )

        active = attention_mask[0].bool()
        active_ids = input_ids[0][active].tolist()
        masks = {
            modality: _as_bool_tensor(mask)[active].tolist()
            for modality, mask in media_masks.items()
        }
        tokenizer = self.processor.tokenizer

        text_sources = [
            TextSource(
                message_index=message_index,
                content_index=content_index,
                role=message.role,
                text=item.text,
                token_ids=tuple(tokenizer.encode(item.text, add_special_tokens=False)),
            )
            for message_index, message in enumerate(messages)
            for content_index, item in enumerate(message.content)
            if isinstance(item, Text)
        ]

        return build_spans(
            active_ids,
            masks,
            [segment for segment in segments if segment.modality != "text"],
            text_sources,
            decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        )

    @staticmethod
    def _build_media_details(
        messages: Sequence[Message],
        segments: Sequence[TokenSegment],
        spans: Sequence[TokenSpan],
        media_properties: Mapping[tuple[int, int], Mapping[str, Any]],
    ) -> tuple[MediaDetail, ...]:
        details = []

        for segment in segments:
            if segment.modality == "text":
                continue

            key = (segment.message_index, segment.content_index)
            item = messages[segment.message_index].content[segment.content_index]
            item_spans = [
                span
                for span in spans
                if (span.message_index, span.content_index) == key
            ]
            groups = sum(1 for span in item_spans if span.kind == segment.modality)
            overhead = sum(
                span.tokens for span in item_spans if span.kind == "template"
            )

            properties: dict[str, Any] = {
                **probe_media(item.path, segment.modality),
                **media_properties.get(key, {}),
            }
            _add_rates(properties, segment.tokens)

            details.append(
                MediaDetail(
                    message_index=segment.message_index,
                    content_index=segment.content_index,
                    role=segment.role,
                    modality=segment.modality,
                    source=str(item.path),
                    tokens=segment.tokens,
                    overhead_tokens=overhead,
                    placeholder_groups=groups,
                    properties=properties,
                )
            )

        return tuple(details)

    def _run_metadata(
        self, kv_cache_dtype: str, add_generation_prompt: bool | None
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "library_version": _package_version("mllm-token-visualiser"),
            "transformers_version": _package_version("transformers"),
            "adapter": type(self).__name__,
            "processor_class": type(self.processor).__name__,
            "add_generation_prompt": add_generation_prompt,
            "kv_cache_dtype": kv_cache_dtype,
        }

        try:
            metadata["kv_cache"] = self._kv_cache_dimensions(kv_cache_dtype)
        except (AttributeError, TypeError, ValueError):
            metadata["kv_cache"] = None

        return metadata

    def _kv_cache_bytes_per_token(self, dtype: str) -> int:
        """Estimate decoder KV-cache bytes per active context position.

        ``2 x cached_layers x num_key_value_heads x head_dim x dtype_bytes``
        """
        dims = self._kv_cache_dimensions(dtype)

        return (
            2
            * dims["cached_layers"]
            * dims["num_key_value_heads"]
            * dims["head_dim"]
            * dims["dtype_bytes"]
        )

    def _kv_cache_dimensions(self, dtype: str) -> dict[str, int]:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype: {dtype}")

        config = self._kv_cache_config()
        num_attention_heads = config.num_attention_heads
        dims = {
            "cached_layers": self._num_kv_cache_layers(config),
            "num_key_value_heads": getattr(
                config, "num_key_value_heads", num_attention_heads
            ),
            "head_dim": getattr(
                config, "head_dim", config.hidden_size // num_attention_heads
            ),
            "dtype_bytes": DTYPE_BYTES[dtype],
        }

        if not all(isinstance(value, int) for value in dims.values()):
            raise TypeError("KV-cache dimensions must be integers.")

        return dims

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

    def _placeholder_token_id(self, modality: MediaModality, default: str) -> int:
        """Return the processor's placeholder token ID for ``modality``."""
        token_id = getattr(self.processor, f"{modality}_token_id", None)
        if isinstance(token_id, int):
            return token_id

        token = getattr(self.processor, f"{modality}_token", default)
        return self.processor.tokenizer.convert_tokens_to_ids(token)

    def _placeholder_mask(
        self, input_ids: Any, modality: MediaModality, default: str
    ) -> torch.Tensor:
        return input_ids[0] == self._placeholder_token_id(modality, default)

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


def _as_bool_tensor(mask: Any) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise SpanAlignmentError(
            "Placeholder masks must be one-dimensional, "
            f"got shape {tuple(tensor.shape)}."
        )
    return tensor.bool()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _add_rates(properties: dict[str, Any], tokens: int) -> None:
    duration = properties.get("duration_seconds")
    if isinstance(duration, int | float) and duration > 0:
        properties["tokens_per_second"] = round(tokens / duration, 2)

    frames = properties.get("frames_sampled")
    if isinstance(frames, int) and frames > 0:
        properties["tokens_per_frame"] = round(tokens / frames, 2)
