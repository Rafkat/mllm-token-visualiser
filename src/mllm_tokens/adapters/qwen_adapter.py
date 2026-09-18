from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.inputs import Audio, Image, Message, Video
from mllm_tokens.report import TokenSegment


@dataclass(frozen=True, slots=True)
class _MediaSource:
    message_index: int
    content_index: int
    role: str
    modality: str


class QwenAdapter(ModelAdapter, ABC):
    @staticmethod
    def _build_media_sources(
        messages: list[Message],
        modality: str,
    ) -> tuple[_MediaSource, ...]:
        media_types = {
            "image": Image,
            "video": Video,
            "audio": Audio,
        }

        if modality not in media_types:
            raise ValueError(f"Unsupported segment modality: {modality}")

        media_type = media_types[modality]
        return tuple(
            _MediaSource(
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
        sources: tuple[_MediaSource, ...],
        token_counts: list[int],
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

    def _build_grid_segments(
        self,
        messages: list[Message],
        inputs: Any,
        modality: str,
    ) -> tuple[TokenSegment, ...]:
        sources = self._build_media_sources(messages, modality)
        grid_key = f"{modality}_grid_thw"
        grids = inputs.get(grid_key)

        if grids is None:
            token_counts = []
        else:
            media_processor = getattr(self.processor, f"{modality}_processor")
            merge_length = media_processor.merge_size**2
            token_counts = [int(grid.prod().item()) // merge_length for grid in grids]

        return self._build_segments_from_counts(
            sources,
            token_counts,
            source_name=grid_key,
        )

    def _build_placeholder_segments(
        self,
        messages: list[Message],
        input_ids: Any,
        attention_mask: Any,
        *,
        modality: str,
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
        segments: tuple[TokenSegment, ...],
        modality: str,
    ) -> int:
        return sum(
            segment.tokens for segment in segments if segment.modality == modality
        )
