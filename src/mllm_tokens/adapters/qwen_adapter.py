from abc import ABC
from typing import Any

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenSegment


class QwenAdapter(ModelAdapter, ABC):
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
