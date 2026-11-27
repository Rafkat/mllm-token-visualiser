from abc import ABC
from typing import Any

from mllm_tokens.adapters.base import MediaModality, ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenSegment

QWEN_PLACEHOLDERS = {
    "image": "<|image_pad|>",
    "video": "<|video_pad|>",
    "audio": "<|audio_pad|>",
}


class QwenAdapter(ModelAdapter, ABC):
    ASSUMPTIONS = (
        "Images and videos are resized and sampled by the processor's default "
        "configuration (see media[].properties for the resulting grid).",
    )

    def _build_grid_segments(
        self,
        messages: list[Message],
        inputs: Any,
        modality: MediaModality,
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

    def _grid_properties(
        self,
        messages: list[Message],
        inputs: Any,
        modality: MediaModality,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Describe the processed patch grid of each image or video."""
        grids = inputs.get(f"{modality}_grid_thw")
        if grids is None:
            return {}

        media_processor = getattr(self.processor, f"{modality}_processor", None)
        merge_size = _int_attr(media_processor, "merge_size")
        patch_size = _int_attr(media_processor, "patch_size")
        temporal_patch_size = _int_attr(media_processor, "temporal_patch_size")

        properties = {}
        sources = self._build_media_sources(messages, modality)
        for source, grid in zip(sources, grids, strict=False):
            t, h, w = (int(value) for value in grid.tolist())
            item: dict[str, Any] = {"grid_thw": [t, h, w], "merge_size": merge_size}

            if patch_size is not None:
                item["patch_size"] = patch_size
                item["processed_height"] = h * patch_size
                item["processed_width"] = w * patch_size

            if modality == "video":
                item["frame_groups"] = t
                if merge_size:
                    item["tokens_per_frame_group"] = h * w // merge_size**2
                if temporal_patch_size is not None:
                    item["temporal_patch_size"] = temporal_patch_size
                    item["frames_sampled"] = t * temporal_patch_size

            properties[(source.message_index, source.content_index)] = item

        return properties

    def _qwen_media_masks(
        self, input_ids: Any, modalities: tuple[MediaModality, ...]
    ) -> dict[str, Any]:
        return {
            modality: self._placeholder_mask(
                input_ids, modality, QWEN_PLACEHOLDERS[modality]
            )
            for modality in modalities
        }


def _int_attr(owner: Any, name: str) -> int | None:
    value = getattr(owner, name, None)
    return value if isinstance(value, int) else None
