from unittest.mock import Mock

import pytest

from mllm_tokens.adapters.base import MediaSource, ModelAdapter


class StubAdapter(ModelAdapter):
    def analyze(
        self,
        messages,
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ):
        raise NotImplementedError


def test_build_segments_rejects_source_count_mismatch() -> None:
    adapter = StubAdapter(
        model_id="test/model",
        processor=Mock(),
        config=Mock(),
    )
    sources = (
        MediaSource(
            message_index=0,
            content_index=0,
            role="user",
            modality="image",
        ),
        MediaSource(
            message_index=0,
            content_index=1,
            role="user",
            modality="image",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot match image placeholder groups with input items: "
            r"processor returned 1 values, but 2 sources were provided\."
        ),
    ):
        adapter._build_segments_from_counts(
            sources=sources,
            token_counts=[256],
            source_name="image placeholder groups",
        )