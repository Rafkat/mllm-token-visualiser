from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mllm_tokens.adapters.gemma4 import Gemma4Adapter
from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter
from mllm_tokens.adapters.nemotron import NemotronAdapter
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter
from mllm_tokens.report import TokenSegment

DECODER = SimpleNamespace(
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    hidden_size=256,
)
# 2 (K and V) x 4 layers x 2 KV heads x 16 head_dim x 2 bytes (bfloat16)
DECODER_BF16_BYTES = 512


class TextConfigOnly:
    def __init__(self, text_config) -> None:
        self._text_config = text_config

    def get_text_config(self):
        return self._text_config


@pytest.mark.parametrize(
    ("adapter_class", "config"),
    [
        (Qwen3VLAdapter, SimpleNamespace(text_config=DECODER)),
        (
            Qwen3OmniAdapter,
            SimpleNamespace(thinker_config=SimpleNamespace(text_config=DECODER)),
        ),
        (Gemma4Adapter, SimpleNamespace(text_config=DECODER)),
        (MiniCPMo45Adapter, TextConfigOnly(DECODER)),
    ],
)
def test_kv_cache_uses_family_decoder_config(adapter_class, config) -> None:
    adapter = adapter_class("model", Mock(), config)

    assert adapter._kv_cache_bytes_per_token("bfloat16") == DECODER_BF16_BYTES
    assert adapter._kv_cache_bytes_per_token("float32") == 2 * DECODER_BF16_BYTES


def test_qwen3vl_kv_cache_accepts_flat_config() -> None:
    adapter = Qwen3VLAdapter("model", Mock(), DECODER)

    assert adapter._kv_cache_bytes_per_token("bfloat16") == DECODER_BF16_BYTES


def test_kv_cache_defaults_for_missing_kv_heads_and_head_dim() -> None:
    config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=4, hidden_size=64)
    adapter = Qwen3VLAdapter("model", Mock(), config)

    # 2 x 2 layers x 4 heads (no GQA) x (64 // 4) head_dim x 2 bytes
    assert adapter._kv_cache_bytes_per_token("float16") == 512


def test_nemotron_kv_cache_counts_only_attention_layers() -> None:
    llm_config = SimpleNamespace(
        hybrid_override_pattern="M-M*-M*M-",
        num_hidden_layers=9,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        hidden_size=256,
    )
    adapter = NemotronAdapter("model", Mock(), SimpleNamespace(llm_config=llm_config))

    # 2 x 2 attention layers x 2 KV heads x 16 head_dim x 1 byte
    assert adapter._kv_cache_bytes_per_token("float8") == 128


def test_kv_cache_rejects_unknown_dtype() -> None:
    adapter = Qwen3VLAdapter("model", Mock(), DECODER)

    with pytest.raises(ValueError, match="Unsupported KV-cache dtype: int3"):
        adapter._kv_cache_bytes_per_token("int3")


def test_build_report_excludes_padding_and_derives_template_tokens() -> None:
    adapter = Qwen3VLAdapter("org/model", Mock(), DECODER)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    segments = (
        TokenSegment(0, 0, "user", "text", 3),
        TokenSegment(0, 1, "user", "image", 2),
    )

    report = adapter._build_report(
        input_ids=torch.zeros((1, 10), dtype=torch.int64),
        attention_mask=attention_mask,
        segments=list(segments),
        kv_cache_dtype="bfloat16",
    )

    assert report.model_id == "org/model"
    assert report.total_tokens == 8
    assert report.text_tokens == 3
    assert report.image_tokens == 2
    assert report.video_tokens == 0
    assert report.audio_tokens == 0
    assert report.template_tokens == 3
    assert report.token_id_bytes == 80
    assert report.kv_cache_bytes_per_token == DECODER_BF16_BYTES
    assert report.kv_cache_bytes == 8 * DECODER_BF16_BYTES
    assert report.segments == segments
