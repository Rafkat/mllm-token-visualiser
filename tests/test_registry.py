import pytest

from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter
from mllm_tokens.registry import get_adapter_class


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-VL-8B-Instruct", Qwen3VLAdapter),
        ("Qwen/qwen3-vl-4b-thinking", Qwen3VLAdapter),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", Qwen3OmniAdapter),
        ("qwen/qwen3-omni-test", Qwen3OmniAdapter),
        ("openbmb/MiniCPM-o-4_5", MiniCPMo45Adapter),
        ("openbmb/minicpm-o-4_5", MiniCPMo45Adapter),
    ],
)
def test_get_adapter_class(model_id, expected) -> None:
    assert get_adapter_class(model_id) == expected


def test_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        get_adapter_class("OpenAI/ChatGPT")
