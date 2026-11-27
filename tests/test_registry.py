from types import SimpleNamespace

import pytest

from mllm_tokens.adapters.gemma4 import Gemma4Adapter
from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter
from mllm_tokens.adapters.nemotron import NemotronAdapter
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter
from mllm_tokens.registry import get_adapter_class, resolve_adapter_class


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-VL-8B-Instruct", Qwen3VLAdapter),
        ("Qwen/qwen3-vl-4b-thinking", Qwen3VLAdapter),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", Qwen3OmniAdapter),
        ("qwen/qwen3-omni-test", Qwen3OmniAdapter),
        ("openbmb/MiniCPM-o-4_5", MiniCPMo45Adapter),
        ("openbmb/minicpm-o-4_5", MiniCPMo45Adapter),
        ("google/gemma-4-e4b-it", Gemma4Adapter),
        ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", NemotronAdapter),
    ],
)
def test_get_adapter_class(model_id, expected) -> None:
    assert get_adapter_class(model_id) == expected


def test_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        get_adapter_class("OpenAI/ChatGPT")


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3_vl", Qwen3VLAdapter),
        ("qwen3_vl_moe", Qwen3VLAdapter),
        ("qwen3_omni_moe", Qwen3OmniAdapter),
        ("gemma4", Gemma4Adapter),
        ("NemotronH_Nano_Omni_Reasoning_V3", NemotronAdapter),
    ],
)
def test_resolve_adapter_class_uses_model_type_for_local_paths(
    model_type, expected
) -> None:
    config = SimpleNamespace(model_type=model_type)

    assert resolve_adapter_class("/models/my-finetune", config) == expected


def test_model_type_takes_precedence_over_model_id() -> None:
    config = SimpleNamespace(model_type="gemma4")

    assert resolve_adapter_class("org/qwen3-vl-named", config) == Gemma4Adapter


@pytest.mark.parametrize("version", ["4.5", 4.5])
def test_minicpmo_matches_supported_version(version) -> None:
    config = SimpleNamespace(model_type="minicpmo", version=version)

    assert resolve_adapter_class("/models/minicpm", config) == MiniCPMo45Adapter


def test_minicpmo_rejects_unvalidated_version() -> None:
    config = SimpleNamespace(model_type="minicpmo", version=2.6)

    with pytest.raises(ValueError, match=r"config\.model_type='minicpmo'"):
        resolve_adapter_class("openbmb/MiniCPM-o-2_6", config)


def test_resolve_adapter_class_falls_back_to_model_id() -> None:
    config = SimpleNamespace(model_type="custom_remote_type")

    assert resolve_adapter_class("Qwen/Qwen3-VL-8B-Instruct", config) == Qwen3VLAdapter
    assert resolve_adapter_class("Qwen/Qwen3-VL-8B-Instruct") == Qwen3VLAdapter


def test_resolve_adapter_class_reports_unknown_model_type() -> None:
    config = SimpleNamespace(model_type="llama")

    with pytest.raises(
        ValueError,
        match=r"Unsupported model: '/models/x' \(config\.model_type='llama'\)",
    ):
        resolve_adapter_class("/models/x", config)
