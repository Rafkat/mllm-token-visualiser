from collections.abc import Callable
from typing import Any

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.gemma4 import Gemma4Adapter
from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter
from mllm_tokens.adapters.nemotron import NemotronAdapter
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter

AdapterFactory = Callable[..., ModelAdapter]

# Exact ``config.model_type`` values, taken from the ``config.json`` of the
# checkpoints each adapter was written for.
MODEL_TYPE_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "qwen3_vl": Qwen3VLAdapter,
    "qwen3_vl_moe": Qwen3VLAdapter,
    "qwen3_omni_moe": Qwen3OmniAdapter,
    "gemma4": Gemma4Adapter,
    "NemotronH_Nano_Omni_Reasoning_V3": NemotronAdapter,
}

# ``minicpmo`` is shared by several MiniCPM-o generations. Only the version
# the adapter handles is matched; ``config.version`` tells them apart.
VERSIONED_MODEL_TYPE_ADAPTERS: dict[tuple[str, str], type[ModelAdapter]] = {
    ("minicpmo", "4.5"): MiniCPMo45Adapter,
}


def get_adapter_class(model_id: str) -> type[ModelAdapter]:
    """Resolve an adapter from a Hugging Face model ID by name matching."""
    normalized = model_id.lower()

    if "qwen3-vl" in normalized:
        return Qwen3VLAdapter
    elif "qwen3-omni" in normalized:
        return Qwen3OmniAdapter
    elif "minicpm-o-4_5" in normalized:
        return MiniCPMo45Adapter
    elif "gemma-4" in normalized:
        return Gemma4Adapter
    elif "nemotron-3-nano-omni" in normalized:
        return NemotronAdapter

    raise ValueError(f"Unsupported model: {model_id!r}. No matching adapter was found.")


def get_adapter_class_from_config(config: Any) -> type[ModelAdapter] | None:
    """Resolve an adapter from ``config.model_type``, or return ``None``."""
    model_type = getattr(config, "model_type", None)

    if not isinstance(model_type, str):
        return None

    if model_type in MODEL_TYPE_ADAPTERS:
        return MODEL_TYPE_ADAPTERS[model_type]

    version = getattr(config, "version", None)
    if version is None:
        return None

    return VERSIONED_MODEL_TYPE_ADAPTERS.get((model_type, str(version)))


def resolve_adapter_class(model_id: str, config: Any = None) -> type[ModelAdapter]:
    """Resolve an adapter, preferring ``config.model_type`` over the model ID.

    Matching on the configuration works for local paths and renamed fine-tunes.
    The model ID is used as a fallback for configurations without a known
    ``model_type``.
    """
    adapter_class = get_adapter_class_from_config(config)
    if adapter_class is not None:
        return adapter_class

    try:
        return get_adapter_class(model_id)
    except ValueError:
        model_type = getattr(config, "model_type", None)
        detail = (
            f" (config.model_type={model_type!r})"
            if isinstance(model_type, str)
            else ""
        )
        raise ValueError(
            f"Unsupported model: {model_id!r}{detail}. No matching adapter was found."
        ) from None
