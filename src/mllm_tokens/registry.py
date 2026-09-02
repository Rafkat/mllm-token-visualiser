from collections.abc import Callable

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.gemma4 import Gemma4Adapter
from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter

AdapterFactory = Callable[..., ModelAdapter]


def get_adapter_class(model_id: str) -> type[ModelAdapter]:
    normalized = model_id.lower()

    if "qwen3-vl" in normalized:
        return Qwen3VLAdapter
    elif "qwen3-omni" in normalized:
        return Qwen3OmniAdapter
    elif "minicpm-o-4_5" in normalized:
        return MiniCPMo45Adapter
    elif "gemma-4" in normalized:
        return Gemma4Adapter

    raise ValueError(f"Unsupported model: {model_id!r}. No matching adapter was found.")
