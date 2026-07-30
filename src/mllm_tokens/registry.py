# TODO перечитать код и задать вопросы по коду
# TODO есть некоторые неясности в имплементации сервиса


from collections.abc import Callable

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter

AdapterFactory = Callable[..., ModelAdapter]


def get_adapter_class(model_id: str) -> type[ModelAdapter]:
    normalized = model_id.lower()

    if "qwen3-vl" in normalized:
        return Qwen3VLAdapter

    raise ValueError(f"Unsupported model: {model_id!r}. No matching adapter was found.")
