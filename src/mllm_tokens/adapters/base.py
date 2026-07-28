from abc import ABC, abstractmethod
from typing import Any

from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport


class ModelAdapter(ABC):
    def __init__(self, model_id: str, processor: Any, config: Any) -> None:
        self.model_id = model_id
        self.processor = processor
        self.config = config

    @abstractmethod
    def analyze(self, messages: list[Message], *, add_generation_prompt: bool, kv_cache_dtype: str) -> TokenReport:
        """Preprocess the message and produce token statistics."""