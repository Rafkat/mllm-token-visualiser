from collections.abc import Sequence
from pathlib import Path
from typing import Self

from transformers import AutoConfig, AutoProcessor

from mllm_tokens.inputs import (
    Audio,
    Image,
    Message,
    Text,
    Video,
)
from mllm_tokens.registry import get_adapter_class
from mllm_tokens.report import TokenReport


class Analyzer:
    def __init__(self, adapter) -> None:
        self._adapter = adapter

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, trust_remote_code: bool = False, **processor_kwargs
    ) -> Self:
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, **processor_kwargs
        )

        adapter_class = get_adapter_class(model_id)

        adapter = adapter_class(
            model_id=model_id,
            processor=processor,
            config=config,
        )

        return cls(adapter)

    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        return self._adapter.analyze(
            messages,
            add_generation_prompt=add_generation_prompt,
            kv_cache_dtype=kv_cache_dtype,
        )

    def analyze_prompt(
        self,
        text: str,
        *,
        images: Sequence[str | Path] = (),
        videos: Sequence[str | Path] = (),
        audios: Sequence[str | Path] = (),
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        content = [
            *(Image(path) for path in images),
            *(Video(path) for path in videos),
            *(Audio(path) for path in audios),
            Text(text),
        ]

        return self.analyze(
            [Message.user(*content)],
            add_generation_prompt=add_generation_prompt,
            kv_cache_dtype=kv_cache_dtype,
        )


def analyze(
    model_id: str,
    messages: list[Message],
    *,
    add_generation_prompt: bool = True,
    kv_cache_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> TokenReport:
    analyzer = Analyzer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    return analyzer.analyze(
        messages,
        add_generation_prompt=add_generation_prompt,
        kv_cache_dtype=kv_cache_dtype,
    )
