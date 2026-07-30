from typing import Any

import torch

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.report import TokenReport

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8": 1,
}


class Qwen3VLAdapter(ModelAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        hf_messages = self._to_hf_messages(messages)

        inputs = self.processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        total_tokens = int(attention_mask.sum().item())
        text_tokens = self._count_content_text_tokens(messages)

        image_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "image_token", "<|image_pad|>"),
        )

        video_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "video_token", "<|video_pad|>"),
        )

        audio_tokens = 0

        template_tokens = (
            total_tokens - text_tokens - image_tokens - video_tokens - audio_tokens
        )

        kv_bytes_per_token = self._kv_cache_bytes_per_token(kv_cache_dtype)

        return TokenReport(
            model_id=self.model_id,
            total_tokens=total_tokens,
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            template_tokens=template_tokens,
            token_id_bytes=(input_ids.numel() * input_ids.element_size()),
            kv_cache_bytes=total_tokens * kv_bytes_per_token,
            kv_cache_bytes_per_token=kv_bytes_per_token,
        )

    def _to_hf_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        result = []

        for message in messages:
            content = []

            for item in message.content:
                if isinstance(item, Text):
                    content.append(
                        {
                            "type": "text",
                            "text": item.value,
                        }
                    )
                elif isinstance(item, Image):
                    content.append(
                        {
                            "type": "image",
                            "path": str(item.path),
                        }
                    )
                elif isinstance(item, Video):
                    content.append(
                        {
                            "type": "video",
                            "path": str(item.path),
                        }
                    )
                elif isinstance(item, Audio):
                    raise ValueError("Qwen3-VL does not support audio input.")

            result.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )

        return result

    def _count_content_text_tokens(
        self,
        messages: list[Message],
    ) -> int:
        tokenizer = self.processor.tokenizer
        count = 0

        for message in messages:
            for item in message.content:
                if isinstance(item, Text):
                    count += len(
                        tokenizer.encode(
                            item.value,
                            add_special_tokens=False,
                        )
                    )

        return count

    def _count_placeholder_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token: str,
    ) -> int:
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)

        return int(((input_ids == token_id) & attention_mask).sum().item())

    def _kv_cache_bytes_per_token(
        self,
        dtype: str,
    ) -> int:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype: {dtype}")

        config = getattr(
            self.config,
            "text_config",
            self.config,
        )

        num_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads

        num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            num_attention_heads,
        )

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // num_attention_heads,
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]
