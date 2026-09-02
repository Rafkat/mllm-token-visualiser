import torch

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Message, Text
from mllm_tokens.report import TokenReport

gemma_audio_variants = ["e2b", "e4b", "12b"]


class Gemma4Adapter(ModelAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        if any(audio_gemma in self.model_id for audio_gemma in gemma_audio_variants):
            audio_count = True
        else:
            self.check_audio_input(normalized_messages)
            audio_count = False

        inputs = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        total_tokens = int(attention_mask.sum().item())
        text_tokens = self._count_content_text_tokens(messages)

        image_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "image_token", "<|image|>"),
        )

        video_tokens = self._count_placeholder_tokens(
            input_ids,
            attention_mask,
            getattr(self.processor, "video_token", "<|video|>"),
        )

        if audio_count:
            audio_tokens = self._count_placeholder_tokens(
                input_ids,
                attention_mask,
                getattr(self.processor, "audio_token", "<|audio|>"),
            )
        else:
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

    def _count_content_text_tokens(self, messages: list[Message]) -> int:
        tokenizer = self.processor.tokenizer
        count = 0

        for message in messages:
            for item in message.content:
                if isinstance(item, Text):
                    count += len(tokenizer.encode(item.text, add_special_tokens=False))

        return count

    def _count_placeholder_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token: str
    ) -> int:
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)

        return int(((input_ids == token_id) & attention_mask).sum().item())

    def _kv_cache_bytes_per_token(
        self,
        dtype: str,
    ) -> int:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype {dtype}")

        text_config = self.config.text_config

        num_layers = text_config.num_hidden_layers
        num_attention_heads = text_config.num_attention_heads

        num_kv_heads = getattr(text_config, "num_key_value_heads", num_attention_heads)

        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // num_attention_heads
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]
