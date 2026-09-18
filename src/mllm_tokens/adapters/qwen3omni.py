from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.adapters.qwen_adapter import QwenAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport


class Qwen3OmniAdapter(QwenAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        inputs = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        total_tokens = int(attention_mask.sum().item())
        text_segments = self._build_text_segments(messages)
        image_segments = self._build_grid_segments(
            messages,
            inputs,
            "image",
        )
        video_segments = self._build_grid_segments(
            messages,
            inputs,
            "video",
        )
        audio_segments = self._build_placeholder_segments(
            messages,
            input_ids,
            attention_mask,
            modality="audio",
            token=getattr(self.processor, "audio_token", "<|audio_pad|>"),
        )
        segments = self._sort_segments(
            text_segments,
            image_segments,
            video_segments,
            audio_segments,
        )

        text_tokens = self._sum_segment_tokens(segments, "text")
        image_tokens = self._sum_segment_tokens(segments, "image")
        video_tokens = self._sum_segment_tokens(segments, "video")
        audio_tokens = self._sum_segment_tokens(segments, "audio")

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
            segments=segments,
        )

    def _kv_cache_bytes_per_token(
        self,
        dtype: str,
    ) -> int:
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported KV-cache dtype {dtype}")

        text_config = self.config.thinker_config.text_config

        num_layers = text_config.num_hidden_layers
        num_attention_heads = text_config.num_attention_heads

        num_kv_heads = getattr(text_config, "num_key_value_heads", num_attention_heads)

        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // num_attention_heads
        )

        return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]
