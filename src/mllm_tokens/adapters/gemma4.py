from typing import Any

from mllm_tokens.adapters.base import MediaModality, ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport, TokenSegment

GEMMA_AUDIO_VARIANTS = ("e2b", "e4b", "12b")


class Gemma4Adapter(ModelAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        supports_audio = any(
            audio_gemma in self.model_id.lower() for audio_gemma in GEMMA_AUDIO_VARIANTS
        )

        if not supports_audio:
            self._check_audio_input(normalized_messages)

        inputs = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        audio_segments = (
            self._build_placeholder_segments(
                messages,
                input_ids,
                attention_mask,
                modality="audio",
                token=getattr(self.processor, "audio_token", "<|audio|>"),
            )
            if supports_audio
            else ()
        )
        segments = self._sort_segments(
            self._build_text_segments(messages),
            self._build_position_segments(messages, inputs, "image"),
            self._build_position_segments(messages, inputs, "video"),
            audio_segments,
        )

        return self._build_report(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segments=segments,
            kv_cache_dtype=kv_cache_dtype,
        )

    def _build_position_segments(
        self,
        messages: list[Message],
        inputs: Any,
        modality: MediaModality,
    ) -> tuple[TokenSegment, ...]:
        sources = self._build_media_sources(messages, modality)
        position_key = f"{modality}_position_ids"
        position_ids = inputs.get(position_key)

        if position_ids is None:
            token_counts = []
        else:
            media_processor = getattr(self.processor, f"{modality}_processor")
            pooling_area = media_processor.pooling_kernel_size**2
            valid_positions = position_ids[..., 0] >= 0
            valid_positions = valid_positions.reshape(valid_positions.shape[0], -1)
            token_counts = [
                int(valid_count.item()) // pooling_area
                for valid_count in valid_positions.sum(dim=1)
            ]

        return self._build_segments_from_counts(
            sources,
            token_counts,
            source_name=position_key,
        )

    def _kv_cache_config(self) -> Any:
        return self.config.text_config
