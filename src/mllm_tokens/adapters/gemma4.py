from typing import Any

from mllm_tokens.adapters.base import MediaModality, ModelAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport, TokenSegment

GEMMA_AUDIO_VARIANTS = ("e2b", "e4b", "12b")


GEMMA_PLACEHOLDERS = {
    "image": "<|image|>",
    "video": "<|video|>",
    "audio": "<|audio|>",
}


class Gemma4Adapter(ModelAdapter):
    ASSUMPTIONS = (
        "Audio support is inferred from the model ID (e2b, e4b, 12b variants).",
        "Gemma 4 sliding-window layers are counted as full-attention layers, "
        "so the KV-cache estimate is an upper bound for long contexts.",
    )

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
            messages=messages,
            media_masks=self._gemma_media_masks(input_ids, supports_audio),
            media_properties={
                **self._position_properties(messages, inputs, "image"),
                **self._position_properties(messages, inputs, "video"),
            },
            add_generation_prompt=add_generation_prompt,
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

    def _gemma_media_masks(
        self, input_ids: Any, supports_audio: bool
    ) -> dict[str, Any]:
        modalities: tuple[MediaModality, ...] = ("image", "video")
        if supports_audio:
            modalities += ("audio",)

        return {
            modality: self._placeholder_mask(
                input_ids, modality, GEMMA_PLACEHOLDERS[modality]
            )
            for modality in modalities
        }

    def _position_properties(
        self,
        messages: list[Message],
        inputs: Any,
        modality: MediaModality,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        position_ids = inputs.get(f"{modality}_position_ids")
        if position_ids is None:
            return {}

        media_processor = getattr(self.processor, f"{modality}_processor", None)
        pooling = getattr(media_processor, "pooling_kernel_size", None)
        valid = (position_ids[..., 0] >= 0).reshape(position_ids.shape[0], -1)

        properties = {}
        sources = self._build_media_sources(messages, modality)
        for index, source in enumerate(sources[: position_ids.shape[0]]):
            item: dict[str, Any] = {"patches": int(valid[index].sum().item())}
            if isinstance(pooling, int):
                item["pooling_kernel_size"] = pooling
            if modality == "video" and position_ids.ndim == 4:
                item["frames_sampled"] = int(position_ids.shape[1])
            properties[(source.message_index, source.content_index)] = item

        return properties

    def _kv_cache_config(self) -> Any:
        return self.config.text_config
