from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mllm_tokens.adapters.base import ModelAdapter
from mllm_tokens.adapters.dtype_bytes import DTYPE_BYTES
from mllm_tokens.inputs import Audio, Image, Message, Video
from mllm_tokens.report import TokenReport, TokenSegment

GEMMA_AUDIO_VARIANTS = ("e2b", "e4b", "12b")


@dataclass(frozen=True, slots=True)
class _MediaSource:
    message_index: int
    content_index: int
    role: str
    modality: str


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
            self.check_audio_input(normalized_messages)

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
        image_segments = self._build_position_segments(messages, inputs, "image")
        video_segments = self._build_position_segments(messages, inputs, "video")
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

    @staticmethod
    def _build_media_sources(
        messages: list[Message],
        modality: str,
    ) -> tuple[_MediaSource, ...]:
        media_types = {"image": Image, "video": Video, "audio": Audio}

        if modality not in media_types:
            raise ValueError(f"Unsupported segment modality: {modality}")

        media_type = media_types[modality]
        return tuple(
            _MediaSource(
                message_index=message_index,
                content_index=content_index,
                role=message.role,
                modality=modality,
            )
            for message_index, message in enumerate(messages)
            for content_index, item in enumerate(message.content)
            if isinstance(item, media_type)
        )

    @staticmethod
    def _build_segments_from_counts(
        sources: tuple[_MediaSource, ...],
        token_counts: list[int],
        *,
        source_name: str,
    ) -> tuple[TokenSegment, ...]:
        if len(token_counts) != len(sources):
            raise ValueError(
                f"Cannot match {source_name} with input items: "
                f"processor returned {len(token_counts)} values, "
                f"but {len(sources)} sources were provided."
            )

        return tuple(
            TokenSegment(
                message_index=source.message_index,
                content_index=source.content_index,
                role=source.role,
                modality=source.modality,
                tokens=tokens,
            )
            for source, tokens in zip(sources, token_counts, strict=True)
        )

    def _build_position_segments(
        self,
        messages: list[Message],
        inputs: Any,
        modality: str,
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

    def _build_placeholder_segments(
        self,
        messages: list[Message],
        input_ids: Any,
        attention_mask: Any,
        *,
        modality: str,
        token: str,
    ) -> tuple[TokenSegment, ...]:
        sources = self._build_media_sources(messages, modality)
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
        matches = (input_ids == token_id) & attention_mask

        if matches.ndim != 2 or matches.shape[0] != 1:
            raise ValueError(
                "TokenSegment currently supports one processed conversation at a time."
            )

        run_lengths = []
        current_run = 0
        for is_match in matches[0].tolist():
            if is_match:
                current_run += 1
            elif current_run:
                run_lengths.append(current_run)
                current_run = 0

        if current_run:
            run_lengths.append(current_run)

        return self._build_segments_from_counts(
            sources,
            run_lengths,
            source_name=f"{modality} placeholder groups",
        )

    @staticmethod
    def _sort_segments(
        *segment_groups: Sequence[TokenSegment],
    ) -> tuple[TokenSegment, ...]:
        return tuple(
            sorted(
                (segment for group in segment_groups for segment in group),
                key=lambda segment: (segment.message_index, segment.content_index),
            )
        )

    @staticmethod
    def _sum_segment_tokens(
        segments: Sequence[TokenSegment],
        modality: str,
    ) -> int:
        return sum(
            segment.tokens for segment in segments if segment.modality == modality
        )

    def _kv_cache_bytes_per_token(self, dtype: str) -> int:
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
