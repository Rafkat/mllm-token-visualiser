from typing import Any

from mllm_tokens.adapters.qwen_adapter import QwenAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport

# The model card's qwen-omni-utils pipeline samples videos at 2 fps. The
# processor's own default (do_sample_frames=False) keeps every source frame,
# which for a 25 s, 24 fps clip is ~12x more tokens than the model sees in
# the reference pipeline and can exceed the context window.
VIDEO_SAMPLE_FPS = 2.0


class Qwen3OmniAdapter(QwenAdapter):
    ASSUMPTIONS = QwenAdapter.ASSUMPTIONS + (
        "KV cache covers the Thinker text decoder only; Talker generation "
        "memory is not included.",
        f"Videos are sampled at {VIDEO_SAMPLE_FPS:g} fps, as in the model card's "
        "qwen-omni-utils pipeline; its per-frame pixel limits are not applied, "
        "so the processor's resize defaults decide the frame resolution.",
    )

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
            **self._video_sampling_kwargs(normalized_messages),
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()

        segments = self._sort_segments(
            self._build_text_segments(messages),
            self._build_grid_segments(messages, inputs, "image"),
            self._build_grid_segments(messages, inputs, "video"),
            self._build_placeholder_segments(
                messages,
                input_ids,
                attention_mask,
                modality="audio",
                token=getattr(self.processor, "audio_token", "<|audio_pad|>"),
            ),
        )

        return self._build_report(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segments=segments,
            kv_cache_dtype=kv_cache_dtype,
            messages=messages,
            media_masks=self._qwen_media_masks(input_ids, ("image", "video", "audio")),
            media_properties={
                **self._grid_properties(messages, inputs, "image"),
                **self._grid_properties(messages, inputs, "video"),
            },
            add_generation_prompt=add_generation_prompt,
        )

    def _kv_cache_config(self) -> Any:
        # The Thinker decoder holds the autoregressive multimodal context.
        # Talker-generation memory is not part of the input-context estimate.
        return self.config.thinker_config.text_config

    @staticmethod
    def _video_sampling_kwargs(messages: list[dict[str, Any]]) -> dict[str, Any]:
        has_video = any(
            isinstance(item, dict) and item.get("type") == "video"
            for message in messages
            for item in message.get("content", [])
        )
        if not has_video:
            return {}

        return {"processor_kwargs": {"do_sample_frames": True, "fps": VIDEO_SAMPLE_FPS}}
