from mllm_tokens.adapters.qwen_adapter import QwenAdapter
from mllm_tokens.inputs import Message
from mllm_tokens.report import TokenReport


class Qwen3VLAdapter(QwenAdapter):
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        kv_cache_dtype: str = "bfloat16",
    ) -> TokenReport:
        normalized_messages = self._normalize_messages(messages)

        self._check_audio_input(normalized_messages)

        inputs = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        segments = self._sort_segments(
            self._build_text_segments(messages),
            self._build_grid_segments(messages, inputs, "image"),
            self._build_grid_segments(messages, inputs, "video"),
        )

        return self._build_report(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"].bool(),
            segments=segments,
            kv_cache_dtype=kv_cache_dtype,
        )
