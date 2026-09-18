from abc import ABC, abstractmethod
from typing import Any

from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.report import TokenReport, TokenSegment


class ModelAdapter(ABC):
    def __init__(self, model_id: str, processor: Any, config: Any) -> None:
        self.model_id = model_id
        self.processor = processor
        self.config = config

    @abstractmethod
    def analyze(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool,
        kv_cache_dtype: str,
    ) -> TokenReport:
        """Preprocess the message and produce token statistics."""

    @staticmethod
    def _normalize_messages(
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
                            "text": item.text,
                        }
                    )
                elif isinstance(item, Image):
                    content.append(
                        {
                            "type": "image",
                            "image": str(item.path),
                        }
                    )
                elif isinstance(item, Video):
                    content.append(
                        {
                            "type": "video",
                            "video": str(item.path),
                        }
                    )
                elif isinstance(item, Audio):
                    content.append(
                        {
                            "type": "audio",
                            "audio": str(item.path),
                        }
                    )
                else:
                    raise ValueError("Unsupported input type.")

            result.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )

        return result

    @staticmethod
    def _check_audio_input(normalized_messages: list[dict]) -> None:
        for message in normalized_messages:
            for content in message["content"]:
                if content["type"] == "audio":
                    raise ValueError("Not supported input type: audio")

    @staticmethod
    def _check_only_one_video_input(normalized_messages: list[dict]) -> None:
        video_counter = 0
        for message in normalized_messages:
            for content in message["content"]:
                if content["type"] == "video":
                    video_counter += 1

        if video_counter > 1:
            raise ValueError(
                "This model currently supports only one video per request. "
                "The Nemotron processor does not correctly expand "
                "multiple <video> placeholders."
            )

    def _build_text_segments(self, messages: list[Message]) -> list[TokenSegment]:
        tokenizer = self.processor.tokenizer
        segments = []

        for message_index, message in enumerate(messages):
            for content_index, item in enumerate(message.content):
                if not isinstance(item, Text):
                    continue

                tokens = len(
                    tokenizer.encode(
                        item.text,
                        add_special_tokens=False,
                    )
                )

                segments.append(
                    TokenSegment(
                        message_index=message_index,
                        content_index=content_index,
                        role=message.role,
                        modality="text",
                        tokens=tokens,
                    )
                )

        return segments

    def _build_image_segments(self, messages: list[Message]) -> list[TokenSegment]:
        return []

    def _build_video_segments(self, messages: list[Message]) -> list[TokenSegment]:
        return []

    def _build_audio_segments(self, messages: list[Message]) -> list[TokenSegment]:
        return []
