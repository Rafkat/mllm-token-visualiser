from abc import ABC, abstractmethod
from typing import Any

from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.report import TokenReport


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

    def _normalize_messages(
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
    def check_audio_input(normalized_messages: list[dict]) -> None:
        for message in normalized_messages:
            for content in message["content"]:
                if content["type"] == "audio":
                    raise ValueError("Not supported input type: audio")
