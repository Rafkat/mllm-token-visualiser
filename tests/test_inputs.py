from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mllm_tokens import Audio, Image, Message, Text, Video


@pytest.mark.parametrize("media_class", [Image, Video, Audio])
def test_media_converts_string_to_path(media_class) -> None:
    media = media_class("assets/sample.bin")

    assert media.path == Path("assets/sample.bin")
    assert isinstance(media.path, Path)


def test_text_stores_value() -> None:
    item = Text("Describe this image.")

    assert item.text == "Describe this image."


def test_input_objects_are_immutable() -> None:
    image = Image("image.jpg")

    with pytest.raises(FrozenInstanceError):
        image.path = Path("another.jpg")


def test_message_roles() -> None:
    content = Text("Hello")

    assert Message.user(content).role == "user"
    assert Message.system(content).role == "system"
    assert Message.assistant(content).role == "assistant"


def test_message_preserves_interleaved_order() -> None:
    message = Message.user(
        Text("First"),
        Image("first.jpg"),
        Text("Second"),
        Video("second.mp4"),
        Audio("third.wav"),
    )

    assert isinstance(message.content[0], Text)
    assert isinstance(message.content[1], Image)
    assert isinstance(message.content[2], Text)
    assert isinstance(message.content[3], Video)
    assert isinstance(message.content[4], Audio)
