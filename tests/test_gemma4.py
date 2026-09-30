from unittest.mock import Mock

import pytest
import torch

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.gemma4 import Gemma4Adapter


def test_gemma4_builds_all_modality_segments() -> None:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    tokenizer.convert_tokens_to_ids.return_value = 42
    processor = Mock(
        tokenizer=tokenizer,
        image_processor=Mock(pooling_kernel_size=2),
        video_processor=Mock(pooling_kernel_size=2),
        audio_token="<|audio|>",
    )
    input_ids = torch.zeros((1, 60), dtype=torch.int64)
    input_ids[0, 20:30] = 42
    image_positions = torch.full((1, 20, 2), -1, dtype=torch.int64)
    image_positions[0, :16] = 0
    video_positions = torch.full((1, 2, 20, 2), -1, dtype=torch.int64)
    video_positions[0, :, :16] = 0
    processor.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, 60), dtype=torch.int64),
        "image_position_ids": image_positions,
        "video_position_ids": video_positions,
    }
    adapter = Gemma4Adapter("google/gemma-4-e4b-it", processor, Mock())
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)

    report = adapter.analyze(
        [
            Message.user(
                Text("Describe everything"),
                Image("image.jpg"),
                Audio("audio.wav"),
                Video("video.mp4"),
            )
        ]
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "text", 2),
        TokenSegment(0, 1, "user", "image", 4),
        TokenSegment(0, 2, "user", "audio", 10),
        TokenSegment(0, 3, "user", "video", 8),
    )
    assert report.template_tokens == 36
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )


def test_gemma4_without_audio_support_rejects_audio() -> None:
    adapter = Gemma4Adapter("google/gemma-4-4b-it", Mock(tokenizer=Mock()), Mock())

    with pytest.raises(ValueError, match="Not supported input type: audio"):
        adapter.analyze([Message.user(Audio("audio.wav"))])
