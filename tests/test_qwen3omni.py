from unittest.mock import Mock

import torch

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.qwen3omni import Qwen3OmniAdapter


def test_qwen3omni_builds_all_modality_segments() -> None:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    tokenizer.convert_tokens_to_ids.return_value = 42
    processor = Mock(
        tokenizer=tokenizer,
        image_processor=Mock(merge_size=2),
        video_processor=Mock(merge_size=2),
        audio_token="<|audio_pad|>",
    )
    input_ids = torch.zeros((1, 50), dtype=torch.int64)
    input_ids[0, 15:27] = 42
    processor.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, 50), dtype=torch.int64),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    adapter = Qwen3OmniAdapter("Qwen/Qwen3-Omni-30B-A3B-Instruct", processor, Mock())
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)

    report = adapter.analyze(
        [
            Message.user(
                Image("image.jpg"),
                Text("Describe everything"),
                Audio("audio.wav"),
                Video("video.mp4"),
            )
        ]
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "image", 4),
        TokenSegment(0, 1, "user", "text", 2),
        TokenSegment(0, 2, "user", "audio", 12),
        TokenSegment(0, 3, "user", "video", 8),
    )
    assert report.text_tokens == 2
    assert report.image_tokens == 4
    assert report.video_tokens == 8
    assert report.audio_tokens == 12
    assert report.template_tokens == 24
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )
