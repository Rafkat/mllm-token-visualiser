from unittest.mock import Mock

import torch

from mllm_tokens import Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter


def test_qwen3vl_builds_text_image_and_video_segments() -> None:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    processor = Mock(
        tokenizer=tokenizer,
        image_processor=Mock(merge_size=2),
        video_processor=Mock(merge_size=2),
    )
    processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 40), dtype=torch.int64),
        "attention_mask": torch.ones((1, 40), dtype=torch.int64),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    adapter = Qwen3VLAdapter("Qwen/Qwen3-VL-8B-Instruct", processor, Mock())
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)

    report = adapter.analyze(
        [
            Message.user(
                Text("Describe both inputs"),
                Image("image.jpg"),
                Video("video.mp4"),
            )
        ]
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "text", 3),
        TokenSegment(0, 1, "user", "image", 4),
        TokenSegment(0, 2, "user", "video", 8),
    )
    assert report.text_tokens == 3
    assert report.image_tokens == 4
    assert report.video_tokens == 8
    assert report.audio_tokens == 0
    assert report.template_tokens == 25
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )
