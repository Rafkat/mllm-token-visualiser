from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.nemotron import NemotronAdapter


def test_nemotron_builds_all_modality_segments() -> None:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<image>": 41,
        "<so_embedding>": 42,
    }.__getitem__
    tokenizer.apply_chat_template.return_value = "processed prompt"
    processor = Mock(
        tokenizer=tokenizer,
        image_token="<image>",
        audio_token="<so_embedding>",
    )
    input_ids = torch.zeros((1, 80), dtype=torch.int64)
    input_ids[0, 10:18] = 41
    input_ids[0, 25:37] = 42
    input_ids[0, 50:57] = 41
    processor.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, 80), dtype=torch.int64),
    }
    adapter = NemotronAdapter(
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        processor,
        Mock(),
    )
    video = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    metadata = SimpleNamespace(
        fps=24.0,
        frames_indices=[0, 12],
        total_num_frames=24,
    )
    adapter._prepare_processor_inputs = Mock(
        return_value=([], ["first.jpg", "second.jpg"], ["audio.wav"], video, metadata)
    )
    adapter._count_video_tokens = Mock(return_value=20)
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)

    report = adapter.analyze(
        [
            Message.user(
                Text("Describe everything"),
                Image("first.jpg"),
                Audio("audio.wav"),
                Video("video.mp4"),
                Image("second.jpg"),
            )
        ]
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "text", 2),
        TokenSegment(0, 1, "user", "image", 8),
        TokenSegment(0, 2, "user", "audio", 12),
        TokenSegment(0, 3, "user", "video", 20),
        TokenSegment(0, 4, "user", "image", 7),
    )
    assert report.template_tokens == 31
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )


def test_nemotron_rejects_multiple_videos() -> None:
    adapter = NemotronAdapter("nemotron", Mock(tokenizer=Mock()), Mock())

    with pytest.raises(ValueError, match="only one video"):
        adapter.analyze([Message.user(Video("first.mp4"), Video("second.mp4"))])
