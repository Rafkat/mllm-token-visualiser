from unittest.mock import Mock

import torch

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.base import MediaSource
from mllm_tokens.adapters.minicpmo45 import MiniCPMo45Adapter


class ProcessorOutput(dict):
    def __getattr__(self, key):
        return self[key]


def test_minicpmo45_builds_all_modality_segments() -> None:
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    processor = Mock(tokenizer=tokenizer)
    processor.return_value = ProcessorOutput(
        input_ids=torch.zeros((1, 50), dtype=torch.int64),
        attention_mask=torch.ones((1, 50), dtype=torch.int64),
        image_bound=[torch.tensor([[5, 9], [12, 18], [20, 27]])],
        audio_bounds=[torch.tensor([[30, 40]])],
    )
    adapter = MiniCPMo45Adapter("openbmb/MiniCPM-o-4_5", processor, Mock())
    adapter._to_minicpm_messages = Mock(
        return_value=(
            {"text": "prompt", "images": None, "audios": None},
            [
                MediaSource(0, 1, "user", "image"),
                MediaSource(0, 3, "user", "video"),
                MediaSource(0, 3, "user", "video"),
            ],
            [MediaSource(0, 2, "user", "audio")],
        )
    )
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)

    report = adapter.analyze(
        [
            Message.user(
                Text("Describe everything"),
                Image("image.jpg"),
                Audio("audio.wav"),
                Video("video.mp4"),
            )
        ],
        add_generation_prompt=True,
        kv_cache_dtype="bfloat16",
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "text", 2),
        TokenSegment(0, 1, "user", "image", 4),
        TokenSegment(0, 2, "user", "audio", 10),
        TokenSegment(0, 3, "user", "video", 13),
    )
    assert report.template_tokens == 21
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )
