from types import SimpleNamespace
from unittest.mock import Mock

import torch
from fakes import FakeTokenizer

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


def test_qwen3vl_report_has_spans_media_details_and_metadata() -> None:
    vocab = {
        "<|im_start|>": 1,
        "user": 2,
        "Describe": 3,
        "<|vision_start|>": 4,
        "<|vision_end|>": 5,
        "<0.5s>": 6,
        "<|image_pad|>": 7,
        "<|video_pad|>": 8,
        "<|im_end|>": 9,
    }
    sequence = (
        [1, 2, 3, 4] + [7] * 4 + [5, 6, 4] + [8] * 4 + [5, 6, 4] + [8] * 4 + [5, 9]
    )
    processor = SimpleNamespace(
        tokenizer=FakeTokenizer(vocab),
        image_token_id=7,
        video_token_id=8,
        image_processor=SimpleNamespace(merge_size=2, patch_size=16),
        video_processor=SimpleNamespace(
            merge_size=2, patch_size=16, temporal_patch_size=2
        ),
        apply_chat_template=Mock(
            return_value={
                "input_ids": torch.tensor([sequence]),
                "attention_mask": torch.ones((1, len(sequence)), dtype=torch.int64),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
            }
        ),
    )
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_size=16,
    )
    adapter = Qwen3VLAdapter("Qwen/Qwen3-VL-8B-Instruct", processor, config)

    report = adapter.analyze(
        [Message.user(Text("Describe"), Image("image.jpg"), Video("video.mp4"))]
    )

    assert report.warnings == ()
    assert [(s.kind, s.content_index, s.tokens) for s in report.spans] == [
        ("template", None, 2),
        ("text", 0, 1),
        ("template", None, 1),
        ("image", 1, 4),
        ("template", None, 3),
        ("video", 2, 4),
        ("template", 2, 3),
        ("video", 2, 4),
        ("template", None, 2),
    ]
    assert sum(span.tokens for span in report.spans) == report.total_tokens

    image, video = report.media
    assert (image.modality, image.tokens, image.placeholder_groups) == ("image", 4, 1)
    assert image.properties["grid_thw"] == [1, 4, 4]
    assert image.properties["processed_width"] == 64
    assert (video.tokens, video.overhead_tokens, video.placeholder_groups) == (8, 3, 2)
    assert video.properties["frame_groups"] == 2
    assert video.properties["frames_sampled"] == 4
    assert video.properties["tokens_per_frame"] == 2.0
    assert video.properties["tokens_per_frame_group"] == 4

    assert report.metadata["adapter"] == "Qwen3VLAdapter"
    assert report.metadata["kv_cache"] == {
        "cached_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "dtype_bytes": 2,
    }
    assert report.metadata["add_generation_prompt"] is True
    assert any("processor's default" in item for item in report.assumptions)
