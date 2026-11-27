from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from fakes import FakeTokenizer

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.nemotron import NemotronAdapter, evs_retained_tokens


def test_nemotron_builds_all_modality_segments() -> None:
    tokenizer = Mock()
    specials = {"<img>": [19], "</img>": [20], "<video>": [43]}
    tokenizer.encode.side_effect = lambda text, **_: specials.get(text, text.split())
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<image>": 41,
        "<so_embedding>": 42,
    }.__getitem__
    tokenizer.apply_chat_template.return_value = "processed prompt"
    processor = Mock(
        tokenizer=tokenizer,
        image_token="<image>",
        audio_token="<so_embedding>",
        image_start_token="<img>",
        image_end_token="</img>",
        video_token="<video>",
    )
    input_ids = torch.zeros((1, 80), dtype=torch.int64)
    input_ids[0, 10:18] = 41
    input_ids[0, 25:37] = 42
    input_ids[0, 40:60] = 43
    input_ids[0, 60:67] = 41
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


def make_nemotron_adapter(vocab: dict[str, int]) -> NemotronAdapter:
    processor = SimpleNamespace(
        tokenizer=FakeTokenizer(vocab),
        image_token="<image>",
        image_token_id=18,
        audio_token="<so_embedding>",
        audio_token_id=27,
        video_token="<video>",
        image_start_token="<img>",
        image_end_token="</img>",
    )
    return NemotronAdapter("nemotron", processor, Mock())


def test_nemotron_video_mask_fallback_uses_img_block_interiors() -> None:
    adapter = make_nemotron_adapter({"<img>": 19, "</img>": 20, "<video>": 30})
    # Without a known video token ID: image block, then two fragmented frames.
    input_ids = torch.tensor([[19, 18, 18, 20, 5, 19, 31, 32, 20, 6, 19, 31, 32, 20]])

    masks = adapter._nemotron_media_masks(input_ids, has_video=True)

    assert masks["image"].nonzero().flatten().tolist() == [1, 2]
    assert masks["video"].nonzero().flatten().tolist() == [6, 7, 11, 12]
    assert not masks["audio"].any()


def make_fragmenting_adapter(**config) -> NemotronAdapter:
    """Adapter whose tokenizer splits ``<video>`` like the FP8 checkpoint."""
    vocab = {"<img>": 19, "</img>": 20, "<": 1, "video": 2, ">": 3}
    adapter = make_nemotron_adapter(vocab)
    adapter.processor.tokenizer.encode = lambda text, **_: (
        [1, 2, 3] if text == "<video>" else [vocab[text]]
    )
    adapter.processor.image_processor = SimpleNamespace(num_image_token=3)
    adapter.config = SimpleNamespace(**config)
    return adapter


def test_nemotron_rebuilds_fragmented_video_frames() -> None:
    adapter = make_fragmenting_adapter(video_context_token_id=99)
    # text, image block, two video frames whose placeholders are fragments
    # (one frame fragmented differently), trailing text.
    ids = [5, 19, 18, 18, 20, 7, 19, 1, 2, 3, 1, 20, 8, 19, 2, 3, 20, 9]
    input_ids = torch.tensor([ids + [0, 0]])
    attention_mask = torch.tensor([[True] * len(ids) + [False, False]])

    rebuilt_ids, rebuilt_mask, video_id, (warning,) = (
        adapter._normalize_video_placeholders(input_ids, attention_mask)
    )

    assert video_id == 99
    assert rebuilt_ids.tolist() == [
        [5, 19, 18, 18, 20, 7, 19, 99, 99, 99, 20, 8, 19, 99, 99, 99, 20, 9]
    ]
    assert rebuilt_mask.all()
    assert "encodes to 3 tokens" in warning
    assert "3 positions of that ID" in warning

    masks = adapter._nemotron_media_masks(
        rebuilt_ids, has_video=True, video_token_id=video_id
    )
    assert masks["video"].nonzero().flatten().tolist() == [7, 8, 9, 13, 14, 15]
    assert masks["image"].nonzero().flatten().tolist() == [2, 3]


def test_nemotron_keeps_processor_output_when_video_token_is_atomic() -> None:
    adapter = make_nemotron_adapter({"<img>": 19, "</img>": 20, "<video>": 30})
    input_ids = torch.tensor([[19, 30, 30, 20]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    result = adapter._normalize_video_placeholders(input_ids, attention_mask)

    assert result[0] is input_ids
    assert result[2:] == (30, ())


def test_nemotron_falls_back_to_fragments_without_video_context_id() -> None:
    adapter = make_fragmenting_adapter()
    input_ids = torch.tensor([[19, 1, 2, 3, 20]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    ids, _, video_id, (warning,) = adapter._normalize_video_placeholders(
        input_ids, attention_mask
    )

    assert ids is input_ids
    assert video_id is None
    assert "video_context_token_id" in warning
    assert "counts those fragments" in warning


def test_evs_retained_tokens_matches_model_formula() -> None:
    # 51 frames x 256 tokens at the checkpoint's default rate of 0.7.
    assert evs_retained_tokens(51 * 256, 51, 0.7) == int(51 * 256 * 0.3)
    # Never fewer than one frame's worth.
    assert evs_retained_tokens(256, 1, 0.7) == 256
    assert evs_retained_tokens(512, 2, 0.0) == 512


def test_nemotron_video_properties_record_sampling() -> None:
    adapter = make_nemotron_adapter({})
    adapter.processor.image_processor = SimpleNamespace(num_image_token=256)
    adapter.config = SimpleNamespace(video_pruning_rate=0.7)
    metadata = SimpleNamespace(
        fps=2.0, source_fps=25.0, frames_indices=[0, 12, 25], total_num_frames=50
    )

    properties = adapter._nemotron_media_properties(
        [Message.user(Image("a.jpg"), Video("v.mp4"))], metadata, [768]
    )

    assert properties == {
        (0, 0): {"tokens_per_tile": 256},
        (0, 1): {
            "frames_sampled": 3,
            "sample_fps": 2.0,
            "max_frames": 256,
            "evs_pruning_rate": 0.7,
            "tokens_after_evs": 256,
        },
    }


def write_test_video(path, *, fps: int, frames: int) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


@pytest.mark.media
@pytest.mark.parametrize(
    ("source_fps", "expected_fps", "expected_frames"),
    [(10, 2.0, 4), (1, 1.0, 2)],
)
def test_load_video_reports_sampling_rate_for_timestamps(
    tmp_path, source_fps, expected_fps, expected_frames
) -> None:
    # The processor stamps frame j at j / metadata.fps, so metadata.fps must
    # be the sampling rate: 2 fps from a 10 fps source, every frame from 1 fps.
    path = tmp_path / "clip.mp4"
    write_test_video(path, fps=source_fps, frames=2 * source_fps)

    video, metadata = NemotronAdapter.load_video(path)

    assert metadata.fps == expected_fps
    assert metadata.source_fps == source_fps
    assert len(metadata.frames_indices) == expected_frames == video.shape[0]
