"""Opt-in checks against real Hugging Face processors.

Downloads processor and configuration files (no model weights). Enable with
``MLLM_TOKENS_INTEGRATION=1``; remote-code models additionally require
``MLLM_TOKENS_TRUST_REMOTE_CODE=1``.
"""

import os
from pathlib import Path

import pytest

from mllm_tokens import Analyzer, Audio, Image, Message, Text, Video

ASSETS = Path(__file__).resolve().parents[2] / "examples" / "assets"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.media,
    pytest.mark.skipif(
        os.environ.get("MLLM_TOKENS_INTEGRATION") != "1",
        reason="set MLLM_TOKENS_INTEGRATION=1 to run real-processor checks",
    ),
]

IMAGE = Image(ASSETS / "image_sample.jpg")
AUDIO = Audio(ASSETS / "audio_sample.wav")
VIDEO = Video(ASSETS / "video_sample.mp4")

CASES = [
    ("Qwen/Qwen3-VL-8B-Instruct", False, [IMAGE, VIDEO]),
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False, [IMAGE, AUDIO, VIDEO]),
    ("google/gemma-4-e4b-it", False, [IMAGE, AUDIO, VIDEO]),
    ("openbmb/MiniCPM-o-4_5", True, [IMAGE, AUDIO]),
    ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", True, [IMAGE, AUDIO]),
]


@pytest.mark.parametrize(("model_id", "remote_code", "media"), CASES)
def test_real_processor_report_is_consistent(model_id, remote_code, media) -> None:
    if remote_code and os.environ.get("MLLM_TOKENS_TRUST_REMOTE_CODE") != "1":
        pytest.skip("set MLLM_TOKENS_TRUST_REMOTE_CODE=1 for remote-code processors")

    analyzer = Analyzer.from_pretrained(model_id, trust_remote_code=remote_code)
    report = analyzer.analyze(
        [Message.user(Text("First look:"), *media, Text("Describe everything."))]
    )

    assert report.warnings == ()
    assert sum(span.tokens for span in report.spans) == report.total_tokens
    for kind, tokens in report.tokens_by_kind().items():
        if kind in ("image", "video", "audio"):
            assert sum(s.tokens for s in report.spans if s.kind == kind) == tokens

    assert len(report.media) == len(media)
    for detail in report.media:
        assert detail.tokens > 0
        assert "probe_error" not in detail.properties


def test_nemotron_video_layout_matches_model() -> None:
    if os.environ.get("MLLM_TOKENS_TRUST_REMOTE_CODE") != "1":
        pytest.skip("set MLLM_TOKENS_TRUST_REMOTE_CODE=1 for remote-code processors")

    analyzer = Analyzer.from_pretrained(
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", trust_remote_code=True
    )
    report = analyzer.analyze([Message.user(VIDEO, Text("Describe the video."))])

    (detail,) = report.media
    frames = detail.properties["frames_sampled"]
    assert report.video_tokens == frames * 256
    assert detail.properties["tokens_after_evs"] < report.video_tokens
    assert sum(span.tokens for span in report.spans) == report.total_tokens

    # Timestamps follow the 2 fps sampling rate, not the source frame rate.
    template = "".join(span.text or "" for span in report.spans)
    assert "Frame 2 sampled at 0.50 seconds" in template
    assert f"Frame {frames} sampled at {(frames - 1) / 2:.2f} seconds" in template


def test_qwen3omni_samples_video_at_two_fps() -> None:
    analyzer = Analyzer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    report = analyzer.analyze([Message.user(VIDEO, Text("Describe the video."))])

    (detail,) = report.media
    # 25.1 s at 2 fps -> 50 frames -> 25 groups of two (temporal patch size 2).
    assert detail.properties["frame_groups"] == 25
    assert report.total_tokens < 20_000


def test_gemma4_audio_support_follows_config_not_model_id(tmp_path) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    from mllm_tokens.adapters.gemma4 import Gemma4Adapter

    # A local copy under a name without "e4b", as a fine-tune would have.
    snapshot = snapshot_download(
        "google/gemma-4-e4b-it", allow_patterns=["*.json", "*.jinja", "*.model"]
    )
    local = tmp_path / "my-gemma-finetune"
    local.symlink_to(snapshot)

    analyzer = Analyzer.from_pretrained(str(local))
    report = analyzer.analyze([Message.user(Text("Transcribe."), AUDIO)])

    assert report.metadata["adapter"] == "Gemma4Adapter"
    assert report.audio_tokens > 0
    assert report.warnings == ()

    # 26B-A4B and 31B have no audio encoder: config.audio_config is null.
    config = AutoConfig.from_pretrained("google/gemma-4-31B-it")
    adapter = Gemma4Adapter("google/gemma-4-31B-it", None, config)
    with pytest.raises(ValueError, match="config.audio_config is null"):
        adapter.analyze([Message.user(AUDIO)])
