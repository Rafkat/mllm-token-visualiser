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
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False, [IMAGE, AUDIO]),
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
