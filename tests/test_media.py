from pathlib import Path

import pytest

from mllm_tokens.media import probe_media

ASSETS = Path(__file__).resolve().parents[1] / "examples" / "assets"

pytestmark = pytest.mark.media


def test_probe_image() -> None:
    properties = probe_media(ASSETS / "image_sample.jpg", "image")

    assert properties["width"] > 0
    assert properties["height"] > 0
    assert properties["file_bytes"] > 0


def test_probe_video_reads_header_only() -> None:
    properties = probe_media(ASSETS / "video_sample.mp4", "video")

    assert properties["width"] > 0
    assert properties["duration_seconds"] > 0
    assert properties["source_fps"] > 0


def test_probe_audio() -> None:
    properties = probe_media(ASSETS / "audio_sample.wav", "audio")

    assert properties["sample_rate"] > 0
    assert properties["duration_seconds"] > 0


def test_probe_missing_file_returns_nothing() -> None:
    assert probe_media("does/not/exist.jpg", "image") == {}


def test_probe_unreadable_file_reports_error(tmp_path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    properties = probe_media(broken, "image")

    assert properties["file_bytes"] == 12
    assert properties["probe_error"].startswith("UnidentifiedImageError")
