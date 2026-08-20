from dataclasses import FrozenInstanceError

import pytest

from mllm_tokens import TokenReport


def make_report(**overrides) -> TokenReport:
    values = {
        "model_id": "test-model",
        "total_tokens": 100,
        "text_tokens": 10,
        "image_tokens": 80,
        "video_tokens": 0,
        "audio_tokens": 0,
        "template_tokens": 10,
        "token_id_bytes": 800,
        "kv_cache_bytes": 2 * 1024**2,
        "kv_cache_bytes_per_token": 1024,
    }
    values.update(overrides)
    return TokenReport(**values)


def test_kv_cache_mib() -> None:
    report = make_report()

    assert report.kv_cache_mib == 2.0


def test_kv_cache_mib_is_none() -> None:
    report = make_report(kv_cache_bytes=None)

    assert report.kv_cache_mib is None


def test_to_dict() -> None:
    report = make_report()
    result = report.to_dict()

    assert result["model_id"] == "test-model"
    assert result["total_tokens"] == 100
    assert result["kv_cache_bytes"] == 2 * 1024**2


def test_report_is_immutable() -> None:
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.total_tokens = 200
