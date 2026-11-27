import csv
import io
import json
from dataclasses import FrozenInstanceError

import pytest

from mllm_tokens import MediaDetail, TokenReport, TokenSegment, TokenSpan


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


def test_breakdowns_by_kind() -> None:
    report = make_report()

    assert report.tokens_by_kind() == {
        "text": 10,
        "image": 80,
        "video": 0,
        "audio": 0,
        "template": 10,
    }
    assert report.token_shares()["image"] == 0.8
    assert report.kv_cache_bytes_by_kind()["image"] == 80 * 1024


def test_breakdowns_without_tokens_or_kv_estimate() -> None:
    report = make_report(total_tokens=0, kv_cache_bytes_per_token=None)

    assert set(report.token_shares().values()) == {0.0}
    assert report.kv_cache_bytes_by_kind() is None


def test_to_json_includes_schema_derived_values_and_accuracy() -> None:
    report = make_report(
        spans=(TokenSpan(0, 2, "text", 0, 0, "user", "Hi", True),),
        media=(
            MediaDetail(0, 1, "user", "image", "a.jpg", 80, properties={"width": 4}),
        ),
        metadata={"adapter": "Stub"},
        assumptions=("assumed",),
        warnings=("warned",),
    )

    payload = json.loads(report.to_json())

    assert payload["schema_version"] == "1"
    assert payload["spans"][0] == {
        "start": 0,
        "end": 2,
        "kind": "text",
        "message_index": 0,
        "content_index": 0,
        "role": "user",
        "text": "Hi",
        "exact": True,
    }
    assert payload["media"][0]["properties"] == {"width": 4}
    assert payload["derived"]["kv_cache_mib"] == 2.0
    assert payload["derived"]["token_shares"]["image"] == 0.8
    assert payload["field_accuracy"]["kv_cache_bytes"].startswith("estimate")
    assert payload["assumptions"] == ["assumed"]
    assert payload["warnings"] == ["warned"]


def test_csv_exports() -> None:
    report = make_report(
        segments=(TokenSegment(0, 0, "user", "text", 10),),
        spans=(TokenSpan(0, 3, "template", text='a,"b"\n'),),
    )

    segment_rows = list(csv.DictReader(io.StringIO(report.segments_csv())))
    span_rows = list(csv.DictReader(io.StringIO(report.spans_csv())))

    assert segment_rows == [
        {
            "message_index": "0",
            "content_index": "0",
            "role": "user",
            "modality": "text",
            "tokens": "10",
        }
    ]
    assert span_rows[0]["tokens"] == "3"
    assert span_rows[0]["text"] == 'a,"b"\n'
    assert span_rows[0]["message_index"] == ""


def test_print_dict_summarises_spans_and_media(capsys) -> None:
    report = make_report(spans=(TokenSpan(0, 100, "image", 0, 0, "user"),))

    report.print_dict()

    output = capsys.readouterr().out
    assert "total_tokens: 100" in output
    assert "spans: 1 spans (see report.spans)" in output
    assert "TokenSpan(" not in output
