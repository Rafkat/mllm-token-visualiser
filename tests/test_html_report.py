import json
import re
from pathlib import Path

import pytest
from test_display import make_report

from mllm_tokens import MediaDetail, TokenSpan
from mllm_tokens.html_report import render_html

ASSETS = Path(__file__).resolve().parents[1] / "examples" / "assets"


def script_json(page: str, element_id: str):
    match = re.search(
        rf'<script type="application/json" id="{element_id}">(.*?)</script>',
        page,
        re.DOTALL,
    )
    assert match, element_id
    return json.loads(match.group(1))


def test_html_is_self_contained() -> None:
    page = render_html(make_report())

    assert page.startswith("<!doctype html>")
    assert "<title>Token report: org/model</title>" in page
    # No external scripts, styles, fonts, or images.
    assert not re.search(r"""(src|href)=["']?(https?:)?//""", page)
    assert "@import" not in page
    assert "<link" not in page


def test_payload_carries_counts_spans_and_notes() -> None:
    payload = script_json(render_html(make_report()), "report-payload")

    composition = {part["kind"]: part["tokens"] for part in payload["composition"]}
    assert composition == {
        "text": 2,
        "image": 10,
        "video": 0,
        "audio": 0,
        "template": 4,
    }
    assert payload["spans"][1] == [
        2,
        4,
        "text",
        "message 0, item 0",
        "Hello world",
        True,
    ]
    assert payload["spans"][4][3] == "message 0, item 1"
    assert payload["items"][1]["details"].startswith("387×516 → 384×512")
    assert payload["items"][0]["details"] == "Hello world"
    assert payload["warnings"] == ["Something to check."]
    assert payload["summary"]["kv_cache"] == "32.00 KiB"
    assert payload["metadata"]["adapter"] == "Qwen3VLAdapter"


def test_full_report_json_is_embedded_for_download() -> None:
    report = make_report()
    embedded = script_json(render_html(report), "report-json")

    assert embedded == json.loads(report.to_json())


def test_untrusted_text_cannot_close_the_script_tag() -> None:
    hostile = "</script><script>alert(1)</script><!--"
    report = make_report(
        model_id="evil</title><script>alert(1)</script>",
        spans=(TokenSpan(0, 16, "template", text=hostile),),
        warnings=(hostile,),
    )

    page = render_html(report)

    assert page.count("<script") == 3  # two JSON blocks and the app script
    assert "<title>Token report: evil&lt;/title&gt;" in page
    payload = script_json(page, "report-payload")
    assert payload["spans"][0][4] == hostile
    assert payload["model_id"] == report.model_id


def test_media_is_not_embedded_by_default(tmp_path) -> None:
    report = image_report(ASSETS / "image_sample.jpg")

    payload = script_json(render_html(report), "report-payload")

    assert payload["embed_media"] is False
    assert "thumbnail" not in payload["items"][1]
    assert payload["items"][1]["source"] == "image_sample.jpg"
    assert "data:image" not in render_html(report)


@pytest.mark.media
def test_embed_media_adds_image_thumbnails() -> None:
    report = image_report(ASSETS / "image_sample.jpg")

    payload = script_json(render_html(report, embed_media=True), "report-payload")

    assert payload["items"][1]["thumbnail"].startswith("data:image/jpeg;base64,")


def test_embed_media_skips_missing_files() -> None:
    report = image_report(Path("does/not/exist.jpg"))

    payload = script_json(render_html(report, embed_media=True), "report-payload")

    assert "thumbnail" not in payload["items"][1]


def test_report_methods_write_html(tmp_path) -> None:
    report = make_report()

    path = report.save_html(tmp_path / "report.html", title="Custom")

    assert path.read_text(encoding="utf-8") == report.to_html(title="Custom")
    assert "<title>Custom</title>" in path.read_text(encoding="utf-8")


def test_report_without_spans_still_renders() -> None:
    payload = script_json(render_html(make_report(spans=())), "report-payload")

    assert payload["spans"] == []


def image_report(source: Path):
    base = make_report()
    detail = base.media[0]
    return make_report(
        media=(
            MediaDetail(
                detail.message_index,
                detail.content_index,
                detail.role,
                detail.modality,
                str(source),
                detail.tokens,
                properties=detail.properties,
            ),
        )
    )
