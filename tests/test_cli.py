import json
from pathlib import Path

import pytest
import typer
from test_display import make_report
from typer.testing import CliRunner

from mllm_tokens import Audio, Image, Text, Video, analyzer
from mllm_tokens.cli import app, build_messages, comparison_rows, parse_content

ASSETS = Path(__file__).resolve().parents[1] / "examples" / "assets"
IMAGE = str(ASSETS / "image_sample.jpg")

runner = CliRunner()


class FakeAnalyzer:
    calls: list[dict] = []

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id, *, trust_remote_code=False):
        if model_id == "broken/model":
            raise ValueError("Unsupported model")
        cls.calls.append({"model": model_id, "trust_remote_code": trust_remote_code})
        return cls(model_id)

    def analyze(self, messages, *, add_generation_prompt, kv_cache_dtype):
        FakeAnalyzer.calls[-1] |= {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "kv_cache_dtype": kv_cache_dtype,
        }
        return make_report(model_id=self.model_id)


@pytest.fixture(autouse=True)
def fake_analyzer(monkeypatch):
    FakeAnalyzer.calls = []
    monkeypatch.setattr(analyzer, "Analyzer", FakeAnalyzer)


def test_parse_content_prefixes_paths_and_text() -> None:
    assert parse_content(f"image:{IMAGE}") == Image(IMAGE)
    assert parse_content(IMAGE) == Image(IMAGE)
    assert parse_content(str(ASSETS / "video_sample.mp4")) == Video(
        str(ASSETS / "video_sample.mp4")
    )
    assert parse_content(str(ASSETS / "audio_sample.wav")) == Audio(
        str(ASSETS / "audio_sample.wav")
    )
    assert parse_content("Describe this.") == Text("Describe this.")
    assert parse_content("text:cat.jpg") == Text("cat.jpg")
    assert parse_content("note: keep it short") == Text("note: keep it short")


def test_parse_content_rejects_missing_media() -> None:
    with pytest.raises(typer.BadParameter, match="image file not found"):
        parse_content("image:missing.jpg")
    with pytest.raises(typer.BadParameter, match="prefix with text:"):
        parse_content("missing.mp4")


def test_build_messages_keeps_order_and_system_prompt() -> None:
    messages = build_messages(["Look:", IMAGE, "Describe."], "Be brief.")

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == (Text("Look:"), Image(IMAGE), Text("Describe."))


def test_analyze_table_output() -> None:
    result = runner.invoke(app, ["analyze", "org/model", IMAGE, "Describe."])

    assert result.exit_code == 0, result.output
    assert "Total tokens" in result.output
    assert "Composition" in result.output
    call = FakeAnalyzer.calls[0]
    assert call["trust_remote_code"] is False
    assert call["add_generation_prompt"] is True
    assert call["kv_cache_dtype"] == "bfloat16"


def test_analyze_json_to_stdout_and_options() -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            "org/model",
            "Hello",
            "--format",
            "json",
            "--trust-remote-code",
            "--kv-dtype",
            "float16",
            "--no-generation-prompt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total_tokens"] == 16
    call = FakeAnalyzer.calls[0]
    assert call["trust_remote_code"] is True
    assert call["add_generation_prompt"] is False
    assert call["kv_cache_dtype"] == "float16"


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("report.html", "<!doctype html>"),
        ("report.json", '"schema_version"'),
        ("spans.csv", "start,end,tokens,kind"),
        ("report.txt", "Total tokens"),
    ],
)
def test_analyze_format_follows_output_suffix(tmp_path, name, marker) -> None:
    output = tmp_path / name

    result = runner.invoke(app, ["analyze", "org/model", "Hi", "-o", str(output)])

    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert marker in text
    assert "\x1b[" not in text  # no ANSI colour codes in files


def test_analyze_segments_csv_overrides_suffix(tmp_path) -> None:
    output = tmp_path / "out.csv"

    runner.invoke(
        app, ["analyze", "org/model", "Hi", "-f", "segments-csv", "-o", str(output)]
    )

    assert output.read_text().startswith("message_index,content_index,role")


def test_analyze_requires_content() -> None:
    result = runner.invoke(app, ["analyze", "org/model"])

    assert result.exit_code != 0


def test_compare_table_and_error_rows() -> None:
    result = runner.invoke(
        app,
        [
            "compare",
            IMAGE,
            "Describe.",
            "-m",
            "org/a",
            "-m",
            "org/b",
            "-m",
            "broken/model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Tokens by kind" in result.output
    assert "Tokens per input item" in result.output
    assert "org/a" in result.output and "org/b" in result.output
    assert "Error: broken/model: ValueError: Unsupported model" in result.output


def test_compare_json() -> None:
    result = runner.invoke(
        app, ["compare", "Hi", "-m", "org/a", "-m", "broken/model", "-f", "json"]
    )

    rows = json.loads(result.output)
    assert rows[0]["model"] == "org/a"
    assert rows[0]["image"] == 10
    assert rows[0]["items"] == [2, 10]
    assert rows[1]["error"] == "ValueError: Unsupported model"


def test_compare_csv_to_file(tmp_path) -> None:
    output = tmp_path / "compare.csv"

    result = runner.invoke(
        app,
        ["compare", "Hi", "-m", "org/a", "-m", "org/b", "-f", "csv", "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    lines = output.read_text().splitlines()
    assert (
        lines[0]
        == "model,total_tokens,text,image,video,audio,template,kv_cache_bytes,error"
    )
    assert lines[1].startswith("org/a,16,2,10,0,0,4,32768,")


def test_compare_needs_two_models() -> None:
    result = runner.invoke(app, ["compare", "Hi", "-m", "org/a"])

    assert result.exit_code != 0


def test_compare_fails_when_every_model_fails() -> None:
    result = runner.invoke(
        app, ["compare", "Hi", "-m", "broken/model", "-m", "broken/model"]
    )

    assert result.exit_code == 1


def test_comparison_rows_shape() -> None:
    rows = comparison_rows([("a", make_report()), ("b", "boom")])

    assert rows[0]["total_tokens"] == 16 and rows[0]["error"] is None
    assert rows[1]["total_tokens"] is None and rows[1]["error"] == "boom"


def test_compare_shows_warning_counts_when_present(monkeypatch) -> None:
    def analyze(self, messages, **_):
        return make_report(model_id=self.model_id, warnings=("a", "b"))

    monkeypatch.setattr(FakeAnalyzer, "analyze", analyze)

    result = runner.invoke(app, ["compare", "Hi", "-m", "org/a", "-m", "org/b"])

    assert "Warnings" in result.output
    assert "⚠ 2" in result.output
