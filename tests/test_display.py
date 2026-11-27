from rich.console import Console

from mllm_tokens import MediaDetail, TokenReport, TokenSegment, TokenSpan
from mllm_tokens.display import (
    KIND_STYLES,
    allocate_cells,
    bucket_kinds,
    describe_media,
    format_bytes,
    render_report,
)

SPANS = (
    TokenSpan(0, 2, "template", text="<s>user\n"),
    TokenSpan(2, 4, "text", 0, 0, "user", "Hello world"),
    TokenSpan(4, 5, "template", text="<img>"),
    TokenSpan(5, 13, "image", 0, 1, "user"),
    TokenSpan(13, 14, "template", 0, 1, "user", "</img>"),
    TokenSpan(14, 16, "image", 0, 1, "user"),
)


def make_report(**overrides) -> TokenReport:
    values = dict(
        model_id="org/model",
        total_tokens=16,
        text_tokens=2,
        image_tokens=10,
        video_tokens=0,
        audio_tokens=0,
        template_tokens=4,
        token_id_bytes=128,
        kv_cache_bytes=16 * 2048,
        kv_cache_bytes_per_token=2048,
        segments=(
            TokenSegment(0, 0, "user", "text", 2),
            TokenSegment(0, 1, "user", "image", 10),
        ),
        spans=SPANS,
        media=(
            MediaDetail(
                0,
                1,
                "user",
                "image",
                "a.jpg",
                10,
                overhead_tokens=1,
                placeholder_groups=2,
                properties={
                    "width": 387,
                    "height": 516,
                    "processed_width": 384,
                    "processed_height": 512,
                    "grid_thw": [1, 32, 24],
                },
            ),
        ),
        metadata={
            "adapter": "Qwen3VLAdapter",
            "processor_class": "Qwen3VLProcessor",
            "kv_cache_dtype": "bfloat16",
        },
        assumptions=("Images use processor defaults.",),
        warnings=("Something to check.",),
    )
    values.update(overrides)
    return TokenReport(**values)


def render_text(report: TokenReport, **options) -> str:
    console = Console(width=100, record=True, color_system=None)
    console.print(render_report(report, width=32, **options))
    return console.export_text()


def test_render_report_sections() -> None:
    output = render_text(make_report())

    assert "org/model" in output
    assert "Total tokens" in output and "16" in output
    assert "32.00 KiB  (2.00 KiB per token, bfloat16)" in output
    assert "Qwen3VLAdapter / Qwen3VLProcessor" in output
    assert "62.5%" in output  # image share
    assert "'Hello world'" in output
    assert "387×516 → 384×512, grid 1×32×24, +1 overhead" in output
    assert "1 cell ≈ 1.0 tokens" in output
    assert "⚠ Warning: Something to check." in output
    assert "• Images use processor defaults." in output
    assert "Template spans" not in output


def test_render_report_glyphs_identify_kinds_without_colour() -> None:
    output = render_text(make_report())

    # Composition bar: 32 cells split 2:10:4 between text, image, template.
    assert "█" * 4 + "▓" * 20 + "·" * 8 in output
    # Token map: one cell per token for a 16-token sequence.
    assert "··██·" + "▓" * 8 + "·▓▓" in output


def test_render_report_template_view() -> None:
    output = render_text(make_report(), show_template=True, max_template_spans=2)

    assert "Template spans" in output
    assert "'<s>user\\n'" in output
    assert "conversation" in output
    assert "… 1 more template spans" in output


def test_render_report_without_spans_or_kv_estimate() -> None:
    output = render_text(
        make_report(
            spans=(),
            media=(),
            kv_cache_bytes=None,
            kv_cache_bytes_per_token=None,
            warnings=(),
            assumptions=(),
        )
    )

    assert "Token map unavailable" in output
    assert "KV cache (estimate)" not in output
    assert "Warnings" not in output


def test_show_prints_to_console() -> None:
    console = Console(width=80, record=True, color_system=None)

    make_report().show(console=console, width=20)

    assert "org/model" in console.export_text()


def test_allocate_cells_is_proportional_and_keeps_small_kinds_visible() -> None:
    cells = allocate_cells({"text": 1, "image": 998, "template": 1}, 10)

    assert cells == {"text": 1, "image": 8, "template": 1}
    assert sum(allocate_cells({"a": 3, "b": 3, "c": 3}, 10).values()) == 10
    assert allocate_cells({"a": 0}, 10) == {"a": 0}


def test_allocate_cells_when_kinds_exceed_width() -> None:
    cells = allocate_cells({"a": 5, "b": 1, "c": 1}, 2)

    assert sum(cells.values()) == 2
    assert cells["a"] >= 1


def test_bucket_kinds_uses_majority_per_cell() -> None:
    spans = (
        TokenSpan(0, 3, "template"),
        TokenSpan(3, 10, "image"),
    )

    assert bucket_kinds(spans, 10) == ["template"] * 3 + ["image"] * 7
    # Cell 0 covers positions 0-5: three template and two image tokens.
    assert bucket_kinds(spans, 2) == ["template", "image"]
    assert bucket_kinds((), 5) == []


def test_describe_media_for_video_and_audio() -> None:
    video = MediaDetail(
        0,
        0,
        "user",
        "video",
        "v.mp4",
        11200,
        overhead_tokens=207,
        properties={
            "duration_seconds": 25.108,
            "frames_sampled": 50,
            "tokens_per_second": 446.07,
            "tokens_per_frame": 224.0,
        },
    )
    audio = MediaDetail(
        0,
        1,
        "user",
        "audio",
        "a.wav",
        625,
        properties={"duration_seconds": 25.0, "sample_rate": 8000},
    )

    assert describe_media(video) == (
        "25.108 s, 50 frames, 446.07 tok/s, 224 tok/frame, +207 overhead"
    )
    assert describe_media(audio) == "25 s, 8,000 Hz"
    assert describe_media(MediaDetail(0, 0, "user", "image", "x.jpg", 1)) == "x.jpg"


def test_format_bytes() -> None:
    assert format_bytes(None) == "—"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.00 KiB"
    assert format_bytes(3 * 1024**3) == "3.00 GiB"


def test_every_kind_has_distinct_glyph() -> None:
    glyphs = [style.glyph for style in KIND_STYLES.values()]

    assert len(set(glyphs)) == len(glyphs)


def test_describe_media_mentions_tokens_after_evs() -> None:
    detail = MediaDetail(
        0, 0, "user", "video", "v.mp4", 13056, properties={"tokens_after_evs": 3916}
    )

    assert describe_media(detail) == "3,916 after EVS"
