"""Command-line interface: ``mllm-tokens analyze`` and ``mllm-tokens compare``.

Content is given as positional arguments in conversation order. Each argument
is ``image:PATH``, ``video:PATH``, ``audio:PATH``, ``text:TEXT``, a path to an
existing file whose extension identifies the modality, or otherwise text.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mllm_tokens.display import format_bytes
from mllm_tokens.inputs import Audio, Image, Message, Text, Video
from mllm_tokens.report import TOKEN_KINDS, TokenReport

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES | AUDIO_SUFFIXES
PREFIXES = {"text": Text, "image": Image, "video": Video, "audio": Audio}


class AnalyzeFormat(StrEnum):
    table = "table"
    json = "json"
    html = "html"
    spans_csv = "spans-csv"
    segments_csv = "segments-csv"


class CompareFormat(StrEnum):
    table = "table"
    json = "json"
    csv = "csv"


app = typer.Typer(
    help="Token accounting for multimodal language models.",
    no_args_is_help=True,
    add_completion=False,
)

ContentArg = Annotated[
    list[str],
    typer.Argument(
        help=(
            "Content in conversation order: image:PATH, video:PATH, audio:PATH, "
            "text:TEXT, a media file path (modality from its extension), or text."
        ),
        show_default=False,
    ),
]
SystemOpt = Annotated[str | None, typer.Option("--system", help="System prompt text.")]
RemoteCodeOpt = Annotated[
    bool,
    typer.Option(
        "--trust-remote-code",
        help="Allow processors that run code from the model repository.",
    ),
]
DtypeOpt = Annotated[
    str, typer.Option("--kv-dtype", help="KV-cache dtype for the estimate.")
]
GenerationOpt = Annotated[
    bool,
    typer.Option(
        "--generation-prompt/--no-generation-prompt",
        help="Append the assistant generation prompt.",
    ),
]
OutputOpt = Annotated[
    Path | None,
    typer.Option(
        "--output", "-o", help="Write to this file instead of standard output."
    ),
]


@app.command()
def analyze(
    model: Annotated[str, typer.Argument(help="Hugging Face model ID or path.")],
    content: ContentArg,
    system: SystemOpt = None,
    output_format: Annotated[
        AnalyzeFormat | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format. Defaults to the --output suffix, else table.",
        ),
    ] = None,
    output: OutputOpt = None,
    trust_remote_code: RemoteCodeOpt = False,
    kv_dtype: DtypeOpt = "bfloat16",
    generation_prompt: GenerationOpt = True,
    show_template: Annotated[
        bool,
        typer.Option("--show-template", help="Table format: list template spans."),
    ] = False,
    embed_media: Annotated[
        bool,
        typer.Option(
            "--embed-media", help="HTML format: embed image thumbnails in the file."
        ),
    ] = False,
) -> None:
    """Analyze one conversation with one model."""
    messages = build_messages(content, system)
    report = _load_and_analyze(
        model, messages, trust_remote_code, kv_dtype, generation_prompt
    )
    fmt = output_format or _format_from_suffix(output) or AnalyzeFormat.table

    if fmt is AnalyzeFormat.table:
        _print_table_report(report, output, show_template=show_template)
        return

    renderers = {
        AnalyzeFormat.json: report.to_json,
        AnalyzeFormat.html: lambda: report.to_html(embed_media=embed_media),
        AnalyzeFormat.spans_csv: report.spans_csv,
        AnalyzeFormat.segments_csv: report.segments_csv,
    }
    _emit(renderers[fmt](), output)


@app.command()
def compare(
    content: ContentArg,
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model to compare; repeat for each model."),
    ],
    system: SystemOpt = None,
    output_format: Annotated[
        CompareFormat, typer.Option("--format", "-f", help="Output format.")
    ] = CompareFormat.table,
    output: OutputOpt = None,
    trust_remote_code: RemoteCodeOpt = False,
    kv_dtype: DtypeOpt = "bfloat16",
    generation_prompt: GenerationOpt = True,
) -> None:
    """Analyze the same conversation with several models.

    A model that fails (unsupported modality, download error) is reported as
    an error row; the others are still compared.
    """
    if len(models) < 2:
        raise typer.BadParameter("give at least two --model options", param_hint="-m")

    messages = build_messages(content, system)
    results: list[tuple[str, TokenReport | str]] = []
    for model in models:
        try:
            report = _load_and_analyze(
                model, messages, trust_remote_code, kv_dtype, generation_prompt
            )
        except Exception as error:  # noqa: BLE001 - shown per model, run continues
            results.append((model, f"{type(error).__name__}: {error}"))
        else:
            results.append((model, report))

    rows = comparison_rows(results)
    if output_format is CompareFormat.json:
        _emit(json.dumps(rows, indent=2, ensure_ascii=False), output)
    elif output_format is CompareFormat.csv:
        _emit(_rows_csv(rows), output)
    else:
        _print_comparison(results, rows, output)

    if all(row["error"] for row in rows):
        raise typer.Exit(code=1)


def parse_content(argument: str) -> Text | Image | Video | Audio:
    """Turn one positional argument into a content item."""
    prefix, separator, value = argument.partition(":")
    if separator and prefix in PREFIXES:
        if prefix != "text" and not Path(value).is_file():
            raise typer.BadParameter(f"{prefix} file not found: {value}")
        return PREFIXES[prefix](value)

    path = Path(argument)
    suffix = path.suffix.lower()
    if path.is_file():
        if suffix in IMAGE_SUFFIXES:
            return Image(argument)
        if suffix in VIDEO_SUFFIXES:
            return Video(argument)
        if suffix in AUDIO_SUFFIXES:
            return Audio(argument)
    elif suffix in MEDIA_SUFFIXES and " " not in argument:
        raise typer.BadParameter(
            f"file not found: {argument} (prefix with text: to pass it as text)"
        )

    return Text(argument)


def build_messages(content: list[str], system: str | None) -> list[Message]:
    if not content:
        raise typer.BadParameter("give at least one content item")

    messages = []
    if system:
        messages.append(Message.system(Text(system)))
    messages.append(Message.user(*(parse_content(item) for item in content)))
    return messages


def comparison_rows(results: list[tuple[str, TokenReport | str]]) -> list[dict]:
    rows = []
    for model, result in results:
        row: dict[str, Any] = {"model": model}
        if isinstance(result, str):
            row |= dict.fromkeys(("total_tokens", *TOKEN_KINDS, "kv_cache_bytes"))
            row |= {"items": None, "warnings": None, "error": result}
        else:
            row["total_tokens"] = result.total_tokens
            row |= result.tokens_by_kind()
            row["kv_cache_bytes"] = result.kv_cache_bytes
            row["items"] = [segment.tokens for segment in result.segments]
            row["warnings"] = len(result.warnings)
            row["error"] = None
        rows.append(row)
    return rows


def _load_and_analyze(
    model: str,
    messages: list[Message],
    trust_remote_code: bool,
    kv_dtype: str,
    generation_prompt: bool,
) -> TokenReport:
    from mllm_tokens.analyzer import Analyzer

    analyzer = Analyzer.from_pretrained(model, trust_remote_code=trust_remote_code)
    return analyzer.analyze(
        messages, add_generation_prompt=generation_prompt, kv_cache_dtype=kv_dtype
    )


def _format_from_suffix(output: Path | None) -> AnalyzeFormat | None:
    if output is None:
        return None
    return {
        ".json": AnalyzeFormat.json,
        ".html": AnalyzeFormat.html,
        ".htm": AnalyzeFormat.html,
        ".csv": AnalyzeFormat.spans_csv,
    }.get(output.suffix.lower())


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        typer.echo(text, nl=not text.endswith("\n"))
        return

    output.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {output}", err=True)


def _file_console(output: Path) -> Console:
    return Console(
        file=output.open("w", encoding="utf-8"), width=120, color_system=None
    )


def _print_table_report(
    report: TokenReport, output: Path | None, *, show_template: bool
) -> None:
    console = _file_console(output) if output else Console()
    try:
        report.show(console=console, show_template=show_template)
    finally:
        if output:
            console.file.close()
            typer.echo(f"Wrote {output}", err=True)


def _print_comparison(
    results: list[tuple[str, TokenReport | str]],
    rows: list[dict],
    output: Path | None,
) -> None:
    console = _file_console(output) if output else Console()
    try:
        console.print(_comparison_table(rows))
        items = _items_table(results)
        if items is not None:
            console.print()
            console.print(items)
        for row in rows:
            if row["error"]:
                console.print(f"✗ Error: {row['model']}: {row['error']}")
    finally:
        if output:
            console.file.close()
            typer.echo(f"Wrote {output}", err=True)


def _comparison_table(rows: list[dict]) -> Table:
    present = [
        k for k in TOKEN_KINDS if any(row[k] for row in rows if not row["error"])
    ]
    table = Table(title="Tokens by kind", title_justify="left", box=None)
    table.add_column("Model")
    table.add_column("Total", justify="right")
    for kind in present:
        table.add_column(kind.capitalize(), justify="right")
    table.add_column("KV cache (est.)", justify="right")
    show_warnings = any(row["warnings"] for row in rows)
    if show_warnings:
        table.add_column("Warnings", justify="right")

    for row in rows:
        if row["error"]:
            cells = ["error"] + ["—"] * (len(present) + 1 + show_warnings)
        else:
            cells = [f"{row['total_tokens']:,}"]
            cells += [f"{row[kind]:,}" for kind in present]
            cells.append(format_bytes(row["kv_cache_bytes"]))
            if show_warnings:
                cells.append(f"⚠ {row['warnings']}" if row["warnings"] else "0")
        table.add_row(row["model"], *cells)

    return table


def _items_table(results: list[tuple[str, TokenReport | str]]) -> Table | None:
    reports = [(model, r) for model, r in results if isinstance(r, TokenReport)]
    if not reports:
        return None

    first = reports[0][1]
    keys = [(s.message_index, s.content_index) for s in first.segments]

    table = Table(title="Tokens per input item", title_justify="left", box=None)
    table.add_column("Model")
    for segment in first.segments:
        table.add_column(
            f"{segment.message_index}.{segment.content_index} {segment.modality}",
            justify="right",
        )

    for model, report in reports:
        tokens = {(s.message_index, s.content_index): s.tokens for s in report.segments}
        table.add_row(
            model, *(f"{tokens[key]:,}" if key in tokens else "—" for key in keys)
        )

    return table


def _rows_csv(rows: list[dict]) -> str:
    import csv
    import io

    columns = ("model", "total_tokens", *TOKEN_KINDS, "kv_cache_bytes", "error")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    app()
