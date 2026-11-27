"""Terminal rendering of a ``TokenReport`` with ``rich``.

Every token kind has a colour *and* a glyph, so the output stays readable
without colour (``NO_COLOR``, logs, colour-vision deficiencies). Colours are
the first four categorical slots of the project's reference chart palette,
dark-surface steps; template overhead is a neutral gray.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from mllm_tokens.report import TOKEN_KINDS, MediaDetail, TokenReport, TokenSpan


@dataclass(frozen=True, slots=True)
class KindStyle:
    color: str
    glyph: str


KIND_STYLES = {
    "text": KindStyle("#3987e5", "█"),
    "image": KindStyle("#d95926", "▓"),
    "video": KindStyle("#199e70", "▒"),
    "audio": KindStyle("#c98500", "░"),
    "template": KindStyle("#8a8984", "·"),
}


def render_report(
    report: TokenReport,
    *,
    width: int = 100,
    map_rows: int = 6,
    show_template: bool = False,
    max_template_spans: int = 40,
) -> RenderableType:
    """Build a renderable overview of ``report``.

    ``width`` is the number of cells in the composition bar and token map.
    ``show_template`` adds the decoded template spans (chat-template overhead).
    """
    parts: list[RenderableType] = [
        Rule(Text(report.model_id, style="bold"), align="left"),
        _summary(report),
        Text(),
        Text("Composition", style="bold"),
        _composition_bar(report, width),
        _composition_table(report),
        Text(),
        Text("Input items", style="bold"),
        _items_table(report),
        Text(),
        Text("Token map", style="bold"),
        _token_map(report, width, map_rows),
    ]

    if show_template:
        parts += [
            Text(),
            Text("Template spans", style="bold"),
            _template_table(report, max_template_spans),
        ]

    if report.warnings:
        parts += [Text(), Text("Warnings", style="bold")]
        parts += [Text(f"⚠ Warning: {warning}") for warning in report.warnings]

    if report.assumptions:
        parts += [Text(), Text("Assumptions", style="bold")]
        parts += [Text(f"• {item}", style="dim") for item in report.assumptions]

    return Group(*parts)


def print_report(
    report: TokenReport, *, console: Console | None = None, **options: Any
) -> None:
    """Print ``report`` to the terminal. Options are passed to ``render_report``."""
    console = console or Console()
    options.setdefault("width", max(20, min(console.width - 2, 120)))
    console.print(render_report(report, **options))


def _summary(report: TokenReport) -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Total tokens", f"{report.total_tokens:,}")
    table.add_row("input_ids", format_bytes(report.token_id_bytes))

    if report.kv_cache_bytes is not None:
        dtype = report.metadata.get("kv_cache_dtype", "")
        table.add_row(
            "KV cache (estimate)",
            f"{format_bytes(report.kv_cache_bytes)}  "
            f"({format_bytes(report.kv_cache_bytes_per_token)} per token, {dtype})",
        )

    adapter = report.metadata.get("adapter")
    if adapter:
        table.add_row(
            "Adapter", f"{adapter} / {report.metadata.get('processor_class')}"
        )

    return table


def _composition_bar(report: TokenReport, width: int) -> Text:
    counts = report.tokens_by_kind()
    cells = allocate_cells(counts, width)
    bar = Text()

    for kind in TOKEN_KINDS:
        style = KIND_STYLES[kind]
        bar.append(style.glyph * cells[kind], style=style.color)

    return bar


def _composition_table(report: TokenReport) -> Table:
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("Kind")
    table.add_column("Tokens", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("KV cache", justify="right")

    shares = report.token_shares()
    kv_bytes = report.kv_cache_bytes_by_kind() or {}

    for kind, tokens in report.tokens_by_kind().items():
        table.add_row(
            _swatch(kind),
            f"{tokens:,}",
            f"{shares[kind]:.1%}",
            format_bytes(kv_bytes[kind]) if kind in kv_bytes else "—",
        )

    return table


def _items_table(report: TokenReport) -> Table:
    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("Msg", justify="right")
    table.add_column("Role")
    table.add_column("Item")
    table.add_column("Tokens", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Details", overflow="fold")

    media = {(m.message_index, m.content_index): m for m in report.media}
    text_previews = _text_previews(report.spans)
    total = report.total_tokens or 1

    for segment in report.segments:
        key = (segment.message_index, segment.content_index)
        detail = media.get(key)

        if detail is not None:
            details = describe_media(detail)
        else:
            details = repr(_shorten(text_previews.get(key, ""), 48))

        table.add_row(
            str(segment.message_index),
            segment.role,
            _swatch(segment.modality),
            f"{segment.tokens:,}",
            f"{segment.tokens / total:.1%}",
            details,
        )

    return table


def _token_map(report: TokenReport, width: int, rows: int) -> RenderableType:
    if not report.spans:
        return Text("Token map unavailable (no spans; see warnings).", style="dim")

    total = report.spans[-1].end
    cells = min(width * rows, total)
    kinds = bucket_kinds(report.spans, cells)
    per_cell = total / cells
    label_width = len(f"{total:,}")

    lines = Text()
    for row_start in range(0, cells, width):
        position = int(row_start * per_cell)
        lines.append(f"{position:>{label_width},} ", style="dim")
        for kind, count in _runs(kinds[row_start : row_start + width]):
            style = KIND_STYLES[kind]
            lines.append(style.glyph * count, style=style.color)
        lines.append("\n")

    legend = Text("  ".join(f"{KIND_STYLES[k].glyph} {k}" for k in TOKEN_KINDS))
    scale = Text(
        f"1 cell ≈ {per_cell:,.1f} tokens; each cell shows its majority kind, "
        "so short spans may be hidden.",
        style="dim",
    )
    return Group(lines, legend, scale)


def _template_table(report: TokenReport, limit: int) -> RenderableType:
    template_spans = [span for span in report.spans if span.kind == "template"]
    if not template_spans:
        return Text("No template spans.", style="dim")

    table = Table(box=None, padding=(0, 2), show_edge=False)
    table.add_column("Positions", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Belongs to")
    table.add_column("Decoded text", overflow="fold")

    for span in template_spans[:limit]:
        owner = (
            "conversation"
            if span.message_index is None
            else f"item {span.message_index}.{span.content_index}"
        )
        table.add_row(
            f"{span.start:,}–{span.end:,}",
            f"{span.tokens:,}",
            owner,
            repr(_shorten(span.text or "", 60)),
        )

    hidden = len(template_spans) - limit
    if hidden > 0:
        return Group(table, Text(f"… {hidden} more template spans", style="dim"))

    return table


def allocate_cells(counts: dict[str, int], width: int) -> dict[str, int]:
    """Split ``width`` cells between kinds proportionally.

    Uses largest remainders, and gives every non-zero kind at least one cell
    when the width allows it, so small kinds stay visible in the bar.
    """
    total = sum(counts.values())
    if total == 0 or width <= 0:
        return dict.fromkeys(counts, 0)

    exact = {kind: count * width / total for kind, count in counts.items()}
    cells = {kind: int(value) for kind, value in exact.items()}

    nonzero = [kind for kind, count in counts.items() if count > 0]
    if len(nonzero) <= width:
        for kind in nonzero:
            cells[kind] = max(cells[kind], 1)

    while sum(cells.values()) > width:
        largest = max(cells, key=lambda kind: (cells[kind], -exact[kind]))
        cells[largest] -= 1

    remainders = sorted(
        counts, key=lambda kind: exact[kind] - int(exact[kind]), reverse=True
    )
    index = 0
    while sum(cells.values()) < width:
        kind = remainders[index % len(remainders)]
        if counts[kind] > 0:
            cells[kind] += 1
        index += 1

    return cells


def bucket_kinds(spans: Sequence[TokenSpan], cells: int) -> list[str]:
    """Return the majority token kind of each of ``cells`` equal buckets."""
    if not spans or cells <= 0:
        return []

    total = spans[-1].end
    result = []
    span_index = 0

    for cell in range(cells):
        start = cell * total / cells
        end = (cell + 1) * total / cells
        weights: dict[str, float] = {}

        while span_index < len(spans) and spans[span_index].end <= start:
            span_index += 1

        index = span_index
        while index < len(spans) and spans[index].start < end:
            span = spans[index]
            overlap = min(end, span.end) - max(start, span.start)
            if overlap > 0:
                weights[span.kind] = weights.get(span.kind, 0.0) + overlap
            index += 1

        result.append(max(weights, key=weights.get) if weights else "template")

    return result


def describe_media(detail: MediaDetail) -> str:
    """Summarise a media item's properties in one line."""
    props = detail.properties
    parts = []

    if "width" in props and "height" in props:
        size = f"{props['width']}×{props['height']}"
        if "processed_width" in props and "processed_height" in props:
            size += f" → {props['processed_width']}×{props['processed_height']}"
        parts.append(size)

    if props.get("duration_seconds") is not None:
        parts.append(f"{props['duration_seconds']:g} s")

    if "sample_rate" in props:
        parts.append(f"{props['sample_rate']:,} Hz")

    if "frames_sampled" in props:
        parts.append(f"{props['frames_sampled']} frames")

    if "slices" in props:
        parts.append(f"{props['slices']} slice(s)")

    if "grid_thw" in props:
        parts.append("grid " + "×".join(str(v) for v in props["grid_thw"]))

    if "tokens_per_second" in props:
        parts.append(f"{props['tokens_per_second']:g} tok/s")

    if "tokens_per_frame" in props:
        parts.append(f"{props['tokens_per_frame']:g} tok/frame")

    if detail.overhead_tokens:
        parts.append(f"+{detail.overhead_tokens:,} overhead")

    if "probe_error" in props:
        parts.append(f"unreadable: {props['probe_error']}")

    return ", ".join(parts) or detail.source


def format_bytes(value: int | None) -> str:
    if value is None:
        return "—"

    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1024

    raise AssertionError("unreachable")


def _swatch(kind: str) -> Text:
    style = KIND_STYLES.get(kind, KIND_STYLES["template"])
    swatch = Text(style.glyph * 2, style=style.color)
    swatch.append(f" {kind}")
    return swatch


def _text_previews(spans: Sequence[TokenSpan]) -> dict[tuple[int, int], str]:
    previews: dict[tuple[int, int], str] = {}
    for span in spans:
        if span.kind == "text" and span.message_index is not None:
            key = (span.message_index, span.content_index)
            previews[key] = previews.get(key, "") + (span.text or "")
    return previews


def _runs(kinds: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for kind in kinds:
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + 1)
        else:
            runs.append((kind, 1))
    return runs


def _shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
