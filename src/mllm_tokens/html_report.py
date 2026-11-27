"""Self-contained HTML rendering of a ``TokenReport``.

The page has no external assets: styles, script, and data are inlined, so the
file can be opened offline or attached to an issue. Media thumbnails are only
embedded when ``embed_media=True``; by default the page contains token counts,
file metadata, and decoded *template* text, never the media itself.

Every token kind has a colour and its own lane in the token map, so identity
never depends on colour alone. Colours are the first four categorical slots of
the project's reference chart palette (light and dark steps); template
overhead is a neutral gray.
"""

import base64
import html
import io
import json
from importlib import resources
from pathlib import Path
from typing import Any

from mllm_tokens.display import describe_media, format_bytes
from mllm_tokens.report import TOKEN_KINDS, TokenReport

THUMBNAIL_SIZE = 240


def render_html(
    report: TokenReport,
    *,
    title: str | None = None,
    embed_media: bool = False,
    max_table_spans: int = 2000,
) -> str:
    """Return a standalone HTML document for ``report``.

    ``embed_media`` embeds small JPEG thumbnails of image inputs read from
    their source files. Leave it off for reports that will be shared.
    ``max_table_spans`` caps the rows of the span tables; the token map always
    shows every span.
    """
    payload = _payload(report, embed_media=embed_media, max_spans=max_table_spans)
    page_title = title or f"Token report: {report.model_id}"

    template = resources.files("mllm_tokens").joinpath("templates/report.html")
    return (
        template.read_text(encoding="utf-8")
        .replace("__TITLE__", html.escape(page_title))
        .replace("__PAYLOAD__", _script_json(payload))
        .replace("__REPORT_JSON__", _script_json(json.loads(report.to_json())))
    )


def save_html(report: TokenReport, path: str | Path, **options: Any) -> Path:
    """Write ``render_html(report, **options)`` to ``path`` and return it."""
    path = Path(path)
    path.write_text(render_html(report, **options), encoding="utf-8")
    return path


def _payload(report: TokenReport, *, embed_media: bool, max_spans: int) -> dict:
    counts = report.tokens_by_kind()
    shares = report.token_shares()
    kv_by_kind = report.kv_cache_bytes_by_kind() or {}
    total = report.total_tokens or 1

    media = {(m.message_index, m.content_index): m for m in report.media}
    previews: dict[tuple[int, int], str] = {}
    for span in report.spans:
        if span.kind == "text" and span.message_index is not None:
            key = (span.message_index, span.content_index)
            previews[key] = previews.get(key, "") + (span.text or "")

    items = []
    for segment in report.segments:
        key = (segment.message_index, segment.content_index)
        detail = media.get(key)
        item: dict[str, Any] = {
            "message": segment.message_index,
            "content": segment.content_index,
            "role": segment.role,
            "kind": segment.modality,
            "tokens": segment.tokens,
            "share": segment.tokens / total,
        }
        if detail is None:
            item["details"] = previews.get(key, "")
        else:
            item["details"] = describe_media(detail)
            item["source"] = Path(detail.source).name
            item["overhead"] = detail.overhead_tokens
            item["properties"] = {
                name: value
                for name, value in detail.properties.items()
                if isinstance(value, (int, float, str, bool, list)) or value is None
            }
            if embed_media and detail.modality == "image":
                thumbnail = _thumbnail(detail.source)
                if thumbnail:
                    item["thumbnail"] = thumbnail
        items.append(item)

    kv_bytes = report.kv_cache_bytes
    metadata = report.metadata

    return {
        "model_id": report.model_id,
        "summary": {
            "total_tokens": report.total_tokens,
            "token_id_bytes": format_bytes(report.token_id_bytes),
            "kv_cache": None if kv_bytes is None else format_bytes(kv_bytes),
            "kv_cache_per_token": (
                None
                if report.kv_cache_bytes_per_token is None
                else format_bytes(report.kv_cache_bytes_per_token)
            ),
            "kv_cache_dtype": metadata.get("kv_cache_dtype"),
            "media_items": len(report.media),
        },
        "composition": [
            {
                "kind": kind,
                "tokens": counts[kind],
                "share": shares[kind],
                "kv_cache": (
                    format_bytes(kv_by_kind[kind]) if kind in kv_by_kind else None
                ),
            }
            for kind in TOKEN_KINDS
        ],
        "items": items,
        "spans": [
            [
                span.start,
                span.end,
                span.kind,
                _owner(span.message_index, span.content_index),
                span.text,
                span.exact,
            ]
            for span in report.spans
        ],
        "max_table_spans": max_spans,
        "metadata": {
            name: value
            for name, value in metadata.items()
            if not isinstance(value, dict)
        }
        | {
            f"kv_cache.{name}": value
            for name, value in (metadata.get("kv_cache") or {}).items()
        },
        "warnings": list(report.warnings),
        "assumptions": list(report.assumptions),
        "embed_media": embed_media,
    }


def _owner(message_index: int | None, content_index: int | None) -> str | None:
    if message_index is None:
        return None
    return f"message {message_index}, item {content_index}"


def _thumbnail(source: str) -> str | None:
    path = Path(source)
    if not path.is_file():
        return None

    try:
        import PIL.Image as PImage

        with PImage.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
    except Exception:  # noqa: BLE001 - a thumbnail is optional decoration
        return None

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _script_json(value: Any) -> str:
    """JSON that is safe inside a ``<script type="application/json">`` tag."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )
