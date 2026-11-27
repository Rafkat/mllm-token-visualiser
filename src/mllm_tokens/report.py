import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "1"

TOKEN_KINDS = ("text", "image", "video", "audio", "template")

# How each count in a report is obtained. Exported with ``to_json()`` so that
# consumers can distinguish processor-exact values from estimates.
FIELD_ACCURACY = {
    "total_tokens": "exact: active positions in the processor output",
    "image_tokens": "exact: image positions in the processor output",
    "video_tokens": "exact: video positions in the processor output",
    "audio_tokens": "exact: audio positions in the processor output",
    "text_tokens": (
        "tokenizer: each Text item tokenized on its own; merges at template "
        "boundaries can differ by a few tokens (see spans[].exact)"
    ),
    "template_tokens": "derived: total minus text and media tokens",
    "token_id_bytes": "exact: size of the input_ids tensor",
    "kv_cache_bytes": "estimate: architectural KV-cache model (see assumptions)",
    "kv_cache_bytes_per_token": "estimate: architectural KV-cache model",
}


@dataclass(frozen=True, slots=True)
class TokenSegment:
    message_index: int
    content_index: int
    role: str
    modality: str
    tokens: int


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A contiguous run of active sequence positions with a single origin.

    ``start``/``end`` are half-open indices into the active (unpadded)
    sequence. Media spans point at the media item that produced them.
    Template spans normally have no item; template positions that fall between
    the placeholders of one media item (for example frame timestamps or
    ``<vision_start>`` markers inside a video) point at that media item.
    ``text`` is the decoded text of text and template spans. ``exact`` is
    ``False`` when a text item could only be located approximately.
    """

    start: int
    end: int
    kind: str
    message_index: int | None = None
    content_index: int | None = None
    role: str | None = None
    text: str | None = None
    exact: bool = True

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class MediaDetail:
    """Per-item details for one image, video, or audio input.

    ``tokens`` are the item's placeholder positions. ``overhead_tokens`` are
    template positions between the item's first and last placeholder, such as
    per-frame timestamps. ``properties`` holds source-file metadata and
    processor-specific details; available keys depend on the model family.
    """

    message_index: int
    content_index: int
    role: str
    modality: str
    source: str
    tokens: int
    overhead_tokens: int = 0
    placeholder_groups: int = 0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenReport:
    model_id: str

    total_tokens: int
    text_tokens: int
    image_tokens: int
    video_tokens: int
    audio_tokens: int
    template_tokens: int

    token_id_bytes: int

    kv_cache_bytes: int | None = None
    kv_cache_bytes_per_token: int | None = None
    segments: tuple[TokenSegment, ...] = ()

    spans: tuple[TokenSpan, ...] = ()
    media: tuple[MediaDetail, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def kv_cache_mib(self) -> float | None:
        if self.kv_cache_bytes is None:
            return None

        return self.kv_cache_bytes / 1024**2

    def tokens_by_kind(self) -> dict[str, int]:
        return {
            "text": self.text_tokens,
            "image": self.image_tokens,
            "video": self.video_tokens,
            "audio": self.audio_tokens,
            "template": self.template_tokens,
        }

    def token_shares(self) -> dict[str, float]:
        """Return each kind's fraction of ``total_tokens``."""
        if self.total_tokens == 0:
            return dict.fromkeys(TOKEN_KINDS, 0.0)

        return {
            kind: tokens / self.total_tokens
            for kind, tokens in self.tokens_by_kind().items()
        }

    def kv_cache_bytes_by_kind(self) -> dict[str, int] | None:
        """Estimated KV-cache bytes attributed to each token kind."""
        if self.kv_cache_bytes_per_token is None:
            return None

        return {
            kind: tokens * self.kv_cache_bytes_per_token
            for kind, tokens in self.tokens_by_kind().items()
        }

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the report, derived breakdowns, and field accuracy."""
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            **self.to_dict(),
            "derived": {
                "kv_cache_mib": self.kv_cache_mib,
                "tokens_by_kind": self.tokens_by_kind(),
                "token_shares": self.token_shares(),
                "kv_cache_bytes_by_kind": self.kv_cache_bytes_by_kind(),
            },
            "field_accuracy": FIELD_ACCURACY,
        }

        return json.dumps(payload, indent=indent, default=str, ensure_ascii=False)

    def segments_csv(self) -> str:
        return _to_csv(
            ("message_index", "content_index", "role", "modality", "tokens"),
            (asdict(segment) for segment in self.segments),
        )

    def spans_csv(self) -> str:
        columns = (
            "start",
            "end",
            "tokens",
            "kind",
            "message_index",
            "content_index",
            "role",
            "exact",
            "text",
        )
        return _to_csv(
            columns,
            ({**asdict(span), "tokens": span.tokens} for span in self.spans),
        )

    def show(self, **options: Any) -> None:
        """Print a rich terminal overview; see ``mllm_tokens.display``.

        Options: ``width``, ``map_rows``, ``show_template``,
        ``max_template_spans``, and ``console``.
        """
        from mllm_tokens.display import print_report

        print_report(self, **options)

    def to_html(self, **options: Any) -> str:
        """Return a self-contained HTML report; see ``mllm_tokens.html_report``.

        Options: ``title``, ``embed_media`` (default ``False``; embeds image
        thumbnails), and ``max_table_spans``.
        """
        from mllm_tokens.html_report import render_html

        return render_html(self, **options)

    def save_html(self, path: str | Path, **options: Any) -> Path:
        """Write the HTML report to ``path``. Options as for ``to_html``."""
        from mllm_tokens.html_report import save_html

        return save_html(self, path, **options)

    def print_dict(self) -> None:
        key_values = [
            (key, value)
            for key, value in self.to_dict().items()
            if key not in ("spans", "media")
        ]
        key_values.append(("spans", f"{len(self.spans)} spans (see report.spans)"))
        key_values.append(("media", f"{len(self.media)} items (see report.media)"))

        print("*" * 50)
        for i, (key, value) in enumerate(key_values):
            print(f"{key}: {value}")
            if i != len(key_values) - 1:
                print("-" * 50)
        print("*" * 50)


def _to_csv(columns: tuple[str, ...], rows) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
