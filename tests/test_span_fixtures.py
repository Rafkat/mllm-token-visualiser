"""Replay span-builder inputs recorded from real processors.

Fixtures are produced by ``scripts/record_span_fixtures.py``; each one stores
the exact sequence, placeholder masks, media counts, and text items that a
real processor produced for the example assets, plus the spans returned when
the fixture was recorded.
"""

import gzip
import json
from pathlib import Path

import pytest

from mllm_tokens.report import TokenSegment
from mllm_tokens.spans import TextSource, build_spans

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "spans").glob("*.json.gz"))


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def replay(fixture: dict):
    active_ids = [
        value for value, count in fixture["active_ids_rle"] for _ in range(count)
    ]
    masks = {}
    for modality, ranges in fixture["mask_ranges"].items():
        mask = [False] * len(active_ids)
        for start, end in ranges:
            mask[start:end] = [True] * (end - start)
        masks[modality] = mask

    table = fixture["decode_table"]

    def decode(ids):
        return "".join(table.get(str(token_id), "") for token_id in ids)

    spans, warnings = build_spans(
        active_ids,
        masks,
        [TokenSegment(**segment) for segment in fixture["media_segments"]],
        [
            TextSource(**{**source, "token_ids": tuple(source["token_ids"])})
            for source in fixture["text_sources"]
        ],
        decode,
    )
    return active_ids, spans, warnings


@pytest.mark.parametrize("path", FIXTURES, ids=[p.name.split(".")[0] for p in FIXTURES])
def test_recorded_layout_replays_to_recorded_spans(path: Path) -> None:
    fixture = load(path)

    active_ids, spans, warnings = replay(fixture)

    assert [
        [s.kind, s.start, s.end, s.message_index, s.content_index, s.exact]
        for s in spans
    ] == fixture["expected_spans"]
    assert list(warnings) == fixture["expected_warnings"]
    assert sum(span.tokens for span in spans) == len(active_ids)

    counts = fixture["counts"]
    assert len(active_ids) == counts["total"]
    for kind in ("image", "video", "audio"):
        assert sum(s.tokens for s in spans if s.kind == kind) == counts[kind]

    located = {(s.message_index, s.content_index) for s in spans if s.kind == "text"}
    assert located == {
        (source["message_index"], source["content_index"])
        for source in fixture["text_sources"]
    }


def test_fixtures_are_present() -> None:
    assert len(FIXTURES) >= 1
