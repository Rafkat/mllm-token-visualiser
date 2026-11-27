import pytest

from mllm_tokens.report import TokenSegment, TokenSpan
from mllm_tokens.spans import SpanAlignmentError, TextSource, build_spans

# Tiny vocabulary: ids map to readable pieces so decoded text is checkable.
VOCAB = {
    1: "<s>",
    2: "user\n",
    3: "Hello",
    4: " world",
    5: "<img>",
    6: "</img>",
    7: "<pad>",
    8: "<t=0.5>",
    9: ":\n",
    10: ":",
    11: "Describe",
    12: "\n",
    13: "<aud>",
}


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


def text(message_index, content_index, value, ids, role="user") -> TextSource:
    return TextSource(message_index, content_index, role, value, tuple(ids))


def assert_covers(spans: tuple[TokenSpan, ...], size: int) -> None:
    assert spans[0].start == 0
    assert spans[-1].end == size
    for left, right in zip(spans, spans[1:], strict=False):
        assert left.end == right.start


def test_exact_text_media_and_template_spans() -> None:
    ids = [1, 2, 3, 4, 5, 7, 7, 7, 6, 11, 12]
    image_mask = [i == 7 for i in ids]

    spans, warnings = build_spans(
        ids,
        {"image": image_mask},
        [TokenSegment(0, 1, "user", "image", 3)],
        [text(0, 0, "Hello world", [3, 4]), text(0, 2, "Describe", [11])],
        decode,
    )

    assert warnings == ()
    assert_covers(spans, len(ids))
    assert [(s.kind, s.start, s.end, s.content_index) for s in spans] == [
        ("template", 0, 2, None),
        ("text", 2, 4, 0),
        ("template", 4, 5, None),
        ("image", 5, 8, 1),
        ("template", 8, 9, None),
        ("text", 9, 10, 2),
        ("template", 10, 11, None),
    ]
    assert spans[0].text == "<s>user\n"
    assert spans[1].text == "Hello world"
    assert all(span.exact for span in spans)


def test_text_merged_with_template_is_located_approximately() -> None:
    # The isolated text "Hello:" is [3, 10], but in context ":" merged with
    # the following newline into ":\n" (id 9).
    ids = [1, 3, 9, 5, 7, 6]

    spans, warnings = build_spans(
        ids,
        {"image": [i == 7 for i in ids]},
        [TokenSegment(0, 1, "user", "image", 1)],
        [text(0, 0, "Hello:", [3, 10])],
        decode,
    )

    assert warnings == ()
    text_span = next(span for span in spans if span.kind == "text")
    assert (text_span.start, text_span.end, text_span.exact) == (1, 3, False)


def test_media_item_split_across_runs_owns_interior_template() -> None:
    # One video with two frame groups separated by a timestamp marker.
    ids = [1, 5, 7, 7, 6, 8, 5, 7, 7, 6, 11]

    spans, warnings = build_spans(
        ids,
        {"video": [i == 7 for i in ids]},
        [TokenSegment(0, 0, "user", "video", 4)],
        [text(0, 1, "Describe", [11])],
        decode,
    )

    assert warnings == ()
    video_spans = [s for s in spans if s.kind == "video"]
    assert [(s.start, s.end) for s in video_spans] == [(2, 4), (7, 9)]
    overhead = [s for s in spans if s.kind == "template" and s.content_index == 0]
    assert [(s.start, s.end, s.text) for s in overhead] == [
        (4, 7, "</img><t=0.5><img>")
    ]


def test_adjacent_items_of_one_modality_are_split_by_counts() -> None:
    ids = [1, 7, 7, 7, 7, 7]

    spans, _ = build_spans(
        ids,
        {"audio": [i == 7 for i in ids]},
        [
            TokenSegment(0, 0, "user", "audio", 2),
            TokenSegment(0, 1, "user", "audio", 3),
        ],
        [],
        decode,
    )

    assert [(s.kind, s.content_index, s.tokens) for s in spans] == [
        ("template", None, 1),
        ("audio", 0, 2),
        ("audio", 1, 3),
    ]


def test_repeated_text_is_located_after_preceding_items() -> None:
    # "Hello" appears in the system prompt and again after the image.
    ids = [3, 12, 5, 7, 6, 3]

    spans, _ = build_spans(
        ids,
        {"image": [i == 7 for i in ids]},
        [TokenSegment(1, 0, "user", "image", 1)],
        [text(0, 0, "Hello", [3], role="system"), text(1, 1, "Hello", [3])],
        decode,
    )

    text_spans = [(s.start, s.message_index, s.role) for s in spans if s.kind == "text"]
    assert text_spans == [(0, 0, "system"), (5, 1, "user")]


def test_placeholder_count_mismatch_raises() -> None:
    ids = [1, 7, 7]

    with pytest.raises(SpanAlignmentError, match="Found 2 image placeholder"):
        build_spans(
            ids,
            {"image": [False, True, True]},
            [TokenSegment(0, 0, "user", "image", 3)],
            [],
            decode,
        )


def test_overlapping_modalities_raise() -> None:
    ids = [7, 7]

    with pytest.raises(SpanAlignmentError, match="more than one modality"):
        build_spans(
            ids,
            {"image": [True, False], "audio": [True, False]},
            [
                TokenSegment(0, 0, "user", "image", 1),
                TokenSegment(0, 1, "user", "audio", 1),
            ],
            [],
            decode,
        )


def test_mask_length_mismatch_raises() -> None:
    with pytest.raises(SpanAlignmentError, match="mask has 1 positions"):
        build_spans(
            [1, 2],
            {"image": [True]},
            [TokenSegment(0, 0, "user", "image", 1)],
            [],
            decode,
        )


def test_missing_mask_and_unlocated_text_are_warnings() -> None:
    ids = [1, 2, 13, 13]

    spans, warnings = build_spans(
        ids,
        {},
        [TokenSegment(0, 0, "user", "audio", 2)],
        [text(0, 1, "absent", [99])],
        decode,
    )

    assert spans == (TokenSpan(0, 4, "template", text="<s>user\n<aud><aud>"),)
    assert warnings == (
        "No placeholder positions available for: audio",
        "Could not locate text item (message 0, content 1) in the processed sequence.",
    )


def test_empty_sequence() -> None:
    assert build_spans([], {}, [], [], decode) == ((), ())


def test_whitespace_only_text_is_skipped() -> None:
    spans, warnings = build_spans([12], {}, [], [text(0, 0, "  ", [12])], decode)

    assert warnings == ()
    assert [span.kind for span in spans] == ["template"]
