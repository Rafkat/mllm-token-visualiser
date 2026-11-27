"""Map every active sequence position to the input item that produced it.

The span builder is model-independent. Adapters supply:

- the active (unpadded) ``input_ids``;
- one boolean mask per media modality marking placeholder positions;
- per-item media token counts (the media ``TokenSegment`` values), used to
  split a modality's placeholder positions between its items in order;
- the user ``Text`` items with their isolated token IDs.

Text items are located in conversation order between the media items. A text
item is ``exact`` when its isolated token IDs occur verbatim in the sequence;
otherwise it is located by its characters (ignoring whitespace) and marked
approximate, which happens when the tokenizer merges text with neighbouring
template characters. Everything else is template.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from mllm_tokens.report import TokenSegment, TokenSpan

_Owner = tuple[str, int | None, int | None, str | None, bool]


class SpanAlignmentError(ValueError):
    """Placeholder positions cannot be matched with the input items."""


@dataclass(frozen=True, slots=True)
class TextSource:
    message_index: int
    content_index: int
    role: str
    text: str
    token_ids: tuple[int, ...]


def build_spans(
    active_ids: Sequence[int],
    media_masks: Mapping[str, Sequence[bool]],
    media_segments: Sequence[TokenSegment],
    text_sources: Sequence[TextSource],
    decode: Callable[[list[int]], str],
) -> tuple[tuple[TokenSpan, ...], tuple[str, ...]]:
    """Return ``(spans, warnings)`` covering every active position once."""
    size = len(active_ids)
    owners: list[_Owner | None] = [None] * size
    warnings: list[str] = []

    _assign_media(owners, media_masks, media_segments)

    unmapped = sorted(
        {
            segment.modality
            for segment in media_segments
            if segment.tokens and segment.modality not in media_masks
        }
    )
    if unmapped:
        warnings.append(
            "No placeholder positions available for: " + ", ".join(unmapped)
        )

    extents = _media_extents(owners)
    warnings.extend(
        _assign_text(owners, active_ids, extents, media_segments, text_sources, decode)
    )
    _attribute_media_overhead(owners, extents)

    return _compress(owners, active_ids, decode), tuple(warnings)


def _assign_media(
    owners: list[_Owner | None],
    media_masks: Mapping[str, Sequence[bool]],
    media_segments: Sequence[TokenSegment],
) -> None:
    for modality, mask in media_masks.items():
        if len(mask) != len(owners):
            raise SpanAlignmentError(
                f"{modality} mask has {len(mask)} positions, "
                f"but the sequence has {len(owners)}."
            )

        segments = [s for s in media_segments if s.modality == modality]
        positions = [index for index, flag in enumerate(mask) if flag]
        expected = sum(segment.tokens for segment in segments)

        if len(positions) != expected:
            raise SpanAlignmentError(
                f"Found {len(positions)} {modality} placeholder positions, "
                f"but {modality} segments account for {expected} tokens."
            )

        position_iter = iter(positions)
        for segment in segments:
            owner: _Owner = (
                modality,
                segment.message_index,
                segment.content_index,
                segment.role,
                True,
            )
            for _ in range(segment.tokens):
                index = next(position_iter)
                if owners[index] is not None:
                    raise SpanAlignmentError(
                        f"Position {index} is claimed by more than one modality."
                    )
                owners[index] = owner


def _media_extents(
    owners: Sequence[_Owner | None],
) -> dict[tuple[int, int], tuple[int, int]]:
    extents: dict[tuple[int, int], tuple[int, int]] = {}

    for index, owner in enumerate(owners):
        if owner is None:
            continue
        key = (owner[1], owner[2])
        first, _ = extents.get(key, (index, index))
        extents[key] = (first, index)

    return extents


def _assign_text(
    owners: list[_Owner | None],
    active_ids: Sequence[int],
    extents: Mapping[tuple[int, int], tuple[int, int]],
    media_segments: Sequence[TokenSegment],
    text_sources: Sequence[TextSource],
    decode: Callable[[list[int]], str],
) -> list[str]:
    warnings = []
    items: list[tuple[tuple[int, int], TextSource | None]] = [
        ((s.message_index, s.content_index), None) for s in media_segments
    ]
    items += [((s.message_index, s.content_index), s) for s in text_sources]
    items.sort(key=lambda item: item[0])

    cursor = 0
    for key, source in items:
        if source is None:
            if key in extents:
                cursor = max(cursor, extents[key][1] + 1)
            continue

        if not source.text.strip():
            continue

        limit = min(
            (first for other, (first, _) in extents.items() if other > key),
            default=len(owners),
        )
        region = [i for i in range(cursor, limit) if owners[i] is None]

        located = _find_exact(region, active_ids, source.token_ids)
        exact = located is not None
        if located is None:
            located = _find_by_characters(region, active_ids, source.text, decode)

        if located is None:
            warnings.append(
                f"Could not locate text item (message {key[0]}, content {key[1]}) "
                "in the processed sequence."
            )
            continue

        owner: _Owner = ("text", key[0], key[1], source.role, exact)
        for index in located:
            owners[index] = owner
        cursor = located[-1] + 1

    return warnings


def _find_exact(
    region: Sequence[int],
    active_ids: Sequence[int],
    token_ids: Sequence[int],
) -> list[int] | None:
    width = len(token_ids)
    if width == 0:
        return None

    region_ids = [active_ids[i] for i in region]
    target = list(token_ids)
    for offset in range(len(region_ids) - width + 1):
        if region_ids[offset : offset + width] == target:
            window = region[offset : offset + width]
            if window[-1] - window[0] == width - 1:
                return list(window)
    return None


def _find_by_characters(
    region: Sequence[int],
    active_ids: Sequence[int],
    text: str,
    decode: Callable[[list[int]], str],
) -> list[int] | None:
    target = "".join(text.split())
    if not target:
        return None

    chars: list[str] = []
    char_positions: list[int] = []
    for index in region:
        piece = decode([active_ids[index]])
        if not isinstance(piece, str):
            return None
        for char in piece:
            if not char.isspace():
                chars.append(char)
                char_positions.append(index)

    found = "".join(chars).find(target)
    if found < 0:
        return None

    first = char_positions[found]
    last = char_positions[found + len(target) - 1]
    return [i for i in region if first <= i <= last]


def _attribute_media_overhead(
    owners: list[_Owner | None],
    extents: Mapping[tuple[int, int], tuple[int, int]],
) -> None:
    for (message_index, content_index), (first, last) in extents.items():
        role = owners[first][3]
        for index in range(first, last + 1):
            if owners[index] is None:
                owners[index] = ("template", message_index, content_index, role, True)


def _compress(
    owners: Sequence[_Owner | None],
    active_ids: Sequence[int],
    decode: Callable[[list[int]], str],
) -> tuple[TokenSpan, ...]:
    template: _Owner = ("template", None, None, None, True)
    spans = []
    start = 0

    for index in range(1, len(owners) + 1):
        current = owners[start] or template
        if index < len(owners) and (owners[index] or template) == current:
            continue

        kind, message_index, content_index, role, exact = current
        text = None
        if kind in ("text", "template"):
            decoded = decode(list(active_ids[start:index]))
            text = decoded if isinstance(decoded, str) else None

        spans.append(
            TokenSpan(
                start=start,
                end=index,
                kind=kind,
                message_index=message_index,
                content_index=content_index,
                role=role,
                text=text,
                exact=exact,
            )
        )
        start = index

    return tuple(spans)
