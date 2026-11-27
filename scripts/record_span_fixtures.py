"""Record golden span-builder fixtures from real model processors.

For each model this runs the real processor on the example assets, captures
the exact inputs passed to ``mllm_tokens.spans.build_spans`` and the spans it
returned, and writes a compact gzip JSON fixture to ``tests/fixtures/spans``.
``tests/test_span_fixtures.py`` replays these fixtures offline.

Only processor/config files are downloaded; no model weights are loaded.
MiniCPM-o and Nemotron processors require ``trust_remote_code``.

Usage (from the repository root):
    python scripts/record_span_fixtures.py [model_key ...]
"""

import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mllm_tokens.adapters.base as base
from mllm_tokens import Analyzer, Audio, Image, Message, Text, Video

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples" / "assets"
OUT_DIR = ROOT / "tests" / "fixtures" / "spans"

IMAGE = Image(ASSETS / "image_sample.jpg")
AUDIO = Audio(ASSETS / "audio_sample.wav")
VIDEO = Video(ASSETS / "video_sample.mp4")

# (model ID, trust_remote_code, media). MiniCPM-o video needs the ffmpeg CLI.
MODELS = {
    "qwen3vl": ("Qwen/Qwen3-VL-8B-Instruct", False, [IMAGE, VIDEO]),
    "qwen3omni": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", False, [IMAGE, AUDIO, VIDEO]),
    "gemma4": ("google/gemma-4-e4b-it", False, [IMAGE, AUDIO, VIDEO]),
    "minicpmo45": ("openbmb/MiniCPM-o-4_5", True, [IMAGE, AUDIO]),
    "nemotron": (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        True,
        [IMAGE, AUDIO, VIDEO],
    ),
}


def conversation(media) -> list[Message]:
    return [
        Message.system(Text("You are a helpful assistant.")),
        Message.user(
            Text("First look:"),
            media[0],
            Text("then"),
            *media[1:],
            Text("Describe everything."),
        ),
    ]


def rle(values) -> list[list]:
    runs: list[list] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def ranges(mask) -> list[list[int]]:
    result, start = [], None
    for index, flag in enumerate([*mask, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            result.append([start, index])
            start = None
    return result


def record(key: str) -> None:
    model_id, remote_code, media = MODELS[key]
    captured = {}
    original = base.build_spans

    def capture(active_ids, masks, media_segments, text_sources, decode):
        spans, warnings = original(
            active_ids, masks, media_segments, text_sources, decode
        )
        media_positions = {
            i for mask in masks.values() for i, f in enumerate(mask) if f
        }
        decode_table = {
            str(token_id): decode([token_id])
            for token_id in sorted(
                {t for i, t in enumerate(active_ids) if i not in media_positions}
            )
        }
        captured.update(
            active_ids_rle=rle(active_ids),
            mask_ranges={m: ranges(mask) for m, mask in masks.items()},
            media_segments=[asdict(s) for s in media_segments],
            text_sources=[asdict(s) for s in text_sources],
            decode_table=decode_table,
            expected_spans=[
                [s.kind, s.start, s.end, s.message_index, s.content_index, s.exact]
                for s in spans
            ],
            expected_warnings=list(warnings),
        )
        return spans, warnings

    base.build_spans = capture
    try:
        analyzer = Analyzer.from_pretrained(model_id, trust_remote_code=remote_code)
        report = analyzer.analyze(conversation(media))
    finally:
        base.build_spans = original

    fixture = {
        "model_id": model_id,
        "processor_class": report.metadata["processor_class"],
        "transformers_version": report.metadata["transformers_version"],
        "counts": report.tokens_by_kind() | {"total": report.total_tokens},
        "report_warnings": list(report.warnings),
        **captured,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{key}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(fixture, handle, ensure_ascii=False)

    inexact = sum(1 for s in report.spans if s.kind == "text" and not s.exact)
    print(
        f"{key}: total={report.total_tokens} spans={len(report.spans)} "
        f"inexact_text={inexact} warnings={len(report.warnings)} "
        f"-> {path.relative_to(ROOT)} ({path.stat().st_size} bytes)"
    )
    for warning in report.warnings:
        print(f"  warning: {warning}")


def main() -> None:
    for key in sys.argv[1:] or MODELS:
        record(key)


if __name__ == "__main__":
    main()
