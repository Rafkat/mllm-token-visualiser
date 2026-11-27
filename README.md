# MLLM Token Visualiser

> Measure how text, images, video, and audio expand into model context—and estimate their memory cost before inference.

`mllm-token-visualiser` is a Python library for analysing tokenised multimodal inputs without loading complete model weights. It exposes one typed, model-independent API while keeping processor-specific behaviour inside adapters.

The library distinguishes user text, modality positions, chat-template overhead, and padding instead of treating every special token as removable noise.

## Features

- Counts total, text, image, video, audio, and template tokens.
- Uses each model family's processor and chat template.
- Preserves interleaved multimodal content order.
- Excludes padding through the attention mask.
- Estimates `input_ids` memory and decoder KV-cache memory.
- Loads processor/configuration metadata rather than full model weights.
- Provides an adapter interface for adding further model families.

## Supported models

| Model family | Text | Image | Video | Audio | Status |
|---|:---:|:---:|:---:|:---:|---|
| Qwen3-VL | ✅ | ✅ | ✅ | — | Implemented |
| Qwen3-Omni | ✅ | ✅ | ✅ | ✅ | Implemented |
| MiniCPM-o 4.5 | ✅ | ✅ | ✅ | ✅ | Implemented |
| Gemma 4 | ✅ | ✅ | Checkpoint-dependent | Checkpoint-dependent | Implemented |
| NVIDIA Nemotron 3 Nano Omni FP8 | ✅ | ✅ | ✅ | ✅ | Implemented |

Gemma 4 modality support depends on the selected checkpoint and its processor. The adapter can account for image, video, and audio placeholders when the checkpoint supports them.

The Nemotron adapter has been checked against the processor of
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (revision `0acd4b1c`).
Other precision variants may use different revisions of the remote processor
and have not yet been validated.
Nemotron 3 Nano Omni currently supports one video per request. Video files are sampled with PyAV before being passed to the model processor.

This project is still pre-`1.0`; its public API and report schema may evolve.

## What it reports

| Metric | Meaning |
|---|---|
| `total_tokens` | All active input positions, excluding padding |
| `text_tokens` | Tokens produced by user-provided text only |
| `image_tokens` | Image positions inserted into model context |
| `video_tokens` | Video positions inserted into model context |
| `audio_tokens` | Audio positions inserted into model context |
| `template_tokens` | Roles, separators, generation prompt, and other template overhead |
| `token_id_bytes` | Memory occupied by the `input_ids` tensor |
| `kv_cache_bytes` | Estimated KV-cache memory for the analysed input |
| `kv_cache_bytes_per_token` | Estimated decoder KV-cache cost per active context position |
| `segments` | Per-content token counts with message index, content index, role, and modality |
| `spans` | The whole active sequence as contiguous runs: `text`, `image`, `video`, `audio`, or `template`, with the source item and decoded text |
| `media` | One entry per image/video/audio: tokens, interior overhead, placeholder groups, source-file metadata, and processor details |
| `metadata` | Library/Transformers versions, adapter, processor class, generation-prompt flag, and KV-cache dimensions |
| `assumptions` | The accounting and estimation assumptions that apply to this report |
| `warnings` | Problems detected during analysis, for example a span that could not be aligned |

KV-cache values are architectural estimates. They exclude model weights, encoder activations, temporary attention buffers, framework overhead, and allocator fragmentation.

## Detailed report

### Token spans

`report.spans` maps every active position to its origin, in sequence order. Spans always cover the whole active sequence exactly once:

```python
for span in report.spans:
    print(span.start, span.end, span.kind, span.content_index, repr(span.text))
```

```text
0     3     template None '<|im_start|>system\n'
3     9     text     0    'You are a helpful assistant.'
...
17    18    template None '<|vision_start|>'
18    210   image    1    None
210   211   template None '<|vision_end|>'
```

- Media spans come from the processor's placeholder positions and are exact. One item can produce several spans, for example one per video frame group.
- Template positions between the first and last placeholder of one media item, such as frame timestamps, are attributed to that item (`message_index`/`content_index` set, `kind == "template"`). The total appears as `MediaDetail.overhead_tokens`.
- Each `Text` item is located in conversation order. `exact` is `False` when its isolated tokenization does not appear verbatim because the tokenizer merged it with neighbouring template characters (for example `":"` + newline). The item is then located by its characters and the span includes the merged token.
- If spans cannot be aligned, the report still contains all counts and explains the problem in `warnings`.

### Media details

```python
for item in report.media:
    print(item.modality, item.tokens, item.overhead_tokens, item.properties)
```

Measured with `Qwen/Qwen3-VL-8B-Instruct` and Transformers 5.14.1 on `examples/assets/video_sample.mp4` (1024×432, 25.1 s, 23.976 fps):

```text
video 11200 207 {'duration_seconds': 25.108, 'grid_thw': [25, 28, 64], 'frames_sampled': 50,
                 'tokens_per_frame_group': 448, 'tokens_per_second': 446.07, ...}
```

Source properties (`width`, `height`, `duration_seconds`, `source_fps`, `sample_rate`) are read from file headers only. Processor properties depend on the model family: patch grids for Qwen, pooled patches for Gemma 4, slices and sampled frames for MiniCPM-o, and tiles and sampled frames for Nemotron.

### Terminal view

```python
report.show()  # summary, composition, items, token map, warnings, assumptions
report.show(show_template=True)  # also list decoded chat-template spans
report.show(width=80, map_rows=4)  # bar/map width in cells, map height in rows
```

Abridged output (bar shortened, one item row shown) for `Qwen/Qwen3-VL-8B-Instruct` with `examples/detailed_report.py`:

```text
Total tokens          11,639
KV cache (estimate)   1.60 GiB  (144.00 KiB per token, bfloat16)

Composition
█▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒··
  Kind          Tokens    Share    KV cache
  ██ text           15     0.1%    2.11 MiB
  ▓▓ image         192     1.6%   27.00 MiB
  ▒▒ video      11,200    96.2%    1.54 GiB
  ░░ audio           0     0.0%         0 B
  ·· template      232     2.0%   32.62 MiB

Input items
  Msg  Role    Item      Tokens  Share  Details
    1  user    ▒▒ video  11,200  96.2%  1024×432 → 1024×448, 25.108 s, 50 frames,
                                        grid 25×28×64, 446.07 tok/s, 224 tok/frame,
                                        +207 overhead
```

Each kind has its own colour and glyph (`█ text`, `▓ image`, `▒ video`, `░ audio`, `· template`), so the view also works without colour, for example with `NO_COLOR=1` or in logs. In the token map one cell stands for several positions and shows their majority kind; use `report.spans` or the template view for exact positions. `mllm_tokens.display.render_report()` returns the `rich` renderable for embedding in other output.

### HTML report

```python
report.save_html("report.html")  # one self-contained file, no external assets
report.save_html("report.html", embed_media=True)  # also embed image thumbnails
html = report.to_html(title="Qwen3-VL, 25 s clip")
```

The page shows the total, the KV-cache estimate, a composition bar, a token map with one lane per kind (hover, or focus it and use the arrow keys, to see each span's positions, owner, and decoded text), the input items with their media properties, the decoded template spans, and the warnings, assumptions, and run details. Every chart has a table view, the page follows the system light or dark theme (with a toggle; `?theme=dark` also works), and "Download JSON" saves `report.to_json()`.

By default the file contains no media: only counts, file names, header metadata, and decoded text and template spans. `embed_media=True` adds 240 px JPEG thumbnails of image inputs; keep it off for reports you share.

### Breakdowns and export

```python
report.tokens_by_kind()  # {'text': ..., 'image': ..., 'template': ...}
report.token_shares()  # fraction of total_tokens per kind
report.kv_cache_bytes_by_kind()  # estimated KV-cache bytes per kind

report.to_json()  # versioned JSON with derived breakdowns and field accuracy
report.segments_csv()  # one row per content item
report.spans_csv()  # one row per span
```

`to_json()` includes `schema_version` and a `field_accuracy` map that labels each count as processor-exact, tokenizer-based, derived, or estimated.

## Why special-token filtering is not enough

An MLLM context contains more than ordinary text tokens:

```text
total context
├── user text
├── image / video / audio positions
└── template overhead
    ├── role markers
    ├── separators and newlines
    └── optional generation prompt
```

Removing every special token would also remove genuine modality positions, while template fragments such as role text or newlines may remain. The library instead combines the original structured input with the final processor output.

## Requirements

- Python `>=3.11,<3.15`
- PyTorch 2.13
- Transformers `>=5.2,<6`
- PyAV 18 or later for Nemotron video decoding
- FFmpeg shared libraries for processor paths that decode audio or video through TorchCodec

TorchAudio 2.11 supports PyTorch 2.11 and later through PyTorch's stable ABI. TorchCodec 0.16 adds support for FFmpeg 9.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

### Development installation

```bash
git clone https://github.com/Rafkat/mllm-token-visualiser.git
cd mllm-token-visualiser
uv sync
```

Install optional Qwen3-Omni utilities with:

```bash
uv sync --extra omni
```

### Install a built wheel

Build the distribution:

```bash
uv build
```

Install version 0.6.0 into another uv project:

```bash
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.6.0-py3-none-any.whl
```

### FFmpeg and video support

TorchCodec requires an FFmpeg installation that exposes shared libraries. On macOS with Homebrew:

```bash
brew install ffmpeg
ffmpeg -version
```

Recent TorchCodec releases support FFmpeg major versions 4 through 9. The FFmpeg executable being available in `PATH` does not by itself guarantee that the operating-system dynamic loader can locate `libavutil`, `libavcodec`, and the other shared libraries.

For Qwen video processing, TorchCodec can be selected explicitly:

```bash
FORCE_QWENVL_VIDEO_READER=torchcodec \
  uv run python examples/basic_video.py
```

## Quick start

```python
from mllm_tokens import Analyzer, Image, Message, Text

analyzer = Analyzer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    Message.user(
        Image("examples/assets/image_sample.jpg"),
        Text("Describe this image."),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```

`Analyzer.from_pretrained()` loads the processor and configuration needed for analysis, not complete model weights. Reuse an `Analyzer` across inputs to avoid repeatedly loading processor metadata.

## Interleaved multimodal input

Content order is preserved:

```python
from mllm_tokens import Analyzer, Image, Message, Text

analyzer = Analyzer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    Message.user(
        Text("First image:"),
        Image("examples/assets/first.jpg"),
        Text("Second image:"),
        Image("examples/assets/second.jpg"),
        Text("Compare them."),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```

Typed wrappers remove ambiguity: `"cat.jpg"` could mean text or a path, whereas `Text(...)` and `Image(...)` state the intended modality.

## Video example

```python
from mllm_tokens import Analyzer, Message, Text, Video

analyzer = Analyzer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    Message.user(
        Video("examples/assets/video_sample.mp4"),
        Text("What happens in this video?"),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```

Run the repository example with:

```bash
uv run python examples/basic_video.py
```

## Qwen3-Omni example

```python
from mllm_tokens import Analyzer, Audio, Image, Message, Text

analyzer = Analyzer.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
)

messages = [
    Message.user(
        Text("Listen to the recording and inspect the image."),
        Audio("examples/assets/audio_sample.wav"),
        Image("examples/assets/image_sample.jpg"),
        Text("Describe the situation and relevant sounds."),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```

Qwen3-Omni videos are sampled at 2 fps, as in the model card's `qwen-omni-utils` pipeline. The processor's own default keeps every source frame (about 12× more tokens for a 24 fps clip, enough to exceed the context window); the per-frame pixel limits of `qwen-omni-utils` are not replicated.

The Qwen3-Omni input KV-cache estimate uses the Thinker text decoder because it maintains the autoregressive multimodal context. Talker-generation memory is separate and is not included in this input-context estimate.

## Gemma 4 example

```python
from mllm_tokens import Analyzer, Image, Message, Text

analyzer = Analyzer.from_pretrained("google/gemma-4-e4b-it")

messages = [
    Message.user(
        Image("examples/assets/image_sample.jpg"),
        Text("Describe the image and identify its most important details."),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```

The Gemma 4 adapter applies the checkpoint's native chat template, counts supported modality placeholders in the active sequence, and derives KV-cache dimensions from `config.text_config`.

## NVIDIA Nemotron example

```python
from mllm_tokens import Analyzer, Audio, Image, Message, Text, Video

analyzer = Analyzer.from_pretrained(
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", trust_remote_code=True
)

messages = [
    Message.user(
        Image("examples/assets/image_sample.jpg"),
        Audio("examples/assets/audio_sample.wav"),
        Video("examples/assets/video_sample.mp4"),
        Text("Describe the image, audio, and video."),
    )
]

report = analyzer.analyze(messages)
report.print_dict()
```
The Nemotron adapter uses PyAV to sample video frames at 2 fps (up to 256 frames) and passes them to the model processor with the sampling rate, so frame timestamps in the prompt are correct. The current processor integration supports one video per request.

Video placeholders need care with this checkpoint. Its tokenizer has no single `<video>` token, so the processor's output splits every frame placeholder into text fragments that the model never matches: the model scatters video embeddings into positions equal to `config.video_context_token_id`. The adapter rebuilds each frame as `num_image_token` (256) positions of that ID, which is the layout the model expects, and reports this as a warning. `video_tokens` is therefore 256 per sampled frame.

The model also applies Efficient Video Sampling (EVS) after the vision encoder and drops a fixed share of video tokens (`config.video_pruning_rate`, 0.7 for this checkpoint). The retained count is deterministic and is reported as `media[i].properties["tokens_after_evs"]`; `total_tokens` and the KV-cache estimate count video tokens before EVS.
NemotronH combines Mamba-2, attention, and MoE layers. Its KV-cache estimate includes only layers marked as attention layers in hybrid_override_pattern; recurrent Mamba state is not included.


## Text-token accounting

Text tokens are calculated from the original `Text` objects with model special tokens disabled. They are not inferred by subtracting every special token from the final template sequence. This keeps role markers, separators, and generation prompts in `template_tokens` rather than misclassifying them as user text.

For batched tensors, modality matching is combined with `attention_mask` so padding positions do not contribute to counts.

## KV-cache estimation

For a conventional decoder transformer, the approximate cache cost per active context position is:

```text
2 × num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes
```

The factor `2` represents keys and values. Query heads are not used because queries are not retained in the KV cache.

## Segments

```
report = analyzer.analyze(messages)

for segment in report.segments:
    print(
        segment.message_index,
        segment.content_index,
        segment.role,
        segment.modality,
        segment.tokens,
    )
```

## Public API

```python
from mllm_tokens import (
    Analyzer,
    Audio,
    Image,
    Message,
    Text,
    Video,
    analyze,
    MediaDetail,
    TokenReport,
    TokenSegment,
    TokenSpan,
)
```

The distribution and import names intentionally differ:

```text
distribution: mllm-token-visualiser
import:       mllm_tokens
```

## Architecture

```text
src/mllm_tokens/
├── __init__.py
├── analyzer.py
├── display.py
├── html_report.py
├── inputs.py
├── media.py
├── registry.py
├── report.py
├── spans.py
├── templates/
│   └── report.html
└── adapters/
    ├── base.py
    ├── dtype_bytes.py
    ├── gemma4.py
    ├── minicpmo45.py
    ├── nemotron.py
    ├── qwen_adapter.py
    ├── qwen3omni.py
    └── qwen3vl.py
```

Each model family owns processor conversion and token accounting inside an adapter. Adapters report placeholder positions and processor details; `spans.py` (model-independent position mapping) and `media.py` (file-header probing) turn them into the detailed report. `Analyzer` depends on the common `ModelAdapter` contract, so model support remains additive.

`Analyzer.from_pretrained()` selects the adapter from the checkpoint's `config.model_type`, before the processor is loaded, so local paths and renamed fine-tunes resolve correctly and unsupported models fail early. Checkpoints whose `model_type` is not recognised fall back to matching the model ID. `minicpmo` is shared by several MiniCPM-o generations, so it additionally requires `config.version == "4.5"`.

| Adapter | `config.model_type` |
|---|---|
| Qwen3-VL | `qwen3_vl`, `qwen3_vl_moe` |
| Qwen3-Omni | `qwen3_omni_moe` |
| MiniCPM-o 4.5 | `minicpmo` (version `4.5`) |
| Gemma 4 | `gemma4` |
| Nemotron 3 Nano Omni | `NemotronH_Nano_Omni_Reasoning_V3` |

## Repository layout

```text
mllm-token-visualiser/
├── pyproject.toml
├── README.md
├── src/mllm_tokens/
├── tests/
└── examples/
    ├── assets/
    ├── basic_audio.py
    ├── basic_image.py
    ├── basic_video.py
    ├── detailed_report.py
    └── interleaved.py
```

## Development

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

Opt-in checks against real Hugging Face processors download processor and configuration files only (no weights):

```bash
MLLM_TOKENS_INTEGRATION=1 uv run pytest tests/integration
# MiniCPM-o and Nemotron processors execute remote code:
MLLM_TOKENS_INTEGRATION=1 MLLM_TOKENS_TRUST_REMOTE_CODE=1 uv run pytest tests/integration
```

`tests/fixtures/spans/` holds span-builder inputs recorded from real processors with `scripts/record_span_fixtures.py`; `tests/test_span_fixtures.py` replays them offline.

Verify the public import:

```bash
uv run python -c "import mllm_tokens; print(mllm_tokens.__file__)"
```

## Testing the wheel externally

Create a separate project so the test cannot accidentally import the source tree:

```bash
mkdir test_project
cd test_project
uv init
uv python pin 3.12
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.6.0-py3-none-any.whl
uv run python -c "import mllm_tokens; print(mllm_tokens.__file__)"
```

The printed path should be inside `test_project/.venv/.../site-packages/`.

## Roadmap

- [x] Typed text, image, video, and audio inputs
- [x] Ordered interleaved message representation
- [x] Adapter registry and common adapter contract
- [x] Structured token and memory report
- [x] Qwen3-VL adapter
- [x] Qwen3-Omni adapter and Thinker KV-cache estimate
- [x] MiniCPM-o 4.5 adapter
- [x] Gemma 4 adapter
- [x] Add NVIDIA Nemotron support after selecting and validating the target multimodal checkpoint family
- [ ] Add a separate streaming-analysis track for incremental MLLM inputs and outputs
- [ ] Report token growth per streaming chunk or turn
- [ ] Estimate KV-cache growth and retained context during streaming inference
- [ ] Separate input-context, generated-text, and generated-audio accounting where the model exposes distinct generation pipelines
- [ ] Add adapter-level unit and integration fixtures
- [ ] Add a CLI built on the library API
- [ ] Add richer terminal visualisation and export formats
- [ ] Add further MLLM adapters
- [ ] Stabilise the public API and publish to PyPI

## Design principles

- **Processor-aware:** count the sequence the model processor creates.
- **Content-aware:** distinguish user content from chat-template overhead.
- **Model-extensible:** isolate family-specific behaviour behind adapters.
- **Lightweight:** analyse tokenisation without loading full model weights when possible.
- **Transparent:** expose assumptions behind every memory estimate.
- **Library first:** keep the core usable from Python; a CLI can reuse it.

## Contributing

Issues and pull requests are welcome, especially for processor compatibility, new adapters, reference token-count fixtures, media-decoding portability, and more accurate memory modelling.

When adding an adapter, include processor-level tests and document the model checkpoints and Transformers versions used for validation.

## License

Licensed under the [Apache License 2.0](LICENSE.md).

---

Built to make multimodal context measurable before it becomes an out-of-memory error.
