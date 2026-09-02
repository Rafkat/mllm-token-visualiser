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
| NVIDIA Nemotron | — | — | — | — | Planned |

Gemma 4 modality support depends on the selected checkpoint and its processor. The adapter can account for image, video, and audio placeholders when the checkpoint supports them.

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

KV-cache values are architectural estimates. They exclude model weights, encoder activations, temporary attention buffers, framework overhead, and allocator fragmentation.

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

Install version 0.4.0 into another uv project:

```bash
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.4.0-py3-none-any.whl
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

## Text-token accounting

Text tokens are calculated from the original `Text` objects with model special tokens disabled. They are not inferred by subtracting every special token from the final template sequence. This keeps role markers, separators, and generation prompts in `template_tokens` rather than misclassifying them as user text.

For batched tensors, modality matching is combined with `attention_mask` so padding positions do not contribute to counts.

## KV-cache estimation

For a conventional decoder transformer, the approximate cache cost per active context position is:

```text
2 × num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes
```

The factor `2` represents keys and values. Query heads are not used because queries are not retained in the KV cache.

## Public API

```python
from mllm_tokens import (
    Analyzer,
    Audio,
    Image,
    Message,
    Text,
    TokenReport,
    Video,
    analyze,
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
├── inputs.py
├── registry.py
├── report.py
└── adapters/
    ├── base.py
    ├── dtype_bytes.py
    ├── gemma4.py
    ├── minicpmo45.py
    ├── qwen3omni.py
    └── qwen3vl.py
```

Each model family owns processor conversion and token accounting inside an adapter. `Analyzer` depends on the common `ModelAdapter` contract, so model support remains additive.

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
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.4.0-py3-none-any.whl
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
- [ ] Add NVIDIA Nemotron support after selecting and validating the target multimodal checkpoint family
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
