# MLLM Token Visualiser

> Inspect how text, images, video, and audio expand into model context—and estimate the memory cost before inference.

`mllm-token-visualiser` is a Python library for analysing the tokenised inputs of multimodal large language models (MLLMs). It provides one model-independent interface while keeping processor-specific behaviour inside adapters.

The first supported model families are **Qwen3-VL** and **Qwen3-Omni**, covering text, image, video, and audio input analysis through one public API.

## Why this project?

For a text-only model, counting tokens is usually straightforward. For an MLLM, the final context also contains:

- image placeholders produced after visual preprocessing;
- video tokens whose count depends on sampling and resolution;
- audio tokens or features;
- chat-template tokens such as roles, separators, and generation prompts;
- padding, which must not be counted as active context;
- model-specific special tokens;
- a KV cache whose size depends on the decoder architecture and cache dtype.

Simply removing every special token is not enough. Template fragments such as `user` and newlines may remain, while some special tokens represent genuine modality positions. This library instead separates **user content**, **modality tokens**, and **template overhead** using the original structured input together with the model processor.

## What it reports

A token analysis can report:

| Metric | Meaning |
|---|---|
| `total_tokens` | All active input positions, excluding padding |
| `text_tokens` | Tokens produced by user-provided text only |
| `image_tokens` | Image positions inserted into the model context |
| `video_tokens` | Video positions inserted into the model context |
| `audio_tokens` | Audio positions inserted into the model context |
| `template_tokens` | Roles, separators, generation prompt, and other template overhead |
| `token_id_bytes` | Memory occupied by the `input_ids` tensor |
| `kv_cache_bytes_per_token` | Estimated decoder KV-cache cost per context position |
| `kv_cache_bytes` | Estimated KV-cache memory for the analysed input |

The KV-cache values are architectural estimates. They do not include model weights, encoder activations, temporary attention buffers, framework overhead, or allocator fragmentation.

## Status

This project is an early prototype. Its public API and report schema may evolve before `1.0`.

| Model family | Status | Modalities |
|---|---|---|
| Qwen3-VL | Implemented | text, image, video |
| Qwen3-Omni | Implemented | text, image, video, audio |
| Other MLLMs | Planned | adapter-dependent |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

### Development installation

```bash
git clone <repository-url>
cd mllm-token-visualiser
uv sync
```

### Install a built wheel

```bash
uv build
uv pip install dist/mllm_token_visualiser-0.1.0-py3-none-any.whl
```

In another uv project, prefer recording the wheel as a dependency:

```bash
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.1.0-py3-none-any.whl
```

### Video support

Video decoding requires FFmpeg and a supported decoding backend. On macOS:

```bash
brew install ffmpeg
uv add torchcodec
```

To explicitly select TorchCodec for Qwen video processing:

```bash
FORCE_QWENVL_VIDEO_READER=torchcodec \
  uv run python examples/basic_video.py
```

TorchCodec is preferred because recent `torchvision` releases removed the legacy `torchvision.io.read_video` API.

## Quick start

```python
from mllm_tokens import Analyzer, Image, Message, Text

analyzer = Analyzer.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
)

messages = [
    Message.user(
        Image("examples/assets/cat.jpg"),
        Text("Describe this image."),
    )
]

report = analyzer.analyze(messages)

print(report)
print(report.to_dict())
```

`Analyzer.from_pretrained()` is intended to load the processor and configuration required for analysis, not the complete model weights. Reuse the same `Analyzer` for multiple inputs to avoid repeatedly loading processor metadata.

## Interleaved multimodal input

Content order is preserved. This matters when text refers to media appearing before or after it:

```python
from mllm_tokens import Analyzer, Image, Message, Text

analyzer = Analyzer.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
)

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
print(report)
```

The wrappers remove ambiguity: a plain string such as `"cat.jpg"` could mean text or a file path, whereas `Text(...)` and `Image(...)` state the modality explicitly.

## Video example

```python
from mllm_tokens import Analyzer, Message, Text, Video

analyzer = Analyzer.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
)

messages = [
    Message.user(
        Video("examples/assets/example.mp4"),
        Text("What happens in this video?"),
    )
]

report = analyzer.analyze(messages)
print(report.to_dict())
```

Run the repository example with:

```bash
uv run python examples/basic_video.py
```

## Qwen3-Omni example

Qwen3-Omni accepts audio in addition to text and visual content. The same structured API can represent an audio-only request or a fully interleaved multimodal message:

```python
from mllm_tokens import Analyzer, Audio, Image, Message, Text

analyzer = Analyzer.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
)

messages = [
    Message.user(
        Text("Listen to the recording and inspect the image."),
        Audio("examples/assets/example.wav"),
        Image("examples/assets/frame.jpg"),
        Text("Describe the situation and any relevant sounds."),
    )
]

report = analyzer.analyze(messages)

print(report)
print(report.to_dict())
```

The Omni adapter accounts for text, image, video, audio, and chat-template positions separately. Its input KV-cache estimate uses the **Thinker text decoder**, because the Thinker maintains the autoregressive multimodal context.

If the model generates speech, the Talker uses a separate KV cache. Talker-generation memory is conceptually distinct from the input-context estimate because its architecture and generated sequence length differ.

## Text-token accounting

Text tokens are calculated from the original `Text` objects, encoded without adding model special tokens. They are **not** obtained by subtracting every special token from the final chat-template sequence.

This distinction prevents template content from being misclassified as user text:

```text
total context
├── user text
├── image / video / audio positions
└── template overhead
    ├── role markers
    ├── separators and newlines
    └── optional generation prompt
```

For batched inputs, modality-token matching is combined with `attention_mask` so padding positions do not contribute to the counts.

## KV-cache estimation

For a conventional decoder transformer, the approximate cache cost per context position is:

```text
2 × num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes
```

The factor `2` represents keys and values. `num_key_value_heads` is used instead of query-head count because queries are not retained in the KV cache.

For Qwen3-Omni input analysis, these values come from the Thinker text decoder:

```python
config.thinker_config.text_config
```

The Thinker holds the autoregressive context for the multimodal request. The Talker has a separate cache during audio generation and should be reported independently because its sequence length and architecture differ.

## Public API

The intended public imports live in `mllm_tokens.__init__`:

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
├── __init__.py          # public API
├── analyzer.py          # user-facing orchestration
├── inputs.py            # typed multimodal input objects
├── registry.py          # model family → adapter factory
├── report.py            # immutable analysis result
└── adapters/
    ├── base.py          # ModelAdapter contract
    ├── qwen3_vl.py      # Qwen3-VL processing and accounting
    └── qwen3_omni.py    # Qwen3-Omni processing and accounting
```

Each model family owns its processor-specific conversion and token accounting inside an adapter. The `Analyzer` depends only on the shared `ModelAdapter` interface, making future model support additive rather than a growing chain of conditions.

## Repository layout

```text
mllm-token-visualiser/
├── pyproject.toml
├── README.md
├── src/mllm_tokens/
├── tests/
└── examples/
    ├── assets/
    ├── basic_text.py
    ├── basic_image.py
    ├── basic_video.py
    └── interleaved.py
```

- `src/` contains the installable library.
- `tests/` contains automated correctness checks.
- `examples/` contains small runnable demonstrations for users and contributors.

## Development

Install the project and development dependencies:

```bash
uv sync
```

Run formatting, linting, and tests:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Build both the wheel and source distribution:

```bash
uv build
```

Verify the public package import:

```bash
uv run python -c "import mllm_tokens; print(mllm_tokens.__file__)"
```

## Testing the wheel as an external user

Create a separate project so the test cannot accidentally import source files from the repository:

```bash
mkdir test_project
cd test_project
uv init
uv python pin 3.12
uv add ../mllm-token-visualiser/dist/mllm_token_visualiser-0.1.0-py3-none-any.whl
uv run python -c "import mllm_tokens; print(mllm_tokens.__file__)"
```

The printed path should be inside `test_project/.venv/.../site-packages/`.

## Roadmap

- [x] Typed text, image, video, and audio input objects
- [x] Ordered interleaved message representation
- [x] Adapter registry and common adapter contract
- [x] Structured token report
- [x] Qwen3-VL token accounting prototype
- [x] Qwen3-Omni text, image, video, and audio accounting
- [x] Qwen3-Omni Thinker KV-cache estimation
- [x] Token-ID and KV-cache memory estimates
- [ ] Separate Thinker and Talker memory reports
- [ ] Add a CLI built on the same library API
- [ ] Add further MLLM adapters
- [ ] Add richer terminal visualisation and export formats
- [ ] Stabilise the public API and publish to PyPI

## Design principles

- **Processor-aware:** count the sequence that the model processor actually creates.
- **Content-aware:** distinguish user content from chat-template overhead.
- **Model-extensible:** isolate family-specific behaviour behind adapters.
- **Lightweight:** analyse tokenisation without loading full model weights when possible.
- **Transparent:** expose assumptions behind every memory estimate.
- **Library first:** keep the core usable from Python; a CLI can reuse the same implementation.

## Contributing

Issues and pull requests are welcome, especially for:

- processor-version compatibility;
- new model adapters;
- reference token-count fixtures;
- video and audio decoding portability;
- more accurate cache-memory modelling.

When adding an adapter, include small processor-level tests and document which model and `transformers` versions were validated.

## License

Choose and add a licence before publishing the project. Apache-2.0 or MIT are common choices for an open-source Python utility; the selected licence should be recorded in both `LICENSE` and `pyproject.toml`.

---

Built to make multimodal context measurable before it becomes an out-of-memory error.
