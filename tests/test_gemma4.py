from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mllm_tokens import Audio, Image, Message, Text, TokenSegment, Video
from mllm_tokens.adapters.gemma4 import Gemma4Adapter

WITH_AUDIO = SimpleNamespace(audio_config=SimpleNamespace(model_type="gemma4_audio"))
WITHOUT_AUDIO = SimpleNamespace(audio_config=None)
NO_AUDIO_ATTRIBUTE = SimpleNamespace()


def make_processor(media: str = "all") -> Mock:
    """Fake processor output for all four modalities, "audio" only, or "none"."""
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: text.split()
    tokenizer.convert_tokens_to_ids.return_value = 42
    processor = Mock(
        tokenizer=tokenizer,
        image_processor=Mock(pooling_kernel_size=2),
        video_processor=Mock(pooling_kernel_size=2),
        audio_token="<|audio|>",
    )
    input_ids = torch.zeros((1, 60), dtype=torch.int64)
    if media != "none":
        input_ids[0, 20:30] = 42
    output = {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, 60), dtype=torch.int64),
    }
    processor.apply_chat_template.return_value = output
    if media != "all":
        return processor

    image_positions = torch.full((1, 20, 2), -1, dtype=torch.int64)
    image_positions[0, :16] = 0
    video_positions = torch.full((1, 2, 20, 2), -1, dtype=torch.int64)
    video_positions[0, :, :16] = 0
    output["image_position_ids"] = image_positions
    output["video_position_ids"] = video_positions
    return processor


def make_adapter(model_id: str, config, media: str = "audio") -> Gemma4Adapter:
    adapter = Gemma4Adapter(model_id, make_processor(media), config)
    adapter._kv_cache_bytes_per_token = Mock(return_value=1024)
    return adapter


def test_gemma4_builds_all_modality_segments() -> None:
    adapter = make_adapter("google/gemma-4-e4b-it", WITH_AUDIO, media="all")

    report = adapter.analyze(
        [
            Message.user(
                Text("Describe everything"),
                Image("image.jpg"),
                Audio("audio.wav"),
                Video("video.mp4"),
            )
        ]
    )

    assert report.segments == (
        TokenSegment(0, 0, "user", "text", 2),
        TokenSegment(0, 1, "user", "image", 4),
        TokenSegment(0, 2, "user", "audio", 10),
        TokenSegment(0, 3, "user", "video", 8),
    )
    assert report.template_tokens == 36
    assert report.total_tokens == (
        report.text_tokens
        + report.image_tokens
        + report.video_tokens
        + report.audio_tokens
        + report.template_tokens
    )


AUDIO_MESSAGE = [Message.user(Text("Transcribe this"), Audio("audio.wav"))]
FALLBACK_WARNING = (
    "Audio support was inferred from the model ID because the config has no "
    "audio_config attribute."
)


def test_audio_config_enables_audio_for_any_model_id() -> None:
    # A local path or renamed fine-tune has no variant name in its ID.
    adapter = make_adapter("/models/my-gemma-finetune", WITH_AUDIO)

    report = adapter.analyze(AUDIO_MESSAGE)

    assert report.audio_tokens == 10
    assert FALLBACK_WARNING not in report.warnings


def test_null_audio_config_rejects_audio_even_for_audio_variant_names() -> None:
    adapter = make_adapter("org/gemma-4-e4b-it-text-only", WITHOUT_AUDIO)

    with pytest.raises(ValueError, match=r"audio \(config.audio_config is null"):
        adapter.analyze(AUDIO_MESSAGE)


def test_null_audio_config_rejects_audio_for_large_checkpoints() -> None:
    adapter = make_adapter("google/gemma-4-31B-it", WITHOUT_AUDIO)

    with pytest.raises(ValueError, match="Not supported input type: audio"):
        adapter.analyze(AUDIO_MESSAGE)


def test_missing_audio_config_falls_back_to_model_id_with_warning() -> None:
    adapter = make_adapter("google/gemma-4-E4B-it", NO_AUDIO_ATTRIBUTE)

    report = adapter.analyze(AUDIO_MESSAGE)

    assert report.audio_tokens == 10
    assert FALLBACK_WARNING in report.warnings


def test_missing_audio_config_and_non_audio_name_rejects_audio() -> None:
    adapter = make_adapter("google/gemma-4-26B-A4B-it", NO_AUDIO_ATTRIBUTE)

    with pytest.raises(ValueError, match="names no audio variant"):
        adapter.analyze(AUDIO_MESSAGE)


def test_fallback_warning_only_appears_with_audio_input() -> None:
    adapter = make_adapter("google/gemma-4-E4B-it", NO_AUDIO_ATTRIBUTE, media="none")

    report = adapter.analyze([Message.user(Text("Hello"))])

    assert FALLBACK_WARNING not in report.warnings
