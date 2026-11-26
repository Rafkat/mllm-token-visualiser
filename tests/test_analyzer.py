from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mllm_tokens.analyzer as analyzer_module
from mllm_tokens import Analyzer, Message, Text
from mllm_tokens.adapters.qwen3vl import Qwen3VLAdapter


def test_analyze_delegates_to_adapter() -> None:
    expected_report = Mock()
    adapter = Mock()
    adapter.analyze.return_value = expected_report

    analyzer = Analyzer(adapter)
    messages = [Message.user(Text("Hello"))]

    result = analyzer.analyze(
        messages, add_generation_prompt=False, kv_cache_dtype="float16"
    )

    assert result is expected_report
    adapter.analyze.assert_called_once_with(
        messages, add_generation_prompt=False, kv_cache_dtype="float16"
    )


def test_analyze_prompt_builds_message_in_expected_order() -> None:
    adapter = Mock()
    analyzer = Analyzer(adapter)

    analyzer.analyze_prompt(
        "Describe everything.",
        images=["image.jpg"],
        videos=["video.mp4"],
        audios=["audio.wav"],
    )

    messages = adapter.analyze.call_args.args[0]
    content = messages[0].content

    assert [type(item).__name__ for item in content] == [
        "Image",
        "Video",
        "Audio",
        "Text",
    ]


def test_from_pretrained_rejects_unsupported_model_before_loading_processor(
    monkeypatch,
) -> None:
    auto_config = Mock()
    auto_config.from_pretrained.return_value = SimpleNamespace(model_type="llama")
    auto_processor = Mock()
    monkeypatch.setattr(analyzer_module, "AutoConfig", auto_config)
    monkeypatch.setattr(analyzer_module, "AutoProcessor", auto_processor)

    with pytest.raises(ValueError, match="Unsupported model"):
        Analyzer.from_pretrained("/models/local-llama")

    auto_processor.from_pretrained.assert_not_called()


def test_from_pretrained_selects_adapter_from_config(monkeypatch) -> None:
    config = SimpleNamespace(model_type="qwen3_vl")
    processor = Mock()
    auto_config = Mock()
    auto_config.from_pretrained.return_value = config
    auto_processor = Mock()
    auto_processor.from_pretrained.return_value = processor
    monkeypatch.setattr(analyzer_module, "AutoConfig", auto_config)
    monkeypatch.setattr(analyzer_module, "AutoProcessor", auto_processor)

    analyzer = Analyzer.from_pretrained("/models/qwen-finetune", min_pixels=64)

    assert isinstance(analyzer._adapter, Qwen3VLAdapter)
    assert analyzer._adapter.config is config
    assert analyzer._adapter.processor is processor
    auto_processor.from_pretrained.assert_called_once_with(
        "/models/qwen-finetune", trust_remote_code=False, min_pixels=64
    )
