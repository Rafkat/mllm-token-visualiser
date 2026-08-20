from unittest.mock import Mock

from mllm_tokens import Analyzer, Message, Text


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
