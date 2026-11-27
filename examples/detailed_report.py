from pathlib import Path

from mllm_tokens import Analyzer, Image, Message, Text, Video


def main() -> None:
    analyzer = Analyzer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    messages = [
        Message.system(Text("You are a helpful assistant.")),
        Message.user(
            Text("First look:"),
            Image("examples/assets/image_sample.jpg"),
            Text("then watch:"),
            Video("examples/assets/video_sample.mp4"),
            Text("Describe everything."),
        ),
    ]

    report = analyzer.analyze(messages)

    report.show(show_template=True)

    Path("report.json").write_text(report.to_json())
    report.save_html("report.html")


if __name__ == "__main__":
    main()
