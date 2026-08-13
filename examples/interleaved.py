from mllm_tokens import Analyzer, Audio, Image, Message, Text


def main() -> None:
    analyzer = Analyzer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    messages = [
        Message.user(
            Text("Listen to the recording and inspect the image."),
            Audio("examples/assets/audio_sample.wav"),
            Image("examples/assets/image_sample.jpg"),
            Text("Describe the situation and any relevant sounds."),
        )
    ]

    report = analyzer.analyze(messages)

    report.print_dict()


if __name__ == "__main__":
    main()
