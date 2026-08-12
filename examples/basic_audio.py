from mllm_tokens import Analyzer


def main() -> None:
    analyzer = Analyzer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

    report = analyzer.analyze_prompt(
        "Describe this audio.",
        audios=["examples/assets/audio_sample.wav"],
    )

    report.print_dict()


if __name__ == "__main__":
    main()
