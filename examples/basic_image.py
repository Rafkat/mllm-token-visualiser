from mllm_tokens import Analyzer


def main() -> None:
    analyzer = Analyzer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    report = analyzer.analyze_prompt(
        "Describe this image.",
        images=["examples/assets/example.jpg"],
    )

    print(report.to_dict())


if __name__ == "__main__":
    main()
