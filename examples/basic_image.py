from mllm_tokens import Analyzer


def main() -> None:
    analyzer = Analyzer.from_pretrained("openbmb/MiniCPM-o-4_5", trust_remote_code=True)

    report = analyzer.analyze_prompt(
        "Describe this image.",
        images=["examples/assets/image_sample.jpg"],
    )

    report.print_dict()


if __name__ == "__main__":
    main()
