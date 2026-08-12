from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TokenReport:
    model_id: str

    total_tokens: int
    text_tokens: int
    image_tokens: int
    video_tokens: int
    audio_tokens: int
    template_tokens: int

    token_id_bytes: int

    kv_cache_bytes: int | None = None
    kv_cache_bytes_per_token: int | None = None

    @property
    def kv_cache_mib(self) -> float | None:
        if self.kv_cache_bytes is None:
            return None

        return self.kv_cache_bytes / 1024**2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


    def print_dict(self) -> None:
        key_values = list(self.to_dict().items())

        print("*" * 50)
        for i, (key, value) in enumerate(key_values):
            print(f"{key}: {value}")
            if i != len(key_values) - 1:
                print("-" * 50)
        print("*" * 50)