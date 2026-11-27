"""Small deterministic tokenizer for adapter tests."""


class FakeTokenizer:
    """Maps whitespace-separated words (and listed specials) to fixed IDs."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.pieces = {token_id: piece for piece, token_id in vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        return [self.vocab[word] for word in text.split()]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(self.pieces[token_id] for token_id in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]
