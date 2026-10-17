from __future__ import annotations

from .schema import LETTERS


CODEBOOKS: tuple[tuple[str, dict[str, str]], ...] = (
    ("identity", {"A": "A", "B": "B", "C": "C", "D": "D"}),
    ("cycle_1", {"A": "B", "B": "C", "C": "D", "D": "A"}),
)


def encode_options(options: dict[str, str], token_to_semantic: dict[str, str]) -> str:
    if set(options) != set(LETTERS) or set(token_to_semantic) != set(LETTERS):
        raise ValueError("options and codebook must contain A-D")
    if set(token_to_semantic.values()) != set(LETTERS):
        raise ValueError("codebook must be a permutation")
    return "\n".join(
        f"({token}) {options[token_to_semantic[token]]}" for token in LETTERS
    )


def to_semantic(
    token_probabilities: dict[str, float], token_to_semantic: dict[str, str]
) -> dict[str, float]:
    return {
        semantic: float(token_probabilities[token])
        for token, semantic in token_to_semantic.items()
    }

