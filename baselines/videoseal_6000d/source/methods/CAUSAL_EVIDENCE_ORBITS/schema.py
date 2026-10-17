from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LETTERS = tuple("ABCD")


@dataclass(frozen=True)
class CandidateSpan:
    doc_id: str
    start: float
    end: float
    semantic_rank: int

    def __post_init__(self) -> None:
        if not self.doc_id or self.end <= self.start or self.semantic_rank < 1:
            raise ValueError("invalid candidate span")


@dataclass(frozen=True)
class Distribution:
    prediction: str
    probabilities: dict[str, float]
    raw: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.prediction not in LETTERS or set(self.probabilities) != set(LETTERS):
            raise ValueError("distribution must contain exactly A-D")
        if any(value < 0.0 for value in self.probabilities.values()):
            raise ValueError("probabilities must be non-negative")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("probabilities must sum to one")


@dataclass(frozen=True)
class CodebookMeasurement:
    codebook_id: str
    token_to_semantic: dict[str, str]
    real: Distribution
    pixel_null: Distribution
    semantic_real: dict[str, float]
    semantic_null: dict[str, float]
    causal_scores: dict[str, float]
    causal_winner: str
    causal_margin: float
    visual_tv: float


@dataclass(frozen=True)
class SpanProbe:
    span: CandidateSpan
    measurements: tuple[CodebookMeasurement, ...]
    invariant: bool
    causal_winner: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArmDecision:
    arm: str
    selected_doc_ids: tuple[str, ...]
    probe_consensus: str
    replay_winner: str
    replay_invariant: bool
    override_allowed: bool
    prediction: str
    replay: tuple[CodebookMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

