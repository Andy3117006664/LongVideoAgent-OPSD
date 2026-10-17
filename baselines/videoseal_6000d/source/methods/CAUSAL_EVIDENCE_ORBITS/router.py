from __future__ import annotations

import hashlib
import math

from .codebooks import to_semantic
from .schema import (
    ArmDecision,
    CandidateSpan,
    CodebookMeasurement,
    Distribution,
    LETTERS,
    SpanProbe,
)


def _winner_and_margin(scores: dict[str, float]) -> tuple[str, float]:
    ordered = sorted(LETTERS, key=lambda letter: (-float(scores[letter]), letter))
    return ordered[0], float(scores[ordered[0]] - scores[ordered[1]])


def measurement(
    codebook_id: str,
    token_to_semantic: dict[str, str],
    real: Distribution,
    pixel_null: Distribution,
) -> CodebookMeasurement:
    semantic_real = to_semantic(real.probabilities, token_to_semantic)
    semantic_null = to_semantic(pixel_null.probabilities, token_to_semantic)
    causal = {
        letter: math.log(max(semantic_real[letter], 1e-12))
        - math.log(max(semantic_null[letter], 1e-12))
        for letter in LETTERS
    }
    winner, margin = _winner_and_margin(causal)
    tv = 0.5 * sum(abs(semantic_real[letter] - semantic_null[letter]) for letter in LETTERS)
    return CodebookMeasurement(
        codebook_id,
        dict(token_to_semantic),
        real,
        pixel_null,
        semantic_real,
        semantic_null,
        causal,
        winner,
        margin,
        tv,
    )


def reduce_probe(span: CandidateSpan, values: tuple[CodebookMeasurement, ...]) -> SpanProbe:
    if len(values) < 2:
        raise ValueError("an orbit needs at least two codebooks")
    winners = {item.causal_winner for item in values}
    invariant = len(winners) == 1
    winner = next(iter(winners)) if invariant else ""
    score = (
        min(item.visual_tv for item in values) * min(item.causal_margin for item in values)
        if invariant
        else -1.0
    )
    return SpanProbe(span, values, invariant, winner, float(score))


def causal_policy(probes: tuple[SpanProbe, ...], count: int = 3) -> tuple[CandidateSpan, ...]:
    values = [item for item in probes if item.invariant and item.score > 0.0]
    values.sort(key=lambda item: (-item.score, item.span.semantic_rank, item.span.doc_id))
    return tuple(item.span for item in values[:count])


def semantic_policy(candidates: tuple[CandidateSpan, ...], count: int = 3) -> tuple[CandidateSpan, ...]:
    return tuple(candidates[:count])


def sham_policy(candidates: tuple[CandidateSpan, ...], uid: str, count: int = 3) -> tuple[CandidateSpan, ...]:
    values = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"causal-evidence-orbits-sham-v1:{uid}:{item.doc_id}".encode()
        ).hexdigest(),
    )
    return tuple(values[:count])


def decide_arm(
    *,
    arm: str,
    selected: tuple[CandidateSpan, ...],
    probe_by_doc: dict[str, SpanProbe],
    replay: tuple[CodebookMeasurement, ...],
    incumbent: str,
) -> ArmDecision:
    probe_winners = [probe_by_doc[item.doc_id].causal_winner for item in selected[:2]]
    probe_consensus = (
        probe_winners[0]
        if len(probe_winners) == 2 and probe_winners[0] and len(set(probe_winners)) == 1
        else ""
    )
    replay_winners = {item.causal_winner for item in replay}
    replay_invariant = len(replay) >= 2 and len(replay_winners) == 1
    replay_winner = next(iter(replay_winners)) if replay_invariant else ""
    override = len(selected) >= 2 and bool(probe_consensus) and replay_winner == probe_consensus
    return ArmDecision(
        arm,
        tuple(item.doc_id for item in selected),
        probe_consensus,
        replay_winner,
        replay_invariant,
        override,
        replay_winner if override else incumbent,
        replay,
    )

