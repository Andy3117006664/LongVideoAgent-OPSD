from __future__ import annotations

import json

import pytest

from methods.CAUSAL_EVIDENCE_ORBITS.codebooks import CODEBOOKS, encode_options, to_semantic
from methods.CAUSAL_EVIDENCE_ORBITS.io import first_retrieval_candidates
from methods.CAUSAL_EVIDENCE_ORBITS.question import parse_question
from methods.CAUSAL_EVIDENCE_ORBITS.router import causal_policy, measurement, reduce_probe
from methods.CAUSAL_EVIDENCE_ORBITS.schema import CandidateSpan, Distribution


def dist(a: float, b: float, c: float, d: float) -> Distribution:
    values = {"A": a, "B": b, "C": c, "D": d}
    return Distribution(max(values, key=values.get), values, "A")


def test_codebook_round_trip() -> None:
    _, cycle = CODEBOOKS[1]
    mapped = to_semantic({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}, cycle)
    assert mapped == {"B": 0.1, "C": 0.2, "D": 0.3, "A": 0.4}
    text = encode_options({"A": "one", "B": "two", "C": "three", "D": "four"}, cycle)
    assert text.splitlines()[0] == "(A) two"


def test_question_contract() -> None:
    stem, options = parse_question("What?\n(A) one\n(B) two\n(C) three\n(D) four")
    assert stem == "What?"
    assert options["D"] == "four"


def test_first_retrieval_candidates(tmp_path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "action": {"name": "visual_retrieve"},
                        "observation": {
                            "ok": True,
                            "metadata": {"fixed30_retrieval": {"selected_doc_ids": ["0_16", "32_48"]}},
                        },
                    }
                ]
            }
        )
    )
    values = first_retrieval_candidates(path, limit=2)
    assert [(item.start, item.end, item.semantic_rank) for item in values] == [(0.0, 16.0, 1), (32.0, 48.0, 2)]


def test_invariant_probe_is_ranked() -> None:
    span = CandidateSpan("0_16", 0.0, 16.0, 1)
    identity = measurement(
        "identity",
        CODEBOOKS[0][1],
        dist(0.7, 0.1, 0.1, 0.1),
        dist(0.25, 0.25, 0.25, 0.25),
    )
    cycle = measurement(
        "cycle_1",
        CODEBOOKS[1][1],
        dist(0.1, 0.1, 0.1, 0.7),
        dist(0.25, 0.25, 0.25, 0.25),
    )
    probe = reduce_probe(span, (identity, cycle))
    assert probe.invariant and probe.causal_winner == "A" and probe.score > 0
    assert causal_policy((probe,)) == (span,)


def test_noninvariant_probe_is_rejected() -> None:
    span = CandidateSpan("0_16", 0.0, 16.0, 1)
    first = measurement("identity", CODEBOOKS[0][1], dist(0.7, 0.1, 0.1, 0.1), dist(0.25, 0.25, 0.25, 0.25))
    second = measurement("cycle_1", CODEBOOKS[1][1], dist(0.7, 0.1, 0.1, 0.1), dist(0.25, 0.25, 0.25, 0.25))
    probe = reduce_probe(span, (first, second))
    assert not probe.invariant
    assert causal_policy((probe,)) == ()


def test_distribution_rejects_bad_sum() -> None:
    with pytest.raises(ValueError):
        Distribution("A", {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}, "A")

