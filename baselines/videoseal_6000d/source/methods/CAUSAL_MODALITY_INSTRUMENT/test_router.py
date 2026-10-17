import pytest

try:
    from .router import temporal_instrument_commit
except ImportError:  # Allow direct pytest collection from the prototype root.
    from router import temporal_instrument_commit


def test_alignment_specific_effect_overrides() -> None:
    decision = temporal_instrument_commit(
        incumbent="A",
        visual_only="A",
        aligned_audio="D",
        shifted_audio=("A", "A"),
        aligned_transcript_nonempty=True,
    )
    assert decision.override
    assert decision.prediction == "D"


@pytest.mark.parametrize(
    ("visual", "aligned", "shifted", "nonempty"),
    [
        ("B", "D", ("A", "A"), True),
        ("A", "D", ("A", "D"), True),
        ("A", "A", ("A", "A"), True),
        ("A", "D", ("A", "A"), False),
    ],
)
def test_ambiguous_effect_preserves_incumbent(
    visual: str,
    aligned: str,
    shifted: tuple[str, ...],
    nonempty: bool,
) -> None:
    decision = temporal_instrument_commit(
        incumbent="A",
        visual_only=visual,
        aligned_audio=aligned,
        shifted_audio=shifted,
        aligned_transcript_nonempty=nonempty,
    )
    assert not decision.override
    assert decision.prediction == "A"


def test_invalid_measurement_is_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_instrument_commit(
            incumbent="A",
            visual_only="",
            aligned_audio="D",
            shifted_audio=("A",),
            aligned_transcript_nonempty=True,
        )
