from __future__ import annotations

from dataclasses import asdict, dataclass


LETTERS = frozenset("ABCD")


@dataclass(frozen=True)
class InstrumentDecision:
    prediction: str
    incumbent: str
    candidate: str
    override: bool
    reason: str
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def temporal_instrument_commit(
    *,
    incumbent: str,
    visual_only: str,
    aligned_audio: str,
    shifted_audio: tuple[str, ...],
    aligned_transcript_nonempty: bool,
) -> InstrumentDecision:
    """Permit an override only when temporal alignment is the unique cause.

    The real visual packet is identical in every arm.  The aligned transcript
    may replace the incumbent only if removing audio and independently shifting
    it to every negative-control time both restore the incumbent.
    """

    values = (incumbent, visual_only, aligned_audio, *shifted_audio)
    if incumbent not in LETTERS:
        raise ValueError("incumbent must be A-D")
    if any(value not in LETTERS for value in values[1:]):
        raise ValueError("every measured prediction must be A-D")
    if not shifted_audio:
        raise ValueError("at least one shifted-audio control is required")

    candidate_differs = aligned_audio != incumbent
    visual_preserves = visual_only == incumbent
    shifted_preserve = all(value == incumbent for value in shifted_audio)
    controls_agree = len(set((visual_only, *shifted_audio))) == 1
    override = (
        aligned_transcript_nonempty
        and candidate_differs
        and visual_preserves
        and shifted_preserve
        and controls_agree
    )
    return InstrumentDecision(
        prediction=aligned_audio if override else incumbent,
        incumbent=incumbent,
        candidate=aligned_audio,
        override=override,
        reason="alignment_specific_modality_effect" if override else "preserve_incumbent",
        diagnostics={
            "aligned_transcript_nonempty": aligned_transcript_nonempty,
            "candidate_differs": candidate_differs,
            "visual_only_preserves_incumbent": visual_preserves,
            "all_shifted_controls_preserve_incumbent": shifted_preserve,
            "all_negative_controls_agree": controls_agree,
            "shifted_control_count": len(shifted_audio),
        },
    )

