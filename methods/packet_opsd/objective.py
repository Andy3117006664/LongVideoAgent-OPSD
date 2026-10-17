"""Packet MMD estimator. Candidate generation never calls an environment tool."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass


# Hold the existing tool type fixed, while retaining its free JSON parameters.
# Closing Qwen3's opening think block avoids sampling a second rationale here.
INSPECTION_PREFIX = '</think>\n<tool_call>{"name":"visual_inspect","arguments":'


def hindsight_messages(messages: list[dict], observation: dict) -> list[dict]:
    """Only the natural next tool response; no terminal gold or new pixels."""
    result = copy.deepcopy(messages)
    output = {key: observation.get(key) for key in ("name", "ok", "output", "error", "metadata")}
    note = (
        "\n\n<observed_inspection_feedback>\n"
        + json.dumps(output, ensure_ascii=False)
        + "\n</observed_inspection_feedback>\n"
        "Reconsider the preceding inspection using only this feedback and the existing history. "
        "Choose its time spans within the unchanged 2 fps / 64-frame shared budget. "
        "Do not invent unseen evidence or answer the question yourself."
    )
    # Keep the original student request intact. The teacher receives privileged
    # hindsight in the user message, not as a fabricated tool execution.
    result[-1]["content"] += note
    return result


def candidate_spans(completion) -> list[dict] | None:
    """Invalid/truncated draws remain invalid, without rejection resampling."""
    if not completion.complete:
        return None
    text = completion.text
    marker = "</tool_call>"
    if marker not in text:
        return None
    body, tail = text.split(marker, 1)
    if tail.strip():
        return None
    try:
        # Prefix contains the outer call and 'arguments:', so the generation
        # must close both its argument dictionary and the outer call.
        call = json.loads(INSPECTION_PREFIX.split("<tool_call>", 1)[1] + body)
    except (ValueError, TypeError):
        return None
    args = call.get("arguments")
    if call.get("name") != "visual_inspect" or not isinstance(args, dict):
        return None
    spans = args.get("spans")
    if not isinstance(spans, list) or not spans:
        return None
    return spans


@dataclass(frozen=True)
class PacketCoefficient:
    value: float
    student_similarity: float
    teacher_similarity: float


def packet_coefficient(student, independent_student, teacher, *, kernel) -> PacketCoefficient:
    """Unbiased gradient coefficient for MMD², conditional on fixed H and F.

    a,a' are independent fresh draws from current p; b is from frozen q.
    2 [k(S(a),S(a')) - k(S(a),S(b))] * grad log p(a).
    The factual action that generated F is intentionally NOT used as a draw.
    This coefficient/surrogate is not an unbiased nonnegative MMD value.
    """
    pp = float(kernel(student, independent_student))
    pq = float(kernel(student, teacher))
    return PacketCoefficient(2.0 * (pp - pq), pp, pq)
