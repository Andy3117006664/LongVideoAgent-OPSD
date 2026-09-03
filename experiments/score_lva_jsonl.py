#!/usr/bin/env python3
"""Score LongVideoAgent validation/rollout JSONL by option accuracy.

The trainer's JSONL dump has one object per trajectory with ``output`` and
``gts`` fields.  This utility intentionally counts every line in the
denominator: malformed or missing ``<answer>`` tags are incorrect rather than
silently dropped.  Rollout dumps can contain repeated trajectories (N>1), so
the optional ``sample_uid``/``qid`` grouping reports a separate majority vote
number without replacing the per-trajectory score.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ANSWER_RE = re.compile(r"<answer>\s*(?:option\s*)?(a?[0-4])\b.*?</answer>", re.I | re.S)


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from trainer JSONL and the evaluator's JSON summary.

    The patched trainer writes one object per line.  The legacy local
    evaluator instead writes ``{"results": [...]}``; accepting that shape
    here makes it harder to accidentally score the metadata wrapper as one
    QA.  A JSON array is accepted as a convenience for hand-built smoke
    files, while malformed records still fail closed.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            for index, obj in enumerate(parsed["results"], 1):
                if not isinstance(obj, dict):
                    raise ValueError(f"expected object in results at {path}:{index}")
                yield obj
            return
        if isinstance(parsed, list):
            for index, obj in enumerate(parsed, 1):
                if not isinstance(obj, dict):
                    raise ValueError(f"expected object at {path}:{index}")
                yield obj
            return
        # A single JSON object may be a valid one-row dump.  If parsing failed
        # we fall through to line-wise JSONL handling so a large JSONL file
        # whose first line starts with ``{`` is not misclassified.
        if isinstance(parsed, dict):
            yield parsed
            return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield obj


def _option(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\ba?([0-4])\b", str(value).strip(), re.I)
    if not match:
        return None
    return f"a{match.group(1)}"


def _first_field(row: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-null field among aliases."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _normalise_question(value: Any) -> str:
    """Use the same whitespace-insensitive key for evaluator summaries."""

    return " ".join(str(value or "").strip().split())


def _load_gold_questions(path: Path | None) -> dict[str, list[str]]:
    """Load source QA labels for the legacy evaluator's summary JSON.

    The official ``evaluate_*`` scripts write ``results`` entries containing
    ``question`` and ``final_answer`` but do not copy the gold label.  Passing
    ``--questions-path`` lets this scorer join those rows back to the source
    LongTVQA/LongTVQA+ file.  Values are kept as queues so duplicate question
    strings are consumed in source order rather than silently overwritten.
    """

    if path is None:
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        source_rows = parsed
    elif isinstance(parsed, dict):
        # A few normalized exports wrap rows under a conventional key.
        source_rows = parsed.get("questions") or parsed.get("data") or []
    else:
        source_rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid questions JSON at {path}:{line_no}: {exc}") from exc
                source_rows.append(item)

    mapping: dict[str, list[str]] = defaultdict(list)
    for item in source_rows:
        if not isinstance(item, dict):
            continue
        question = _first_field(item, "question", "q")
        gold_value = _first_field(item, "answer", "answer_idx", "label", "ground_truth")
        if isinstance(gold_value, dict):
            gold_value = _first_field(gold_value, "answer", "answer_idx", "label", "ground_truth")
        gold = _option(gold_value)
        key = _normalise_question(question)
        if key and gold is not None:
            mapping[key].append(gold)
    return mapping


def _prediction(output: Any) -> str | None:
    matches = ANSWER_RE.findall(str(output or ""))
    if not matches:
        return None
    return _option(matches[-1])


def _group_key(row: dict[str, Any], index: int) -> str:
    for key in ("sample_uid", "qid", "uid"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    # A validation dump with n=1 has no need for grouping; this fallback keeps
    # each line distinct instead of accidentally merging unrelated questions.
    return f"line:{index}"


def score(path: Path, gold_questions: Path | None = None) -> dict[str, Any]:
    rows = list(_rows(path))
    gold_map = _load_gold_questions(gold_questions)
    # Work on a copy: source queues are consumed only for rows lacking an
    # embedded gold label, preserving embedded trainer labels untouched.
    gold_queues = {key: list(values) for key, values in gold_map.items()}
    correct = 0
    invalid = 0
    groups: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for index, row in enumerate(rows):
        # Trainer dumps use output/gts; legacy evaluator summaries use
        # final_answer and answer/answer_idx.  Keep prediction extraction
        # tag-based so an arbitrary option mentioned in reasoning is never
        # counted as the final answer.
        pred_value = _first_field(row, "output", "final_answer", "response", "prediction")
        pred = _prediction(pred_value)
        # ``final_answer`` in the legacy summary is already post-processed
        # (usually ``a0``–``a4``) and has no XML wrapper.  Only use this direct
        # fallback for that explicit field; never infer an answer from free
        # form ``output``/``response`` text.
        if pred is None and "output" not in row and "response" not in row:
            pred = _option(pred_value)
        gold_value = _first_field(
            row, "gts", "ground_truth", "gt", "answer", "answer_idx", "label"
        )
        if isinstance(gold_value, dict):
            gold_value = _first_field(gold_value, "answer", "ground_truth", "answer_idx", "label")
        if gold_value is None and gold_questions is not None:
            question = _first_field(row, "question", "q")
            queue = gold_queues.get(_normalise_question(question), [])
            if queue:
                gold_value = queue.pop(0)
        gold = _option(gold_value)
        if pred is None or gold is None:
            invalid += 1
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        groups[_group_key(row, index)].append((pred, gold))

    majority_correct = 0
    majority_total = 0
    for choices in groups.values():
        # Only use majority when all trajectories in a group share a label;
        # otherwise the group has no evaluable gold and is counted invalid.
        golds = {gold for _, gold in choices if gold is not None}
        if len(golds) != 1:
            continue
        gold = next(iter(golds))
        preds = [pred for pred, _ in choices if pred is not None]
        majority = Counter(preds).most_common(1)[0][0] if preds else None
        majority_total += 1
        majority_correct += int(majority == gold)

    return {
        "file": str(path),
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "invalid_or_missing_answer": invalid,
        "groups": len(groups),
        "majority_groups": majority_total,
        "majority_correct": majority_correct,
        "majority_accuracy": majority_correct / majority_total if majority_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help=(
            "Optional LongTVQA/LongTVQA+ source JSON/JSONL. Required when "
            "scoring official evaluator summaries whose results omit gold labels."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(score(args.jsonl, args.questions_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
