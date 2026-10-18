"""Score two frozen fixed300 arms without importing a model or training code.

Gold is read only after both arms pass the complete identity and prediction
consistency checks. It is a separate JSONL with exactly uid/video_id/gold per
row. The returned/written report contains aggregates, never individual answers.
This terminal scoring module must not be used to select a training checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any


FIXED_ROWS = 300
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261018
CHOICES = frozenset("ABCD")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _jsonl(path: Path, label: str) -> tuple[list[dict], str]:
    # Hash exactly the bytes scored, rather than rereading possibly changed files.
    content = path.read_bytes()
    rows = []
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _json(line)
        except ValueError as exc:
            raise ValueError(f"{label} line {number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {number}: object required")
        rows.append(row)
    if len(rows) != FIXED_ROWS:
        raise ValueError(f"{label} must contain exactly 300 rows")
    return rows, hashlib.sha256(content).hexdigest()


def _string(row: dict, key: str, label: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label}: {key} must be a string" + ("" if allow_empty else " and nonempty"))
    return value


class FixedIdentities:
    """Normalize only the aliases explicitly published in identity_mapping."""

    def __init__(self, path: Path):
        content = path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        rows = _json(content.decode("utf-8"))
        if not isinstance(rows, list) or len(rows) != FIXED_ROWS:
            raise ValueError("identity mapping must contain exactly 300 rows")
        self.rows: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}
        ordinals = set()
        for number, row in enumerate(rows, 1):
            label = f"identity mapping row {number}"
            if not isinstance(row, dict):
                raise ValueError(f"{label}: object required")
            qid = _string(row, "qid", label)
            video = _string(row, "video_id", label)
            ordinal = row.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise ValueError(f"{label}: integer ordinal required")
            if qid in self.rows or ordinal in ordinals:
                raise ValueError("Duplicate qid or ordinal in identity mapping")
            aliases = {qid}
            if row.get("sample_uid") is not None:
                aliases.add(_string(row, "sample_uid", label))
            for alias in aliases:
                if alias in self.aliases and self.aliases[alias] != qid:
                    raise ValueError("Ambiguous UID alias in identity mapping")
                self.aliases[alias] = qid
            videos = {video}
            if row.get("original_video_id") is not None:
                videos.add(_string(row, "original_video_id", label))
            self.rows[qid] = {"ordinal": ordinal, "video_id": video, "videos": videos}
            ordinals.add(ordinal)
        if ordinals != set(range(FIXED_ROWS)):
            raise ValueError("identity mapping ordinals must cover 0 through 299")
        self.order = sorted(self.rows, key=lambda qid: self.rows[qid]["ordinal"])

    def index(self, rows: list[dict], label: str) -> dict[str, dict]:
        indexed = {}
        for number, row in enumerate(rows, 1):
            prefix = f"{label} row {number}"
            uid = _string(row, "uid", prefix)
            qid = self.aliases.get(uid)
            if qid is None:
                raise ValueError(f"{prefix}: identity outside fixed300")
            if qid in indexed:
                raise ValueError(f"{prefix}: duplicate normalized identity")
            video = _string(row, "video_id", prefix)
            identity = self.rows[qid]
            if video not in identity["videos"]:
                raise ValueError(f"{prefix}: video identity differs from mapping")
            if "ordinal" in row:
                ordinal = row["ordinal"]
                if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != identity["ordinal"]:
                    raise ValueError(f"{prefix}: ordinal differs from mapping")
            indexed[qid] = row
        if set(indexed) != set(self.rows):
            raise ValueError(f"{label}: missing fixed300 identities")
        return indexed


def _load_arm(search_path: Path, final_path: Path, identities: FixedIdentities, label: str) -> dict:
    search_rows, search_hash = _jsonl(search_path, f"{label} search")
    final_rows, final_hash = _jsonl(final_path, f"{label} final")
    search = identities.index(search_rows, f"{label} search")
    final = identities.index(final_rows, f"{label} final")
    for ordinal, qid in enumerate(identities.order):
        prefix = f"{label} row {ordinal + 1}"
        original, adopted = search[qid], final[qid]
        prediction = _string(original, "prediction", prefix, allow_empty=True)
        state = _string(original, "state", prefix)
        if original.get("smoke") is True or original.get("fixed300_complete", True) is not True:
            raise ValueError(f"{prefix}: smoke or partial search cannot be scored")
        if _string(adopted, "original_prediction", prefix, allow_empty=True) != prediction:
            raise ValueError(f"{prefix}: final original_prediction differs from search")
        if _string(adopted, "search_state", prefix) != state:
            raise ValueError(f"{prefix}: final search_state differs from search")
        forced = adopted.get("has_forced_full_video")
        if not isinstance(forced, bool):
            raise ValueError(f"{prefix}: has_forced_full_video must be boolean")
        observer = _string(adopted, "observer_prediction", prefix, allow_empty=True)
        # Preserve the captured coordinator, which does not gate on status.
        take_observer = forced and observer in CHOICES
        expected_source = "DirectObserver" if take_observer else "OriginalSearch"
        expected_prediction = observer if take_observer else prediction
        if adopted.get("final_source") != expected_source:
            raise ValueError(f"{prefix}: final_source violates original adoption rule")
        if _string(adopted, "final_prediction", prefix, allow_empty=True) != expected_prediction:
            raise ValueError(f"{prefix}: final_prediction violates original adoption rule")
    return {"search": search, "final": final, "sha256": {"search": search_hash, "final": final_hash}}


def exact_mcnemar(gain: int, loss: int) -> float:
    """Exact two-sided Binomial(n=discordant, p=.5), including both tails."""
    if (isinstance(gain, bool) or isinstance(loss, bool)
            or not isinstance(gain, int) or not isinstance(loss, int)
            or gain < 0 or loss < 0):
        raise ValueError("Discordant counts must be nonnegative integers")
    discordant = gain + loss
    if not discordant:
        return 1.0
    tail_numerator = sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1))
    return min(1.0, 2 * tail_numerator / (1 << discordant))


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def paired_bootstrap(differences: list[int], *, seed: int = BOOTSTRAP_SEED,
                     iterations: int = BOOTSTRAP_ITERATIONS) -> list[float]:
    """Question-paired percentile CI of trained minus baseline accuracy.

    Each draw resamples complete question pairs, never each arm separately.
    The formal report always uses 10,000 replicates and the declared seed.
    """
    if not differences or any(value not in (-1, 0, 1) for value in differences):
        raise ValueError("Paired correctness differences must be -1, 0 or 1")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Positive bootstrap iterations required")
    generator = random.Random(seed)
    count = len(differences)
    draw = generator.randrange
    replicates = sorted(sum(differences[draw(count)] for _ in range(count)) / count
                        for _ in range(iterations))
    return [_percentile(replicates, .025), _percentile(replicates, .975)]


def paired_video_bootstrap(differences: list[int], canonical_video_ids: list[str], *,
                           seed: int = BOOTSTRAP_SEED,
                           iterations: int = BOOTSTRAP_ITERATIONS) -> list[float]:
    """Resample video clusters, retaining all paired questions in each draw.

    Each replicate draws as many videos as were observed, with replacement.
    Its accuracy difference is total paired difference / total question count;
    unequal cluster sizes therefore do not turn this into mean video accuracy.
    This allows arbitrary dependence within a video, assuming independent video
    clusters. It remains a small-sample sensitivity analysis, not a guarantee.
    """
    if (not differences or len(differences) != len(canonical_video_ids)
            or any(value not in (-1, 0, 1) for value in differences)):
        raise ValueError("One canonical video ID per paired -1/0/1 difference required")
    if any(not isinstance(video, str) or not video for video in canonical_video_ids):
        raise ValueError("Canonical video IDs must be nonempty strings")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Positive bootstrap iterations required")
    clusters: dict[str, list[int]] = {}
    for difference, video in zip(differences, canonical_video_ids):
        aggregate = clusters.setdefault(video, [0, 0])
        aggregate[0] += difference
        aggregate[1] += 1
    # Cluster IDs remain internal; the public report includes their count only.
    values = list(clusters.values())
    count = len(values)
    generator = random.Random(seed)
    draw = generator.randrange
    replicates = []
    for _ in range(iterations):
        difference_sum = question_count = 0
        for _ in range(count):
            difference, questions = values[draw(count)]
            difference_sum += difference
            question_count += questions
        replicates.append(difference_sum / question_count)
    replicates.sort()
    return [_percentile(replicates, .025), _percentile(replicates, .975)]


def _aggregate(baseline: list[bool], trained: list[bool], canonical_video_ids: list[str]) -> dict:
    if (len(baseline) != FIXED_ROWS or len(trained) != FIXED_ROWS
            or len(canonical_video_ids) != FIXED_ROWS):
        raise ValueError("Paired scoring requires exactly 300 results per arm")
    base_correct, trained_correct = sum(baseline), sum(trained)
    differences = [int(after) - int(before) for before, after in zip(baseline, trained)]
    gain, loss = differences.count(1), differences.count(-1)
    ci = paired_bootstrap(differences)
    video_ci = paired_video_bootstrap(differences, canonical_video_ids)
    return {
        "denominator": FIXED_ROWS,
        "baseline": {"correct": base_correct, "accuracy": base_correct / FIXED_ROWS},
        "trained": {"correct": trained_correct, "accuracy": trained_correct / FIXED_ROWS},
        "trained_only_gain": gain, "baseline_only_loss": loss, "net_correct": gain - loss,
        "both_correct": sum(before and after for before, after in zip(baseline, trained)),
        "neither_correct": sum(not before and not after for before, after in zip(baseline, trained)),
        "accuracy_difference": (gain - loss) / FIXED_ROWS,
        "accuracy_difference_percentage_points": 100 * (gain - loss) / FIXED_ROWS,
        "mcnemar": {"test": "exact two-sided binomial", "discordant_pairs": gain + loss,
                    "p_value": exact_mcnemar(gain, loss),
                    "assumption": "independent question pairs; within-video dependence can make p optimistic"},
        "paired_bootstrap": {
            "unit": "question", "replicates": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
            "method": "percentile; linear interpolation at (replicates - 1) * quantile",
            "assumption": "independent question pairs; does not account for shared videos",
            "confidence": .95, "accuracy_difference_ci": ci,
            "percentage_point_difference_ci": [100 * value for value in ci],
        },
        "paired_video_bootstrap": {
            "unit": "canonical video cluster", "video_clusters": len(set(canonical_video_ids)),
            "replicates": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
            "method": "resample complete video clusters; ratio of total paired difference to total questions; percentile CI",
            "assumption": "independent video clusters; arbitrary within-video dependence",
            "confidence": .95, "accuracy_difference_ci": video_ci,
            "percentage_point_difference_ci": [100 * value for value in video_ci],
        },
    }


def compare_fixed300(*, baseline_search: Path, baseline_final: Path, trained_search: Path,
                     trained_final: Path, identity_mapping: Path, gold: Path) -> dict:
    """Validate both complete arms before opening terminal-only gold input."""
    identities = FixedIdentities(identity_mapping)
    baseline = _load_arm(baseline_search, baseline_final, identities, "baseline")
    trained = _load_arm(trained_search, trained_final, identities, "trained")
    # This is deliberately the first read of the separate Gold file.
    gold_rows, _ = _jsonl(gold, "gold")
    if any(set(row) != {"uid", "video_id", "gold"} for row in gold_rows):
        raise ValueError("Gold rows must contain only uid, video_id and gold")
    answers = identities.index(gold_rows, "gold")
    if any(not isinstance(row.get("gold"), str) or row["gold"] not in CHOICES
           for row in answers.values()):
        raise ValueError("Gold must be one exact uppercase A-D choice per identity")

    def correctness(arm: dict, stage: str) -> list[bool]:
        result = []
        for qid in identities.order:
            search, final = arm["search"][qid], arm["final"][qid]
            if stage == "search":
                correct = search["state"] == "completed" and search["prediction"] == answers[qid]["gold"]
            elif final["final_source"] == "DirectObserver":
                # A legal Observer adoption can rescue a failed search.
                correct = final["final_prediction"] == answers[qid]["gold"]
            else:
                correct = search["state"] == "completed" and final["final_prediction"] == answers[qid]["gold"]
            result.append(correct)
        return result

    report = {
        "schema": "packet-opsd-fixed300-paired-v1", "state": "COMPLETE",
        "denominator": FIXED_ROWS, "direction": "trained minus baseline",
        "gold_access": "terminal scoring only; never sent to a model or checkpoint selection",
        "failed_search_policy": "state != completed is incorrect; retained in denominator",
        "final_failure_policy": "OriginalSearch inherits search failure; valid DirectObserver may rescue",
        "identity_mapping_sha256": identities.sha256,
        "input_artifact_sha256": {
            f"{label}_{stage}": arm["sha256"][stage]
            for label, arm in (("baseline", baseline), ("trained", trained))
            for stage in ("search", "final")
        },
        "arms": {
            label: {"failed_search_rows": sum(row["state"] != "completed" for row in arm["search"].values()),
                    "observer_adopted": sum(row["final_source"] == "DirectObserver" for row in arm["final"].values())}
            for label, arm in (("baseline", baseline), ("trained", trained))
        },
    }
    canonical_video_ids = [identities.rows[qid]["video_id"] for qid in identities.order]
    for stage in ("search", "final"):
        report[stage] = _aggregate(correctness(baseline, stage), correctness(trained, stage), canonical_video_ids)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-search", "baseline-final", "trained-search", "trained-final",
                 "identity-mapping", "gold", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite a paired report")
    report = compare_fixed300(**{key: value for key, value in vars(args).items() if key != "output"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "denominator": report["denominator"],
                      "search_net_correct": report["search"]["net_correct"],
                      "final_net_correct": report["final"]["net_correct"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
