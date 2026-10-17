#!/usr/bin/env python3
"""Run an answer-blind oracle ladder over annotated LVQA evidence spans.

This is a diagnostic, not an online method or benchmark score.  Reference
spans are used only to construct an offline upper-bound evidence packet.  The
benchmark answer is never placed in a model request.  The ladder separates:

1. reference_raw: can the frozen visual model answer from the right pixels?
2. witness_answer: can an option-blind witness observer plus the frozen
   VideoSEAL planner compose those pixels into a better answer?
3. protected_answer: can a deterministic support certificate retain gains
   without overwriting an incumbent on ambiguous evidence?

The output is append-only and resumable.  Gold and baseline correctness are
stored for a separate offline reducer only after all model calls finish.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OPTION_LINE_RE = re.compile(r"(?m)^\s*\([A-H]\)\s+")
VALID_OPTIONS = tuple("ABCDEFGH")
WRITE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def latest_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("video_id") or ""), str(row.get("uid") or row.get("qa_uid") or ""))
        if all(key):
            output[key] = row
    return output


def load_tasks(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if "extra_info" not in table.column_names:
        raise ValueError(f"missing extra_info column: {path}")
    tasks: list[dict[str, Any]] = []
    for raw in table["extra_info"].to_pylist():
        info = dict(raw or {})
        video_id = str(info.get("video_id") or info.get("video_key") or "").strip()
        uid = str(info.get("qa_uid") or info.get("uid") or "").strip()
        question = str(info.get("question") or "").strip()
        if not (video_id and uid and question):
            continue
        info["video_id"] = video_id
        info["uid"] = uid
        info["question"] = question
        tasks.append(info)
    return tasks


def parse_clock(value: str) -> float | None:
    parts = value.strip().split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if len(values) == 2:
        return 60.0 * values[0] + values[1]
    if len(values) == 3:
        return 3600.0 * values[0] + 60.0 * values[1] + values[2]
    return None


def parse_reference_spans(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        spans: list[tuple[float, float]] = []
        for item in value:
            if isinstance(item, dict):
                start = item.get("start", item.get("start_time"))
                end = item.get("end", item.get("end_time"))
                try:
                    pair = (float(start), float(end))
                except (TypeError, ValueError):
                    spans.extend(parse_reference_spans(item))
                else:
                    spans.append((min(pair), max(pair)))
            else:
                spans.extend(parse_reference_spans(item))
        return list(dict.fromkeys(spans))
    text = str(value or "")
    pattern = re.compile(
        r"(\d{1,4}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(?:-|–|—|to)\s*"
        r"(\d{1,4}:\d{2}(?::\d{2}(?:\.\d+)?)?)",
        re.I,
    )
    spans = []
    for match in pattern.finditer(text):
        start, end = parse_clock(match.group(1)), parse_clock(match.group(2))
        if start is not None and end is not None:
            spans.append((min(start, end), max(start, end)))
    return list(dict.fromkeys(spans))


def question_stem(question: str) -> str:
    match = OPTION_LINE_RE.search(question)
    return question[: match.start()].strip() if match else question.strip()


def option_letters(question: str) -> list[str]:
    letters = re.findall(r"(?m)^\s*\(([A-H])\)\s+", question.upper())
    return list(dict.fromkeys(letters))


def resolve_video(task: dict[str, Any]) -> Path | None:
    candidates = [
        Path(str(task.get("VIDEO_PATH") or "")),
        ROOT / "datasets" / "LVBench" / "video_chunks" / "extracted" / str(task.get("video_file") or ""),
        ROOT / "datasets" / "LVBench" / "video_chunks" / "extracted" / f"{task['video_id']}.mp4",
        ROOT / "datasets" / "CG-Bench" / "videos" / f"{task['video_id']}.mp4",
    ]
    for candidate in candidates:
        try:
            if str(candidate) and candidate.is_file():
                return candidate
        except OSError:
            # Dataset manifests may retain a stale /nfs alias after the OFS
            # volume is remounted at the equivalent project-local /home path.
            continue
    return None


def media_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def coverage_uniform_timestamps(
    spans: Iterable[tuple[float, float]], *, max_frames: int, duration: float
) -> list[float]:
    """Allocate frames by span duration, then cover each complete interval.

    Unlike the baseline's fixed-FPS prefix truncation, this diagnostic sampler
    never turns a long requested interval into its first ``max_frames / fps``
    seconds.  It is used only for the privileged reference-span oracle.
    """
    valid: list[tuple[float, float]] = []
    for raw_start, raw_end in spans:
        start, end = max(0.0, float(raw_start)), max(0.0, float(raw_end))
        if duration > 0.0:
            start, end = min(start, duration), min(end, duration)
        if end < start:
            start, end = end, start
        if end <= start:
            end = min(duration, start + 1.0) if duration > 0.0 else start + 1.0
        if end > start:
            valid.append((start, end))
    if not valid or max_frames <= 0:
        return []

    widths = [max(end - start, 1e-3) for start, end in valid]
    allocation = [1 for _ in valid]
    remaining = max(0, max_frames - len(valid))
    total_width = sum(widths)
    raw_extra = [remaining * width / total_width for width in widths]
    floors = [int(math.floor(value)) for value in raw_extra]
    allocation = [base + extra for base, extra in zip(allocation, floors)]
    leftover = remaining - sum(floors)
    order = sorted(range(len(valid)), key=lambda i: (raw_extra[i] - floors[i], widths[i]), reverse=True)
    for index in order[:leftover]:
        allocation[index] += 1

    timestamps: list[float] = []
    for (start, end), count in zip(valid, allocation):
        if count == 1:
            timestamps.append(0.5 * (start + end))
        else:
            timestamps.extend(start + (end - start) * index / (count - 1) for index in range(count))
    return [round(value, 3) for value in sorted(dict.fromkeys(timestamps))[:max_frames]]


def extract_frames(video: Path, timestamps: list[float], output_dir: Path) -> tuple[list[Path], list[float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    actual: list[float] = []
    for index, timestamp in enumerate(timestamps):
        target = output_dir / f"frame_{index:03d}.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=768:-2", "-q:v", "3", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
        if result.returncode == 0 and target.is_file() and target.stat().st_size:
            frames.append(target)
            actual.append(timestamp)
    return frames, actual


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_chat(
    *, base_url: str, api_key: str, model: str, content: str | list[dict[str, Any]],
    timeout: float, max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"] or "").strip()


def parse_object(text: str) -> dict[str, Any]:
    # MLLM servers may prepend <think> text or append a short explanation.
    # Greedy regex extraction is invalid when the reasoning contains braces;
    # instead scan every opening brace with JSONDecoder.raw_decode.
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON object in response: {text[:300]}")


def normalize_answer(value: Any, letters: Iterable[str]) -> str:
    valid = set(letters)
    text = str(value or "").strip().upper()
    return text if text in valid else "unknown"


def parse_raw_response(text: str, letters: list[str]) -> dict[str, Any]:
    """Recover the answer even when a verbose JSON response is truncated."""
    try:
        value = parse_object(text)
    except ValueError:
        match = re.search(
            r'(?:"answer"\s*:\s*"|(?:final_)?answer\s*[:=]\s*)([A-H]|unknown)\b',
            text,
            re.I,
        )
        if not match:
            return {"answer": "unknown", "confidence": None, "reason": "response did not contain a parseable final answer"}
        return {"answer": normalize_answer(match.group(1), letters), "confidence": None, "reason": "recovered from truncated response"}
    return {
        "answer": normalize_answer(value.get("answer"), letters),
        "confidence": value.get("confidence"),
        "reason": str(value.get("reason") or "")[:1000],
    }


def raw_prompt(question: str) -> str:
    return f"""You are the frozen visual answerer in a long-video QA diagnostic.

Use only the supplied timestamped frames. Select one option only when the
pixels support it; otherwise answer unknown. Do not use outside knowledge.

Question:
{question}

Return exactly one JSON object with only three fields: `answer`, `confidence`,
and `reason`. `answer` must be one option letter or `unknown`; `reason` must be
at most 20 words. Do not list frame timestamps.
"""


def witness_prompt(stem: str) -> str:
    return f"""You are the OBSERVER in a two-agent long-video QA diagnostic.

You see timestamped frames and the question stem, but no answer choices and no
benchmark answer. Record only directly visible, question-relevant facts.
Keep repeated-looking people, objects, and event instances separate unless
the frames provide an identity link. Explicitly record temporal order and
unresolved identity/relation links. Never guess an answer option.

Question stem:
{stem}

Return exactly one JSON object with these fields: `events`, `links`,
`global_facts`, and `open_questions`.

- Each event must contain `id`, `start_time`, `end_time`, `description`,
  `entities`, and `visual_cues`.
- Each link must contain `source`, `target`, `type`, `claim`, and
  `provenance_times`. The type must be one of same_entity, different_entity,
  before, after, during, overlap, interaction, state_change, or unknown.
- Each global fact must contain `claim` and `provenance_times`.
- Open questions are short strings.

Do not copy schema descriptions or placeholder phrases into the values. If a
field has no directly observed content, return an empty list for that field.
"""


def adjudicator_prompt(question: str, witness: dict[str, Any], letters: list[str]) -> str:
    assessments = ",".join(
        f'{{"option":"{letter}","status":"SUPPORTED|REFUTED|UNKNOWN","evidence_ids":["e1"]}}'
        for letter in letters
    )
    return f"""You are the RELATION LINKER and answer adjudicator.

Use only the option-blind witness ledger below. Do not use outside knowledge
or infer facts absent from the ledger. Check identity and temporal links before
combining events. Assess every option independently.

Question and choices:
{question}

Witness ledger:
{json.dumps(witness, ensure_ascii=False, separators=(',', ':'))}

Return exactly one JSON object:
{{"option_assessment":[{assessments}],"answer":"{'|'.join(letters)}|unknown","reason":"brief ledger-grounded relation"}}
"""


def normalize_assessments(value: Any, letters: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            option = str(item.get("option") or "").strip().upper()
            status = str(item.get("status") or "UNKNOWN").strip().upper()
            if option in letters and status in {"SUPPORTED", "REFUTED", "UNKNOWN"}:
                result[option] = status
    return result


def certificate_predictions(
    *, baseline: str, candidate: str, assessments: dict[str, str], letters: list[str]
) -> tuple[str, str, bool, bool]:
    supported = [letter for letter in letters if assessments.get(letter) == "SUPPORTED"]
    unique = len(supported) == 1 and candidate == supported[0]
    strict = unique and all(assessments.get(letter) == "REFUTED" for letter in letters if letter != candidate)
    support_prediction = candidate if unique else baseline
    strict_prediction = candidate if strict else baseline
    return support_prediction, strict_prediction, unique, strict


def content_with_frames(prompt: str, frames: list[Path], timestamps: list[float]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame, timestamp in zip(frames, timestamps):
        content.append({"type": "text", "text": f"[Observed timestamp: {timestamp:.3f} seconds]"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(frame)}})
    return content


def run_task(task: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    key = (task["video_id"], task["uid"])
    row: dict[str, Any] = {
        "schema_version": "reference-oracle-ladder-v1",
        "video_id": key[0],
        "uid": key[1],
        "question": task["question"],
        "baseline_prediction": str(baseline.get("prediction") or "").strip().upper(),
        "diagnostic_status": "failed",
        "gold_exposed_to_models": False,
    }
    spans = parse_reference_spans(task.get("time_reference") or task.get("clue_intervals"))
    video = resolve_video(task)
    letters = option_letters(task["question"])
    if not spans:
        row["diagnostic_status"] = "reference_missing"
    elif video is None:
        row["diagnostic_status"] = "video_missing"
    elif not letters:
        row["diagnostic_status"] = "options_missing"
    else:
        duration = media_duration(video)
        timestamps = coverage_uniform_timestamps(spans, max_frames=args.max_frames, duration=duration)
        with tempfile.TemporaryDirectory(prefix="oracle-ladder-", dir=args.temp_root) as directory:
            frames, actual = extract_frames(video, timestamps, Path(directory))
            row["frame_count"] = len(frames)
            row["reference_spans_for_offline_diagnostic"] = [list(span) for span in spans]
            row["sampled_timestamps"] = actual
            if not frames:
                row["diagnostic_status"] = "frame_extract_failed"
            else:
                try:
                    raw_text = request_chat(
                        base_url=args.visual_base_url, api_key=args.api_key, model=args.visual_model,
                        content=content_with_frames(raw_prompt(task["question"]), frames, actual),
                        timeout=args.timeout, max_tokens=260,
                    )
                    raw_value = parse_raw_response(raw_text, letters)
                    row["reference_raw"] = {
                        "answer": raw_value["answer"],
                        "confidence": raw_value.get("confidence"),
                        "reason": str(raw_value.get("reason") or "")[:1000],
                    }

                    if args.raw_only:
                        row["diagnostic_status"] = "ok"
                    else:
                        witness_text = request_chat(
                            base_url=args.visual_base_url, api_key=args.api_key, model=args.visual_model,
                            content=content_with_frames(witness_prompt(question_stem(task["question"])), frames, actual),
                            timeout=args.timeout, max_tokens=1200,
                        )
                        witness = parse_object(witness_text)
                        row["witness"] = witness

                        adjudicator_text = request_chat(
                            base_url=args.planner_base_url, api_key=args.api_key, model=args.planner_model,
                            content=adjudicator_prompt(task["question"], witness, letters),
                            timeout=args.timeout, max_tokens=700,
                        )
                        adjudicator = parse_object(adjudicator_text)
                        candidate = normalize_answer(adjudicator.get("answer"), letters)
                        assessments = normalize_assessments(adjudicator.get("option_assessment"), letters)
                        support, strict, unique, closed = certificate_predictions(
                            baseline=row["baseline_prediction"], candidate=candidate,
                            assessments=assessments, letters=letters,
                        )
                        row["witness_adjudication"] = {
                            "answer": candidate,
                            "assessments": assessments,
                            "reason": str(adjudicator.get("reason") or "")[:1000],
                            "unique_support_certificate": unique,
                            "strict_closed_certificate": closed,
                        }
                        row["support_protected_prediction"] = support
                        row["strict_protected_prediction"] = strict
                        row["diagnostic_status"] = "ok"
                except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"

    # Attach labels only after every answer-blind model request has completed.
    row["ground_truth_for_offline_reducer_only"] = str(
        task.get("ground_truth") or baseline.get("ground_truth") or ""
    ).strip().upper()
    row["baseline_correct_for_offline_reducer_only"] = bool(baseline.get("correct"))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-base-url", default="http://127.0.0.1:18101/v1")
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--visual-model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--planner-model", default="VideoSEAL_8B")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--only-baseline-wrong", action="store_true", help="Select only rows where frozen baseline is wrong (offline diagnostic stratification).")
    parser.add_argument("--max-reference-width", type=float, help="Exclude global/coarse references wider than this many seconds.")
    parser.add_argument("--raw-only", action="store_true", help="Run only the reference-pixel visual answer stage.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temp-root", type=Path, default=ROOT / ".tmp" / "diagnostics" / "oracle_ladder_frames")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    baseline = latest_results(args.frozen_results)
    done = {
        (str(row.get("video_id") or ""), str(row.get("uid") or ""))
        for row in read_jsonl(args.output)
        if str(row.get("diagnostic_status")) in {"ok", "reference_missing", "video_missing", "options_missing"}
    }
    tasks = [task for task in load_tasks(args.parquet) if (task["video_id"], task["uid"]) in baseline]
    if args.only_baseline_wrong:
        tasks = [
            task for task in tasks
            if str(baseline[(task["video_id"], task["uid"])].get("prediction") or "").strip().upper()
            != str(task.get("ground_truth") or "").strip().upper()
        ]
    if args.max_reference_width is not None:
        tasks = [
            task for task in tasks
            if (spans := parse_reference_spans(task.get("time_reference") or task.get("clue_intervals")))
            and sum(end - start for start, end in spans) <= max(0.0, args.max_reference_width)
        ]
    tasks.sort(key=lambda task: hashlib.sha256(f"{task['video_id']}:{task['uid']}".encode()).hexdigest())
    if args.max_items is not None:
        tasks = tasks[: max(0, args.max_items)]
    pending = [task for task in tasks if (task["video_id"], task["uid"]) not in done]

    mode = "a" if args.output.exists() else "w"
    completed = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_task, task, baseline[(task["video_id"], task["uid"])], args): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                with WRITE_LOCK:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                completed += 1
                print(json.dumps({
                    "completed_this_run": completed,
                    "pending_this_run": len(pending),
                    "video_id": row["video_id"],
                    "uid": row["uid"],
                    "status": row["diagnostic_status"],
                }), flush=True)
    print(json.dumps({"selected": len(tasks), "already_done": len(tasks) - len(pending), "completed_this_run": completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
