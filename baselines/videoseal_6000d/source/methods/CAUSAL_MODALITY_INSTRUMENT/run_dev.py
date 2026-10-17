#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any
from urllib import request


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "VideoSEAL"), str(ROOT / "research" / "diagnostics")]

from methods.CAUSAL_EVIDENCE_ORBITS.frames import extract_packet  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.io import read_baseline  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.schema import CandidateSpan  # noqa: E402
from methods.CAUSAL_MODALITY_INSTRUMENT.router import temporal_instrument_commit  # noqa: E402


WRITE_LOCK = threading.Lock()


def image_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_answer(raw: str) -> str:
    value = str(raw or "").strip().upper()
    return value[:1] if value[:1] in set("ABCD") else ""


def ask(
    args: argparse.Namespace,
    *,
    question: str,
    frames: list[Path],
    timestamps: list[float],
    transcript: str,
) -> tuple[str, str]:
    prompt = (
        "Answer the multiple-choice question using only the supplied timestamped evidence. "
        "Transcript evidence is time-aligned speech and may contain recognition errors. "
        "Return exactly one token: A, B, C, or D.\n\n"
        f"Question:\n{question}\n\nTranscript evidence:\n{transcript or '[none supplied]'}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame, timestamp in zip(frames, timestamps):
        content.append({"type": "text", "text": f"[video t={timestamp:.3f}s]"})
        content.append({"type": "image_url", "image_url": {"url": image_url(frame), "detail": "low"}})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        method="POST",
    )
    with request.urlopen(req, timeout=args.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = str(body["choices"][0]["message"]["content"] or "").strip()
    return parse_answer(raw), raw


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def shifted_spans(spans: list[list[float]], duration: float, fraction: float) -> list[list[float]]:
    output = []
    for start, end in spans:
        width = end - start
        usable = max(width, duration - width)
        shifted = (start + duration * fraction) % usable
        output.append([shifted, min(duration, shifted + width)])
    return output


def transcribe(model: Any, video: Path, spans: list[list[float]], directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, (start, end) in enumerate(spans):
        audio = directory / f"{index:02d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                "-y", str(audio),
            ],
            check=True,
        )
        segments, _ = model.transcribe(
            str(audio), beam_size=5, vad_filter=True, condition_on_previous_text=False
        )
        text = " ".join(str(segment.text or "").strip() for segment in segments).strip()
        if text:
            lines.append(f"[{start:.1f}-{end:.1f}s] {text}")
    return "\n".join(lines)


def run_one(
    *,
    asr: Any,
    asr_lock: threading.Lock,
    helper: Any,
    item: dict[str, Any],
    task: dict[str, Any],
    baseline: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    uid = str(item["uid"])
    video = helper.resolve_video(task)
    if video is None:
        raise FileNotFoundError(f"video unavailable for {uid}")
    spans = [[float(a), float(b)] for a, b in item["spans"]]
    duration = video_duration(video)
    negative_spans = [
        shifted_spans(spans, duration, 1.0 / 3.0),
        shifted_spans(spans, duration, 2.0 / 3.0),
    ]
    candidates = tuple(
        CandidateSpan(f"{a:g}_{b:g}", a, b, rank)
        for rank, (a, b) in enumerate(spans, 1)
    )
    with tempfile.TemporaryDirectory(prefix=f"cmi-{uid}-", dir=args.temp_root) as raw:
        directory = Path(raw)
        frames, timestamps, _ = extract_packet(
            helper, video, candidates, directory / "frames", fps=args.fps, cap=args.frames
        )
        with asr_lock:
            aligned_text = transcribe(asr, video, spans, directory / "aligned")
            shifted_texts = [
                transcribe(asr, video, values, directory / f"shifted_{index}")
                for index, values in enumerate(negative_spans)
            ]
        visual_prediction, visual_raw = ask(
            args,
            question=baseline["question"],
            frames=frames,
            timestamps=timestamps,
            transcript="",
        )
        aligned_prediction, aligned_raw = ask(
            args,
            question=baseline["question"],
            frames=frames,
            timestamps=timestamps,
            transcript=aligned_text,
        )
        shifted_measurements = [
            ask(
                args,
                question=baseline["question"],
                frames=frames,
                timestamps=timestamps,
                transcript=text,
            )
            for text in shifted_texts
        ]
    incumbent = str(baseline["prediction"]).upper()
    decision = temporal_instrument_commit(
        incumbent=incumbent,
        visual_only=visual_prediction,
        aligned_audio=aligned_prediction,
        shifted_audio=tuple(value[0] for value in shifted_measurements),
        aligned_transcript_nonempty=bool(aligned_text.strip()),
    )
    return {
        "schema_version": "causal-modality-instrument-result-v1",
        "video_id": str(item["video_id"]),
        "uid": uid,
        "question": str(baseline["question"]),
        "incumbent": incumbent,
        "spans": spans,
        "negative_spans": negative_spans,
        "transcripts": {"aligned": aligned_text, "shifted": shifted_texts},
        "measurements": {
            "visual_only": {"prediction": visual_prediction, "raw": visual_raw},
            "aligned_audio": {"prediction": aligned_prediction, "raw": aligned_raw},
            "shifted_audio": [
                {"prediction": prediction, "raw": raw_value}
                for prediction, raw_value in shifted_measurements
            ],
        },
        "decision": decision.to_dict(),
        "frame_count": len(frames),
        "ground_truth_for_offline_reducer_only": str(baseline.get("ground_truth") or "").upper(),
        "status": "ok",
        "information_boundary": {
            "gold_online": False,
            "baseline_correctness_online": False,
            "time_reference_online": False,
            "same_visual_packet_across_arms": True,
        },
    }


def summarize(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    report: dict[str, Any] = {
        "schema_version": "causal-modality-instrument-summary-v1",
        "coverage": {"expected": expected, "valid": len(valid), "failed": expected - len(valid)},
        "arms": {},
    }
    paths = {
        "incumbent": lambda row: row["incumbent"],
        "visual_only": lambda row: row["measurements"]["visual_only"]["prediction"],
        "aligned_audio": lambda row: row["measurements"]["aligned_audio"]["prediction"],
        "candidate": lambda row: row["decision"]["prediction"],
    }
    for name, getter in paths.items():
        pairs = [(row, getter(row)) for row in valid]
        report["arms"][name] = {
            "n": len(pairs),
            "correct": sum(prediction == row["ground_truth_for_offline_reducer_only"] for row, prediction in pairs),
            "fixed": sum(
                row["incumbent"] != row["ground_truth_for_offline_reducer_only"]
                and prediction == row["ground_truth_for_offline_reducer_only"]
                for row, prediction in pairs
            ),
            "broken": sum(
                row["incumbent"] == row["ground_truth_for_offline_reducer_only"]
                and prediction != row["ground_truth_for_offline_reducer_only"]
                for row, prediction in pairs
            ),
            "changed": sum(prediction != row["incumbent"] for row, prediction in pairs),
        }
        report["arms"][name]["net"] = report["arms"][name]["fixed"] - report["arms"][name]["broken"]
    candidate = report["arms"]["candidate"]
    report["mechanism"] = {
        "overrides": sum(bool(row["decision"]["override"]) for row in valid),
        "aligned_transcript_nonempty": sum(bool(row["transcripts"]["aligned"].strip()) for row in valid),
    }
    report["promotion_gate"] = {
        "all_rows_valid": len(valid) == expected,
        "positive_net_gain": candidate["net"] > 0,
        "at_least_one_override": report["mechanism"]["overrides"] > 0,
    }
    report["promoted"] = all(report["promotion_gate"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18405/v1")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--asr-model", default="Systran/faster-whisper-tiny")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-device-index", type=int, default=0)
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "baselines" / "VideoSEAL"), str(root / "research" / "diagnostics")]
    import run_fixed30_inspector_pilots as helper
    from faster_whisper import WhisperModel

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))["items"]
    baseline = read_baseline(args.baseline)
    tasks = helper.load_tasks(args.parquet)
    args.temp_root = (args.temp_root or root / ".tmp" / "causal_modality_instrument").resolve()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    asr = WhisperModel(
        args.asr_model,
        device=args.asr_device,
        device_index=args.asr_device_index,
        compute_type=args.asr_compute_type,
        download_root=str(root / ".tmp" / "asr_models"),
    )
    asr_lock = threading.Lock()
    keyed = {(str(item["video_id"]), str(item["uid"])): item for item in protocol}
    missing = [key for key in keyed if key not in baseline or key not in tasks]
    if missing:
        raise RuntimeError(f"protocol rows missing from baseline or tasks: {missing}")

    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    run_one,
                    asr=asr,
                    asr_lock=asr_lock,
                    helper=helper,
                    item=item,
                    task=tasks[key],
                    baseline=baseline[key],
                    args=args,
                ): key
                for key, item in keyed.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    value = future.result()
                except Exception as exc:
                    value = {"video_id": key[0], "uid": key[1], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                rows.append(value)
                with WRITE_LOCK:
                    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "processed": len(rows),
                            "total": len(keyed),
                            "uid": key[1],
                            "status": value["status"],
                            "decision": value.get("decision", {}),
                            "error": value.get("error", ""),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report = summarize(rows, len(keyed))
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["coverage"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

