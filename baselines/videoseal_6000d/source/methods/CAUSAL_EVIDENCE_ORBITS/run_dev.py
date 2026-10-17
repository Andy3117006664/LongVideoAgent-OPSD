#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "VideoSEAL"), str(ROOT / "research" / "diagnostics")]

from methods.CAUSAL_EVIDENCE_ORBITS.codebooks import CODEBOOKS  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.frames import extract_packet, make_pixel_nulls  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.io import (  # noqa: E402
    first_retrieval_candidates,
    read_baseline,
    read_protocol,
)
from methods.CAUSAL_EVIDENCE_ORBITS.model import VisualMeasurementClient  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.question import parse_question  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.router import (  # noqa: E402
    causal_policy,
    decide_arm,
    measurement,
    reduce_probe,
    semantic_policy,
    sham_policy,
)
from methods.CAUSAL_EVIDENCE_ORBITS.schema import (  # noqa: E402
    CandidateSpan,
    CodebookMeasurement,
    SpanProbe,
)


WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_orbit(
    client: VisualMeasurementClient,
    *,
    stem: str,
    options: dict[str, str],
    frames: list[Path],
    nulls: list[Path],
    timestamps: list[float],
) -> tuple[CodebookMeasurement, ...]:
    output = []
    for codebook_id, token_to_semantic in CODEBOOKS:
        real = client.measure(
            stem=stem,
            options=options,
            token_to_semantic=token_to_semantic,
            frames=frames,
            timestamps=timestamps,
        )
        pixel_null = client.measure(
            stem=stem,
            options=options,
            token_to_semantic=token_to_semantic,
            frames=nulls,
            timestamps=timestamps,
        )
        output.append(measurement(codebook_id, token_to_semantic, real, pixel_null))
    return tuple(output)


def run_one(
    *,
    helper: Any,
    task: dict[str, Any],
    baseline: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    key = (str(baseline["video_id"]), str(baseline["uid"]))
    question = str(baseline["question"])
    stem, options = parse_question(question)
    incumbent = str(baseline["prediction"]).upper()
    trajectory = Path(str(baseline["trajectory_path"]))
    candidates = first_retrieval_candidates(trajectory, limit=args.candidates)
    if len(candidates) != args.candidates:
        raise RuntimeError(f"candidate pool has {len(candidates)}/{args.candidates} spans")
    video = helper.resolve_video(task)
    if video is None:
        raise FileNotFoundError(f"video unavailable for {key}")
    client = VisualMeasurementClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    probes: list[SpanProbe] = []
    replays: dict[str, tuple[CodebookMeasurement, ...]] = {}
    edges: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix=f"ceo-{key[1]}-", dir=args.temp_root) as raw:
        directory = Path(raw)
        for span in candidates:
            frames, timestamps, edge = extract_packet(
                helper,
                video,
                (span,),
                directory / "probes" / span.doc_id / "real",
                fps=args.probe_fps,
                cap=args.probe_frames,
            )
            nulls = make_pixel_nulls(frames, directory / "probes" / span.doc_id / "null")
            values = run_orbit(
                client,
                stem=stem,
                options=options,
                frames=frames,
                nulls=nulls,
                timestamps=timestamps,
            )
            probes.append(reduce_probe(span, values))
            edges[f"probe:{span.doc_id}"] = edge
        frozen_probes = tuple(probes)
        policies = {
            "causal_orbit": causal_policy(frozen_probes, args.selected),
            "semantic_control": semantic_policy(candidates, args.selected),
            "hash_sham": sham_policy(candidates, key[1], args.selected),
        }
        for arm, selected in policies.items():
            if not selected:
                replays[arm] = ()
                continue
            frames, timestamps, edge = extract_packet(
                helper,
                video,
                selected,
                directory / "replays" / arm / "real",
                fps=args.replay_fps,
                cap=args.replay_frames,
            )
            nulls = make_pixel_nulls(frames, directory / "replays" / arm / "null")
            replays[arm] = run_orbit(
                client,
                stem=stem,
                options=options,
                frames=frames,
                nulls=nulls,
                timestamps=timestamps,
            )
            edges[f"replay:{arm}"] = edge
    probe_by_doc = {item.span.doc_id: item for item in probes}
    arms = {
        arm: decide_arm(
            arm=arm,
            selected=selected,
            probe_by_doc=probe_by_doc,
            replay=replays[arm],
            incumbent=incumbent,
        )
        for arm, selected in policies.items()
    }
    return {
        "schema_version": "causal-evidence-orbits-v1",
        "video_id": key[0],
        "uid": key[1],
        "question": question,
        "incumbent_prediction": incumbent,
        "candidate_pool": [item.__dict__ for item in candidates],
        "probes": [item.to_dict() for item in probes],
        "arms": {name: value.to_dict() for name, value in arms.items()},
        "dynamic_edges": edges,
        "ground_truth_for_offline_reducer_only": str(baseline.get("ground_truth") or "").upper(),
        "status": "ok",
        "information_boundary": {
            "gold_online": False,
            "time_reference_online": False,
            "baseline_correctness_online": False,
            "incumbent_answer_used_only_for_conservative_fallback": True,
            "candidate_source": "first_fixed30_retrieval_top8",
        },
    }


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
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--selected", type=int, default=3)
    parser.add_argument("--probe-frames", type=int, default=4)
    parser.add_argument("--probe-fps", type=float, default=0.25)
    parser.add_argument("--replay-frames", type=int, default=64)
    parser.add_argument("--replay-fps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "baselines" / "VideoSEAL"), str(root / "research" / "diagnostics")]
    import run_fixed30_inspector_pilots as helper

    args.temp_root = (args.temp_root or root / ".tmp" / "causal_evidence_orbits").resolve()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    keys = read_protocol(args.protocol)
    baselines = read_baseline(args.baseline)
    tasks = helper.load_tasks(args.parquet)
    missing = [key for key in keys if key not in baselines or key not in tasks]
    if missing:
        raise RuntimeError(f"protocol rows missing from baseline or parquet: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run_one, helper=helper, task=tasks[key], baseline=baselines[key], args=args): key
                for key in keys
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    value = future.result()
                    rows.append(value)
                except Exception as exc:
                    value = {
                        "video_id": key[0],
                        "uid": key[1],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(value)
                with WRITE_LOCK:
                    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "processed": len(rows) + len(failures),
                            "total": len(keys),
                            "uid": key[1],
                            "status": value["status"],
                            "error": value.get("error", ""),
                            "updated_at": utc_now(),
                        }
                    ),
                    flush=True,
                )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

