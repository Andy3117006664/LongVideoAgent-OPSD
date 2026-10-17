#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any
from urllib import request


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "VideoSEAL"), str(ROOT / "research" / "diagnostics")]

from methods.CAUSAL_EVIDENCE_ORBITS.frames import extract_packet  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.io import read_baseline  # noqa: E402
from methods.CAUSAL_EVIDENCE_ORBITS.schema import CandidateSpan  # noqa: E402
from methods.TOPOLOGICAL_EVIDENCE_COMPILER import choose_representation  # noqa: E402


WRITE_LOCK = threading.Lock()


class TransformersObserver:
    """Qwen3-VL native Transformers observer used when vLLM lacks an adapter."""

    def __init__(self, model_path: str, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
        model_cls = {
            "qwen3_vl": Qwen3VLForConditionalGeneration,
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        }.get(model_type)
        if model_cls is None:
            raise ValueError(f"unsupported Transformers observer model_type={model_type!r}")
        self.model_type = model_type
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = torch.device(device)

    def ask(
        self,
        *,
        question: str,
        frames: list[Path],
        timestamps: list[float],
        spans: list[list[float]],
        max_new_tokens: int,
        representation: str,
        fps: float,
    ) -> tuple[str, str]:
        content: list[dict[str, Any]] = []
        video_metadata: list[dict[str, Any]] = []
        opened_images: list[Any] = []
        if representation == "span_videos":
            from PIL import Image

            grouped: list[list[tuple[Path, float]]] = [[] for _ in spans]
            for frame, timestamp in zip(frames, timestamps):
                candidates = [
                    index
                    for index, (start, end) in enumerate(spans)
                    if start - 1e-3 <= timestamp <= end + 1e-3
                ]
                if not candidates:
                    candidates = [
                        min(range(len(spans)), key=lambda index: abs(timestamp - (spans[index][0] + spans[index][1]) / 2.0))
                    ]
                grouped[candidates[0]].append((frame, timestamp))
            for span_index, ((start, end), group) in enumerate(zip(spans, grouped), 1):
                if not group:
                    continue
                clip = []
                for frame, _ in group:
                    image = Image.open(frame).convert("RGB")
                    opened_images.append(image)
                    clip.append(image)
                # Qwen3-VL uses temporal patches of two frames. Repeating a
                # singleton is padding only; it introduces no new evidence.
                if len(clip) == 1:
                    clip.append(clip[0])
                content.append({
                    "type": "text",
                    "text": f"[evidence clip {span_index}: source time {start:.3f}s to {end:.3f}s]",
                })
                content.append({"type": "video", "video": clip})
                video_metadata.append({
                    "total_num_frames": len(clip),
                    "fps": fps,
                    "duration": len(clip) / fps,
                    "frames_indices": list(range(len(clip))),
                    "height": clip[0].height,
                    "width": clip[0].width,
                })
        else:
            for frame, timestamp in zip(frames, timestamps):
                content.append({"type": "text", "text": f"[video t={timestamp:.3f}s]"})
                content.append({"type": "image", "image": str(frame)})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        messages[0]["content"].append({
            "type": "text",
            "text": (
                "Answer the multiple-choice question using only the supplied timestamped video frames. "
                "Return exactly one token: A, B, C, or D.\n\n"
                f"Question:\n{question}"
            ),
        })
        processor_kwargs: dict[str, Any] = {}
        if video_metadata:
            processor_kwargs["video_metadata"] = [video_metadata]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **processor_kwargs,
            )
        finally:
            for image in opened_images:
                image.close()
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip()
        prediction = raw.upper()[:1]
        return (prediction if prediction in set("ABCD") else ""), raw


def image_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def ask(
    args: argparse.Namespace,
    *,
    question: str,
    frames: list[Path],
    timestamps: list[float],
) -> tuple[str, str]:
    prompt = (
        "Answer the multiple-choice question using only the supplied timestamped video frames. "
        "Return exactly one token: A, B, C, or D.\n\n"
        f"Question:\n{question}"
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
    prediction = raw.upper()[:1]
    return (prediction if prediction in set("ABCD") else ""), raw


def run_one(
    *,
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
    candidates = tuple(
        CandidateSpan(f"{a:g}_{b:g}", float(a), float(b), rank)
        for rank, (a, b) in enumerate(item["spans"], 1)
    )
    with tempfile.TemporaryDirectory(prefix=f"observer-{uid}-", dir=args.temp_root) as raw:
        frames, timestamps, edge = extract_packet(
            helper,
            video,
            candidates,
            Path(raw) / "frames",
            fps=args.fps,
            cap=args.frames,
        )
        effective_representation = (
            choose_representation(str(baseline["question"]))
            if args.representation == "topology"
            else args.representation
        )
        if args.backend == "transformers":
            observer = args.observer
            prediction, model_raw = observer.ask(
                question=str(baseline["question"]),
                frames=frames,
                timestamps=timestamps,
                spans=[[float(a), float(b)] for a, b in item["spans"]],
                max_new_tokens=args.max_tokens,
                representation=effective_representation,
                fps=args.fps,
            )
        else:
            prediction, model_raw = ask(
                args,
                question=str(baseline["question"]),
                frames=frames,
                timestamps=timestamps,
            )
    return {
        "schema_version": "observer-capacity-audit-v1",
        "video_id": str(item["video_id"]),
        "uid": uid,
        "question": str(baseline["question"]),
        "incumbent": str(baseline["prediction"]).upper(),
        "prediction": prediction,
        "raw": model_raw,
        "spans": [[float(a), float(b)] for a, b in item["spans"]],
        "frame_count": len(frames),
        "representation": effective_representation,
        "representation_policy": args.representation,
        "dynamic_edge": edge,
        "ground_truth_for_offline_reducer_only": str(baseline.get("ground_truth") or "").upper(),
        "status": "ok" if prediction else "parse_error",
        "information_boundary": {
            "gold_online": False,
            "baseline_correctness_online": False,
            "time_reference_online": False,
            "audio_online": False,
        },
    }


def summarize(rows: list[dict[str, Any]], expected: int, model: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    correct = sum(row["prediction"] == row["ground_truth_for_offline_reducer_only"] for row in valid)
    incumbent_correct = sum(row["incumbent"] == row["ground_truth_for_offline_reducer_only"] for row in valid)
    fixed = sum(
        row["incumbent"] != row["ground_truth_for_offline_reducer_only"]
        and row["prediction"] == row["ground_truth_for_offline_reducer_only"]
        for row in valid
    )
    broken = sum(
        row["incumbent"] == row["ground_truth_for_offline_reducer_only"]
        and row["prediction"] != row["ground_truth_for_offline_reducer_only"]
        for row in valid
    )
    return {
        "schema_version": "observer-capacity-summary-v1",
        "model": model,
        "coverage": {"expected": expected, "valid": len(valid), "failed": expected - len(valid)},
        "incumbent_correct": incumbent_correct,
        "observer_correct": correct,
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "changed": sum(row["prediction"] != row["incumbent"] for row in valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--backend", choices=("api", "transformers"), default="api")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--representation", choices=("images", "span_videos", "topology"), default="images")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "baselines" / "VideoSEAL"), str(root / "research" / "diagnostics")]
    import run_fixed30_inspector_pilots as helper

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))["items"]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    protocol = [item for index, item in enumerate(protocol) if index % args.num_shards == args.shard_index]
    if args.limit > 0:
        protocol = protocol[: args.limit]
    baseline = read_baseline(args.baseline)
    tasks = helper.load_tasks(args.parquet)
    args.temp_root = (args.temp_root or root / ".tmp" / "observer_capacity").resolve()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.observer = None
    if args.backend == "transformers":
        if args.model_path is None:
            raise ValueError("--model-path is required with --backend transformers")
        args.observer = TransformersObserver(str(args.model_path), device=args.device)
    keyed = {(str(item["video_id"]), str(item["uid"])): item for item in protocol}
    missing = [key for key in keyed if key not in baseline or key not in tasks]
    if missing:
        raise RuntimeError(f"protocol rows missing from baseline or tasks: {missing}")

    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        # Transformers generation is stateful and memory-heavy; serialize it
        # unless an explicitly thread-safe observer backend is supplied.
        workers = 1 if args.backend == "transformers" else max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_one,
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
                            "prediction": value.get("prediction", ""),
                            "error": value.get("error", ""),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report = summarize(rows, len(keyed), args.model)
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["coverage"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
