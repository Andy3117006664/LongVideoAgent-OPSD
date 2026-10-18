"""Scripted CPU checks for native API evaluation and parent-side reduction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from methods.packet_opsd.api_evaluate import (
    _empty_row, _failure, _publish_results, _validate_worker_row,
    evaluate_api_tasks, native_api_rollout,
)
from methods.packet_opsd.evaluate import EvaluationTask


def entry(ordinal, uid=None):
    uid = uid or f"task-{ordinal}"
    return EvaluationTask({"uid": uid, "video_id": f"video-{ordinal}",
                           "question": f"Question {ordinal}", "video_path": "/unused/video.mp4",
                           "visual_index": "/unused/index"}, ordinal, uid, f"video-{ordinal}")


def scripted_worker(item, output, settings, result_queue):
    if item.task["uid"] == "timeout":
        time.sleep(10)
        return
    if item.task["uid"] == "crash":
        os._exit(7)
    row = dict(_empty_row(item), state="completed", prediction="A")
    if item.task["uid"] == "mismatch":
        row["question_sha256"] = "wrong"
    result_queue.put({"ordinal": item.ordinal, "row": row})


class NativeApiTests(unittest.TestCase):
    def test_native_agent_retains_original_client_and_environment(self):
        from methods.packet_opsd import rollout
        seen = {}
        original_client = object()

        class Agent:
            def __init__(self, *, tools, llm_backend):
                self.llm = original_client
                seen.update(tools=set(tools), backend=llm_backend)

            def run(self, **kwargs):
                seen.update(kwargs)
                seen["client_unchanged"] = self.llm is original_client
                seen["environment"] = {key: os.environ.get(key) for key in
                    ("AGENT_LLM_API_BASE", "AGENT_LLM_MODEL", "AGENT_LLM_MAX_TOKENS",
                     "AGENT_LLM_TEMPERATURE", "AGENT_API_USE_MESSAGES", "AGENT_PLANNER_NO_THINK")}
                self.traj = {"uid": kwargs["uid"], "video_id": kwargs["video_id"], "steps": []}
                return {"final": "A"}

        with patch.dict(os.environ, {"AGENT_LLM_API_BASE": "old", "AGENT_LLM_MODEL": "old-model"}), \
                patch.object(rollout, "_load_videoseal", return_value=(Agent, lambda: {
                    "visual_retrieve": object(), "visual_inspect": object(), "extra": object()})):
            result = native_api_rollout(entry(0).task, save_dir=Path("/unused/output"),
                                        planner_endpoint="http://same-engine/v1", planner_model="trained-lora")
            self.assertEqual(os.environ["AGENT_LLM_API_BASE"], "old")
            self.assertEqual(os.environ["AGENT_LLM_MODEL"], "old-model")
        self.assertTrue(seen["client_unchanged"])
        self.assertEqual(seen["tools"], {"visual_retrieve", "visual_inspect"})
        self.assertIsNone(seen["groundtruth"])
        self.assertEqual(seen["max_steps"], 16)
        self.assertEqual(seen["environment"], {
            "AGENT_LLM_API_BASE": "http://same-engine/v1", "AGENT_LLM_MODEL": "trained-lora",
            "AGENT_LLM_MAX_TOKENS": "4096", "AGENT_LLM_TEMPERATURE": "0.0",
            "AGENT_API_USE_MESSAGES": "0", "AGENT_PLANNER_NO_THINK": "0"})
        self.assertEqual(result.factual_records, [])
        self.assertEqual(result.result, {"final": "A"})

    def test_parent_rejects_worker_identity_and_state_changes(self):
        item = entry(1)
        valid = dict(_empty_row(item), state="completed", prediction="A")
        self.assertEqual(_validate_worker_row(item, valid)["state"], "completed")
        for key in ("ordinal", "uid", "video_id", "question_sha256"):
            with self.subTest(key=key):
                result = _validate_worker_row(item, dict(valid, **{key: "wrong"}))
                self.assertEqual(result["error_type"], "WorkerIdentityMismatch")
                self.assertEqual(result["ordinal"], 1)
                self.assertEqual(result["prediction"], "")
        self.assertEqual(_validate_worker_row(item, dict(valid, state="skipped"))["state"], "failed")

    def test_complete_reduction_keeps_all_300_including_failures(self):
        entries = [entry(i) for i in range(300)]
        rows = {item.ordinal: _failure(item, "TaskTimeout", "scripted") for item in entries}
        rows[2].update(state="completed", prediction="B")
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            summary = _publish_results(entries, rows, output, smoke=False)
            values = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual(summary["result_rows"], 300)
            self.assertEqual(summary["failed_rows"], 299)
            self.assertTrue(summary["observer_compatible"])
            self.assertEqual([row["ordinal"] for row in values], list(range(300)))
            self.assertEqual(values[2]["question_sha256"], hashlib.sha256(b"Question 2").hexdigest())
            self.assertFalse(summary["accuracy_evaluated"])
            with self.assertRaises(FileExistsError):
                _publish_results(entries, rows, output, smoke=False)

    def test_incomplete_or_duplicate_reduction_never_publishes_results(self):
        entries = [entry(i) for i in range(300)]
        rows = {item.ordinal: _failure(item, "scripted", "") for item in entries}
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            with self.assertRaises(ValueError):
                _publish_results(entries, {0: rows[0]}, output, smoke=False)
            with self.assertRaises(ValueError):
                _publish_results(entries[:1], {0: rows[0]}, output, smoke=False)
            self.assertFalse((output / "results.jsonl").exists())

    def test_spawned_workers_timeout_and_identity_failure_keep_smoke_rows(self):
        entries = [entry(0), entry(1, "timeout"), entry(2, "mismatch"), entry(3, "crash")]
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            summary = evaluate_api_tasks(entries, output, settings={}, workers=2,
                                         task_timeout=1.5, smoke=True, worker_target=scripted_worker)
            values = [json.loads(line) for line in (output / "SMOKE_RESULTS.jsonl").read_text().splitlines()]
            self.assertEqual(summary["result_rows"], 4)
            self.assertEqual(summary["failed_rows"], 3)
            self.assertFalse(summary["observer_compatible"])
            self.assertFalse((output / "results.jsonl").exists())
            self.assertEqual(values[1]["error_type"], "TaskTimeout")
            self.assertEqual(values[2]["error_type"], "WorkerIdentityMismatch")
            self.assertEqual(values[3]["error_type"], "WorkerExited")
            self.assertEqual(len((output / "PARTIAL_RESULTS.jsonl").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
