"""Synthetic public identities and scripted rollouts; no media/model calls."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from methods.packet_opsd.evaluate import evaluate_tasks, load_evaluation_tasks


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.video = self.root / "synthetic.mp4"
        self.video.touch()
        self.index = self.root / "index"
        self.index.mkdir()
        self.questions = [f"Toy {i}. A. red B. blue C. green D. gray" for i in range(300)]
        self.mapping = self.root / "mapping.json"
        self.mapping.write_text(json.dumps([
            {"ordinal": i, "qid": str(i), "sample_uid": f"lvbench:{i}",
             "video_id": f"video{i}", "original_video_id": f"oldvideo{i}",
             "question_sha256": hashlib.sha256(self.questions[i].encode()).hexdigest()}
            for i in range(300)
        ]))
        self.manifest = self.root / "tasks.jsonl"

    def task(self, ordinal, *, uid=None, video_id=None):
        return {"uid": uid or str(ordinal), "video_id": video_id or f"video{ordinal}",
                "question": self.questions[ordinal], "video_path": str(self.video),
                "visual_index": str(self.index)}

    def write(self, tasks):
        self.manifest.write_text("".join(json.dumps(task) + "\n" for task in tasks))

    def test_uid_video_aliases_normalize_and_mapping_ordinal_controls_order(self):
        self.write([self.task(2, uid="lvbench:2", video_id="oldvideo2"),
                    self.task(1, uid="runtime-prefix:1"), self.task(0)])
        entries = load_evaluation_tasks(self.manifest, self.mapping, smoke_limit=3)
        self.assertEqual([entry.ordinal for entry in entries], [0, 1, 2])
        self.assertEqual([entry.task["uid"] for entry in entries], ["0", "1", "2"])
        self.assertEqual([entry.task["video_id"] for entry in entries], ["video0", "video1", "video2"])
        self.assertEqual(entries[2].input_uid, "lvbench:2")
        self.assertEqual(entries[2].input_video_id, "oldvideo2")
        self.write([self.task(0), self.task(0, uid="lvbench:0")])
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            load_evaluation_tasks(self.manifest, self.mapping, smoke_limit=2)

    def test_question_hash_unknown_uid_wrong_video_and_gold_fields_are_rejected(self):
        for task, error in (
            ({**self.task(0), "question": "changed question"}, "question hash"),
            (self.task(0, uid="outside:9999"), "outside"),
            (self.task(0, video_id="video1"), "wrong video"),
            ({**self.task(0), "ground_truth": "FORBIDDEN"}, "unexpected fields"),
        ):
            with self.subTest(error=error):
                self.write([task])
                with self.assertRaisesRegex(ValueError, error):
                    load_evaluation_tasks(self.manifest, self.mapping, smoke_limit=1)

    def test_formal_run_requires_all_300_rows(self):
        self.write([self.task(i) for i in range(299)])
        with self.assertRaisesRegex(ValueError, "all 300"):
            load_evaluation_tasks(self.manifest, self.mapping)
        entries = load_evaluation_tasks(self.manifest, self.mapping, smoke_limit=1)
        with self.assertRaisesRegex(ValueError, "complete validated fixed300"):
            evaluate_tasks(entries, object(), self.root / "invalid-formal", smoke=False)
        self.write([self.task(i) for i in reversed(range(300))])
        entries = load_evaluation_tasks(self.manifest, self.mapping)
        self.assertEqual(len(entries), 300)
        self.assertEqual([entry.ordinal for entry in entries], list(range(300)))

    def test_failed_row_keeps_denominator_and_smoke_cannot_feed_fixed300_observer(self):
        self.write([self.task(1, uid="lvbench:1"), self.task(0)])
        entries = load_evaluation_tasks(self.manifest, self.mapping, smoke_limit=2)
        calls = []

        def scripted_run(task, policy, **kwargs):
            calls.append((task["uid"], kwargs))
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["max_new_tokens"], 4096)
            if task["uid"] == "1":
                raise RuntimeError("scripted task failure")
            return SimpleNamespace(
                trajectory={"uid": task["uid"], "video_id": task["video_id"],
                            "run_id": "scripted-run", "steps": [], "groundtruth": None},
                result={"final": "C"},
            )

        output = self.root / "smoke"
        summary = evaluate_tasks(entries, object(), output, smoke=True, run_task_fn=scripted_run)
        rows = [json.loads(line) for line in (output / "SMOKE_RESULTS.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["uid"] for row in rows], ["0", "1"])
        self.assertEqual(rows[0]["prediction"], "C")
        self.assertEqual(rows[1]["state"], "failed")
        self.assertEqual(rows[1]["prediction"], "")
        self.assertEqual(rows[1]["error_type"], "RuntimeError")
        self.assertTrue(Path(rows[0]["trajectory_path"]).is_absolute())
        self.assertTrue(Path(rows[0]["trajectory_path"]).is_file())
        self.assertEqual(summary["result_rows"], 2)
        self.assertEqual(summary["failed_rows"], 1)
        self.assertFalse(summary["fixed300_complete"])
        self.assertFalse(summary["observer_compatible"])
        self.assertFalse(summary["accuracy_evaluated"])
        self.assertFalse((output / "results.jsonl").exists())
        self.assertTrue(all(row["smoke"] and not row["fixed300_complete"] for row in rows))


if __name__ == "__main__":
    unittest.main()
