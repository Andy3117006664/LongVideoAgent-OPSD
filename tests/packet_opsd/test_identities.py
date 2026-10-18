"""Answer-free mapping selection, overlap guards and pre-model CLI binding."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from methods.packet_opsd import api_evaluate, evaluate, train
from methods.packet_opsd.identities import resolve_identity_mapping


class BeforeModelBoundary(RuntimeError):
    pass


class MappingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.video = self.root / "training-video.mp4"
        self.video.touch()
        self.index = self.root / "index"
        self.index.mkdir()
        self.mapping = self.root / "new-mapping.json"
        self.rows = [
            {"ordinal": i, "qid": str(1000 + i), "sample_uid": f"lvbench:{1000+i}",
             "video_id": f"new-test-{i % 32}", "original_video_id": f"physical-test-{i % 32}",
             "question_sha256": hashlib.sha256(f"heldout question {i}".encode()).hexdigest()}
            for i in range(300)
        ]
        self.mapping.write_text(json.dumps(self.rows))
        self.eval_manifest = self.root / "evaluation.jsonl"
        self.eval_manifest.write_text(json.dumps({
            "uid": self.rows[0]["sample_uid"], "video_id": self.rows[0]["original_video_id"],
            "question": "heldout question 0", "video_path": str(self.video),
            "visual_index": str(self.index)}) + "\n")
        self.train_manifest = self.root / "train.jsonl"
        self.task = {"uid": "lvbench:old-test-now-training", "video_id": "training-video",
                     "question": "new independent training question", "video_path": str(self.video),
                     "visual_index": str(self.index)}
        self.write_train(self.task)

    def write_train(self, task):
        self.train_manifest.write_text(json.dumps(task) + "\n")

    def test_default_is_the_same_archived_public_contract(self):
        bound = resolve_identity_mapping()
        expected = Path(train.__file__).resolve().parents[2] / "baselines/videoseal_6000d/selection/identity_mapping.json"
        self.assertEqual(bound.path, expected.resolve())
        self.assertEqual(bound.sha256, hashlib.sha256(expected.read_bytes()).hexdigest())
        self.assertEqual(len(bound.rows), 300)

    def test_resolved_binding_uses_the_same_loaded_bytes_for_guards_and_hash(self):
        alias = self.root / "mapping-link.json"
        alias.symlink_to(self.mapping)
        bound = resolve_identity_mapping(Path(os.path.relpath(alias)))
        original = self.mapping.read_bytes()
        self.assertEqual(bound.path, self.mapping.resolve())
        self.assertEqual(bound.metadata(), {"identity_mapping": str(self.mapping.resolve()),
                                           "identity_mapping_sha256": hashlib.sha256(original).hexdigest()})
        self.mapping.write_text("changed after this snapshot")
        self.assertEqual(len(train.load_tasks(self.train_manifest, bound)), 1)
        entries = evaluate.load_evaluation_tasks(self.eval_manifest, bound, smoke_limit=1)
        self.assertEqual(entries[0].task["uid"], "1000")
        self.assertEqual(bound.sha256, hashlib.sha256(original).hexdigest())

    def test_invalid_or_annotated_mappings_are_rejected(self):
        cases = []
        cases.append(self.rows[:-1])
        for field, value in (("ordinal", True), ("ordinal", 300), ("qid", 1000),
                             ("question_sha256", "bad-hash"), ("ground_truth", "FORBIDDEN"),
                             ("time_reference", [[1, 2]])):
            rows = deepcopy(self.rows)
            rows[0][field] = value
            cases.append(rows)
        rows = deepcopy(self.rows)
        rows[1]["sample_uid"] = rows[0]["qid"]
        cases.append(rows)
        rows = deepcopy(self.rows)
        rows[1]["original_video_id"] = rows[0]["video_id"]
        cases.append(rows)
        for rows in cases:
            with self.subTest(first=rows[0]):
                self.mapping.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    resolve_identity_mapping(self.mapping)
        self.mapping.write_text('{"rows": [], "rows": []}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            resolve_identity_mapping(self.mapping)

    def test_new_mapping_keeps_uid_question_video_and_gold_guards(self):
        self.assertEqual(len(train.load_tasks(self.train_manifest, self.mapping)), 1)
        cases = [
            (dict(self.task, uid="runtime:1000"), False, "UID overlaps"),
            (dict(self.task, question="heldout question 0"), True, "question overlaps"),
            (dict(self.task, video_id="physical-test-0"), False, "video overlaps"),
            (dict(self.task, ground_truth="FORBIDDEN"), True, "unexpected fields"),
        ]
        for task, shared, error in cases:
            self.write_train(task)
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    train.load_tasks(self.train_manifest, self.mapping, allow_shared_videos=shared)
        # The physical target behind a symlink also participates in video checks.
        physical = self.root / "physical-test-0.mp4"
        physical.touch()
        link = self.root / "apparently-fresh-video.mp4"
        link.symlink_to(physical)
        self.write_train(dict(self.task, video_path=str(link)))
        with self.assertRaisesRegex(ValueError, "video overlaps"):
            train.load_tasks(self.train_manifest, self.mapping)

    def cli_args(self, kind, output):
        common = ["--tasks", str(self.train_manifest if kind == "train" else self.eval_manifest),
                  "--output", str(output), "--inspector-endpoint", "http://synthetic-inspector/v1",
                  "--compactor-endpoint", "http://synthetic-compactor/v1"]
        if kind == "api":
            return common + ["--planner-endpoint", "http://synthetic-planner/v1",
                             "--planner-model", "synthetic-base", "--smoke-limit", "1"]
        if kind == "hf":
            return common + ["--model", str(self.root / "model"),
                             "--adapter", str(self.root / "adapter"), "--smoke-limit", "1"]
        return common + ["--model", "synthetic-model"]

    def test_all_cli_mappings_fail_before_output_or_model_on_missing_and_bad_contract(self):
        broken = self.root / "broken.json"
        broken.write_text(json.dumps(self.rows[:-1]))
        for kind, module in (("train", train), ("api", api_evaluate), ("hf", evaluate)):
            for path in (self.root / "missing.json", broken):
                output = self.root / f"invalid-{kind}-{path.stem}"
                with self.subTest(kind=kind, mapping=path.name), \
                        patch.object(train, "configure_baseline", side_effect=AssertionError("too late")):
                    with self.assertRaises((FileNotFoundError, ValueError)):
                        module.main(self.cli_args(kind, output) + ["--identity-mapping", str(path)])
                    self.assertFalse(output.exists())

    def test_api_and_hf_configs_bind_the_explicit_absolute_path_and_loaded_hash(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        model_load = Mock(return_value=object())
        policy_module = ModuleType("methods.packet_opsd.policy")
        policy_module.HFPolicy = SimpleNamespace(load=model_load)
        expected_sha = hashlib.sha256(self.mapping.read_bytes()).hexdigest()
        for kind, module, worker_name in (("api", api_evaluate, "evaluate_api_tasks"),
                                           ("hf", evaluate, "evaluate_tasks")):
            output = self.root / f"valid-{kind}"
            with self.subTest(kind=kind), patch.object(train, "configure_baseline", return_value={}), \
                    patch.object(module, worker_name, return_value={"failed_rows": 0}) as worker, \
                    patch.dict(sys.modules, {"methods.packet_opsd.policy": policy_module}):
                self.assertEqual(module.main(self.cli_args(kind, output) + [
                    "--identity-mapping", os.path.relpath(self.mapping)]), 0)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["identity_mapping"], str(self.mapping.resolve()))
            self.assertEqual(config["identity_mapping_sha256"], expected_sha)
            self.assertEqual(config["selected_ordinals"], [0])
            self.assertEqual(worker.call_args.args[0][0].task["uid"], "1000")
            self.assertEqual(worker.call_args.args[0][0].task["video_id"], "new-test-0")
        self.assertEqual(model_load.call_count, 1)

    def test_training_config_passes_the_bound_mapping_to_checkpoint_config(self):
        policy_module = ModuleType("methods.packet_opsd.policy")
        model_load = Mock(side_effect=BeforeModelBoundary("do not load a model"))
        policy_module.HFPolicy = SimpleNamespace(load=model_load)
        policy_module.ContextOverflow = type("ContextOverflow", (Exception,), {})
        torch_module = ModuleType("torch")
        torch_module.manual_seed = lambda seed: None
        torch_module.cuda = SimpleNamespace(manual_seed_all=lambda seed: None)
        output = self.root / "valid-train"
        with patch.dict(sys.modules, {"torch": torch_module, "methods.packet_opsd.policy": policy_module}), \
                patch.object(train, "configure_baseline", return_value={}):
            with self.assertRaises(BeforeModelBoundary):
                train.main(self.cli_args("train", output) + ["--identity-mapping", os.path.relpath(self.mapping)])
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["identity_mapping"], str(self.mapping.resolve()))
        self.assertEqual(config["identity_mapping_sha256"], hashlib.sha256(self.mapping.read_bytes()).hexdigest())
        self.assertEqual(config["training_split_policy"], "video-disjoint")
        model_load.assert_called_once()

    def test_omitting_the_flag_keeps_the_archived_mapping_location_for_all_clis(self):
        repo = self.root / "synthetic-repo"
        archived = repo / "baselines/videoseal_6000d/selection/identity_mapping.json"
        archived.parent.mkdir(parents=True)
        archived.write_bytes(self.mapping.read_bytes())
        source = repo / "baselines/videoseal_6000d/source"
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        policy_module = ModuleType("methods.packet_opsd.policy")
        policy_module.ContextOverflow = type("ContextOverflow", (Exception,), {})
        policy_module.HFPolicy = SimpleNamespace(load=Mock(return_value=object()))
        for kind, module, worker_name in (("api", api_evaluate, "evaluate_api_tasks"),
                                           ("hf", evaluate, "evaluate_tasks")):
            output = self.root / f"default-{kind}"
            with patch.object(module, "_source_paths", return_value=(repo, source)), \
                    patch.object(train, "configure_baseline", return_value={}), \
                    patch.object(module, worker_name, return_value={"failed_rows": 0}), \
                    patch.dict(sys.modules, {"methods.packet_opsd.policy": policy_module}):
                self.assertEqual(module.main(self.cli_args(kind, output)), 0)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["identity_mapping"], str(archived.resolve()))
            self.assertEqual(config["identity_mapping_sha256"], hashlib.sha256(archived.read_bytes()).hexdigest())
        torch_module = ModuleType("torch")
        torch_module.manual_seed = lambda seed: None
        torch_module.cuda = SimpleNamespace(manual_seed_all=lambda seed: None)
        policy_module.HFPolicy = SimpleNamespace(load=Mock(side_effect=BeforeModelBoundary()))
        output = self.root / "default-train"
        with patch.object(train, "__file__", str(repo / "methods/packet_opsd/train.py")), \
                patch.object(train, "configure_baseline", return_value={}), \
                patch.dict(sys.modules, {"torch": torch_module, "methods.packet_opsd.policy": policy_module}):
            with self.assertRaises(BeforeModelBoundary):
                train.main(self.cli_args("train", output))
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["identity_mapping"], str(archived.resolve()))
        self.assertEqual(config["identity_mapping_sha256"], hashlib.sha256(archived.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
