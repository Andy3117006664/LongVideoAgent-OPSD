"""Exercise the real captured agent loop with scripted text and fake tools."""

from dataclasses import dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from methods.packet_opsd import rollout
from methods.packet_opsd.packets import MediaSpec, SamplerSpec, build_action_packet
from methods.packet_opsd.train import factual_record_skip_reason


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    prompt_ids: list[int]
    complete: bool = True


class ScriptPolicy:
    def __init__(self, texts):
        self.completions = [Completion(text, [i + 10], [1, 2]) for i, text in enumerate(texts)]
        self.calls = []

    def sample(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.completions[len(self.calls) - 1]


def _original_agent():
    root = Path(__file__).resolve().parents[2]
    baseline = root / "baselines" / "videoseal_6000d" / "source" / "baselines" / "VideoSEAL"
    sys.path.insert(0, str(baseline))
    # OpenCV is only used for actual media probing. Tests provide duration
    # metadata and fake tools, so importing the real loop needs no media stack.
    missing_cv2 = "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None
    stubs = {"cv2": types.ModuleType("cv2")} if missing_cv2 else {}
    if missing_cv2:
        # The package initializer eagerly imports the real frame decoder;
        # expose its package path so the actual io.py/time.py can load alone.
        video_package = types.ModuleType("videoseal.utils.video")
        video_package.__path__ = [str(baseline / "videoseal" / "utils" / "video")]
        stubs["videoseal.utils.video"] = video_package
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("videoseal.agents.tool_agent").SimpleToolAgent


class RolloutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_class = _original_agent()
        base = importlib.import_module("videoseal.tools.base")
        cls.Tool, cls.ToolOutput = base.Tool, base.ToolOutput

    def setUp(self):
        self.tool_calls = []
        self.inspection_outputs = []
        calls = self.tool_calls
        inspection_outputs = self.inspection_outputs
        Tool, ToolOutput = self.Tool, self.ToolOutput

        class Retrieve(Tool):
            def __init__(self):
                super().__init__("visual_retrieve")

            def forward(self, *, query, visual_index=None, **kwargs):
                calls.append((self.name, {"query": query}))
                return ToolOutput(self.name, output="NATURAL_RETRIEVAL_TEXT")

            @property
            def json(self):
                return {"type": "function", "function": {"name": self.name, "parameters": {
                    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

        class Inspect(Tool):
            def __init__(self):
                super().__init__("visual_inspect")

            def forward(self, *, video_path, spans, questions, context=None, prompt_type=0):
                calls.append((self.name, {"spans": spans, "questions": questions, "context": context}))
                if inspection_outputs:
                    return ToolOutput(self.name, **inspection_outputs.pop(0))
                return ToolOutput(self.name, output={"answer": "C"}, metadata={
                    "timestamps": [1.0, 1.5],
                    "exclusive_inspector_output": {"control_state": "decisive"},
                })

            @property
            def json(self):
                return {"type": "function", "function": {"name": self.name, "parameters": {
                    "type": "object", "properties": {"spans": {"type": "array"}, "context": {"type": "string"}},
                    "required": ["spans"]}}}

        self.build_map = lambda: {"visual_retrieve": Retrieve, "visual_inspect": Inspect}
        self.task = {
            "question": "Choose the event. A. one B. two C. three D. four",
            "video_id": "v1", "uid": "u1", "video_path": "/unused/test.mp4",
            "visual_index": "/unused/index", "video_duration_sec": 12,
            "ground_truth": "DO_NOT_PASS_GOLD", "gt": "DO_NOT_PASS_GOLD",
            "time_reference": "DO_NOT_INJECT_TIME_REFERENCE",
        }
        self.retrieve = '<tool_call>{"name":"visual_retrieve","arguments":{"query":"event"}}</tool_call>'
        self.inspect = ('<tool_call>{"name":"visual_inspect","arguments":{"spans":'
                        '[{"start_time":"00:00:01","end_time":"00:00:03"}],"context":"look closely"}}</tool_call>')
        self.media = MediaSpec("v1", 12, 2, 320, 240, frame_count=24)
        self.sampler = SamplerSpec()

    def _inspection(self, *, decisive=False, error=None, parity=True, pixels=True):
        packet = build_action_packet(self.media, [{"start_time": "00:00:01", "end_time": "00:00:03"}], self.sampler)
        return {"output": {"answer": "C"} if decisive else "SEARCH_MORE",
                "error": error, "metadata": {
                    "timestamps": list(packet.timestamps) if parity else [-1.0],
                    "actual_pixels": sum(f.final_width * f.final_height for f in packet.frames) if pixels else -1,
                    "exclusive_inspector_output": {"control_state": "decisive" if decisive else "search_more"},
                }}

    def _eligible(self, record):
        return factual_record_skip_reason(record, self.media, self.sampler) is None

    def _run_saved(self, policy, directory, **kwargs):
        with patch.object(rollout, "_load_videoseal", return_value=(self.agent_class, self.build_map)):
            return rollout.run_task(self.task, policy, directory, **kwargs)

    def _run(self, policy, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            return self._run_saved(policy, directory, **kwargs)

    def test_original_prompt_and_terminal_inspection_are_recorded(self):
        policy = ScriptPolicy([self.retrieve, self.inspect])
        output = self._run(policy)
        self.assertEqual(len(policy.calls), 2)  # auto commit needs no Planner answer turn
        self.assertEqual(output.result["final"], "C")
        self.assertEqual(len(output.factual_records), 1)
        record = output.factual_records[0]
        self.assertEqual(record["step_index"], 1)
        self.assertIs(record["completion"], policy.completions[1])
        self.assertEqual(record["messages"], policy.calls[1][0])
        self.assertEqual(record["messages"][0]["content"], output.trajectory["system_prompt"])
        self.assertIn("<tools>", record["messages"][1]["content"])
        self.assertIn("NATURAL_RETRIEVAL_TEXT", record["messages"][1]["content"])
        self.assertIn(self.task["question"], record["messages"][1]["content"])
        self.assertNotIn("DO_NOT_", str(record["messages"]))
        self.assertEqual(record["observation"]["metadata"]["timestamps"], [1.0, 1.5])
        self.assertEqual(record["action"]["arguments"]["context"], "look closely")
        self.assertIsNone(output.trajectory["groundtruth"])
        self.assertEqual(self.tool_calls[1][1]["questions"], self.task["question"])
        self.assertEqual(policy.calls[0][1], {
            "adapter": "student", "prefix": "", "max_new_tokens": 4096, "temperature": 0.7,
        })

    def test_forced_full_video_fallback_has_no_training_record(self):
        policy = ScriptPolicy([self.retrieve])
        output = self._run(policy, max_steps=1)
        self.assertEqual(len(policy.calls), 1)
        self.assertEqual(output.factual_records, [])
        self.assertTrue(output.trajectory["steps"][-1]["observation"]["forced"])
        self.assertEqual([name for name, _ in self.tool_calls], ["visual_retrieve", "visual_inspect"])

    def test_invalid_turn_keeps_exact_feedback_and_step_alignment(self):
        policy = ScriptPolicy(["invalid plain text", self.inspect])
        policy.completions[1].complete = False
        output = self._run(policy)
        record = output.factual_records[0]
        self.assertEqual(record["step_index"], 1)
        self.assertIn("<format_feedback>", record["messages"][1]["content"])
        self.assertIs(record["completion"], policy.completions[1])
        self.assertFalse(record["completion"].complete)

    def test_policy_context_error_propagates_and_environment_is_restored(self):
        class Overflow:
            def sample(self, messages, **kwargs):
                raise ValueError("context overflow")

        with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "9", "SUBTITLE_PATH": "previous"}):
            with self.assertRaisesRegex(ValueError, "context overflow"):
                self._run(Overflow())
            self.assertEqual(os.environ["AGENT_LLM_TEMPERATURE"], "9")
            self.assertEqual(os.environ["SUBTITLE_PATH"], "previous")
        self.assertEqual(self.tool_calls, [])

    def test_training_prefix_preserves_first_factual_record_without_future_calls(self):
        texts = [self.retrieve, self.inspect, self.retrieve, self.inspect]
        self.inspection_outputs[:] = [self._inspection(), self._inspection(decisive=True)]
        full_policy = ScriptPolicy(texts)
        full = self._run(full_policy)
        self.assertEqual(len(full_policy.calls), 4)  # default / evaluation still completes
        self.assertEqual(full.result["final"], "C")
        self.assertNotIn("training_prefix_stop", full.trajectory)
        self.tool_calls.clear()
        self.inspection_outputs[:] = [self._inspection(), self._inspection(decisive=True)]
        prefix_policy = ScriptPolicy(texts)
        prefix = self._run(prefix_policy, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(prefix.factual_records[0], full.factual_records[0])
        self.assertIs(prefix.factual_records[0]["completion"], prefix_policy.completions[1])
        self.assertEqual(len(prefix_policy.calls), 2)
        self.assertEqual([name for name, _ in self.tool_calls], ["visual_retrieve", "visual_inspect"])
        self.assertEqual(len(prefix.trajectory["steps"]), 2)
        self.assertTrue(prefix.result["training_prefix_stop"]["partial"])
        self.assertNotIn("training prefix", str(prefix.factual_records))

    def test_training_prefix_does_not_intercept_decisive_inspector(self):
        policy = ScriptPolicy([self.retrieve, self.inspect])
        self.inspection_outputs[:] = [self._inspection(decisive=True)]
        output = self._run(policy, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(output.result["final"], "C")
        self.assertEqual(output.trajectory["note"], "auto-committed decisive visual_inspect verdict")
        self.assertNotIn("training_prefix_stop", output.result)
        self.assertNotIn("training_prefix_stop", output.trajectory)

    def test_prefix_filter_skips_incomplete_failed_and_nonconformant_inspections(self):
        # Include non-tool, unknown and rejected turns to exercise the same
        # Completion/trajectory alignment as an unmodified complete rollout.
        original_class = self.agent_class

        class RejectOne(original_class):
            def _validate_tool_action(self, tool_name, args):
                return "scripted rejection" if args.get("context") == "reject" else None

        self.agent_class = RejectOne
        unknown = '<tool_call>{"name":"unknown_tool","arguments":{}}</tool_call>'
        reject = self.inspect.replace("look closely", "reject")
        policy = ScriptPolicy(["invalid plain text", unknown, reject] + [self.inspect] * 5 + [self.retrieve])
        policy.completions[3].complete = False
        self.inspection_outputs[:] = [self._inspection(), self._inspection(error="failed inspect"),
                                     self._inspection(parity=False), self._inspection(pixels=False),
                                     self._inspection()]
        output = self._run(policy, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(len(policy.calls), 8)
        self.assertEqual(len(self.tool_calls), 5)
        self.assertEqual([record["step_index"] for record in output.factual_records], [3, 4, 5, 6, 7])
        self.assertEqual([factual_record_skip_reason(record, self.media, self.sampler)
                          for record in output.factual_records],
                         ["ineligible_factual_inspection", "ineligible_factual_inspection",
                          "packet_parity", "packet_parity", None])
        self.assertEqual(output.result["training_prefix_stop"]["eligible_records"], 1)
        self.assertIn("<format_feedback>", output.factual_records[-1]["messages"][1]["content"])

    def test_training_prefix_count_and_saved_partial_marker_are_explicit(self):
        policy = ScriptPolicy([self.inspect, self.inspect, self.retrieve])
        self.inspection_outputs[:] = [self._inspection(), self._inspection()]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "9", "SUBTITLE_PATH": "previous"}):
                output = self._run_saved(policy, directory, max_factual_records=2, record_filter=self._eligible)
                self.assertEqual(os.environ["AGENT_LLM_TEMPERATURE"], "9")
                self.assertEqual(os.environ["SUBTITLE_PATH"], "previous")
            saved = json.loads((Path(directory) / output.trajectory["run_id"] / "trajectory.json").read_text())
        self.assertEqual(saved, output.trajectory)
        self.assertIsNone(saved["answer"])
        self.assertIsNone(output.result["answer"])
        self.assertEqual(saved["training_prefix_stop"], {"partial": True, "eligible_records": 2,
                         "max_factual_records": 2,
                         "reason": "training prefix ready before next Planner generation"})
        self.assertIn("QA incomplete", saved["note"])
        self.assertTrue(saved["finished_at"])
        self.assertIsNotNone(saved["elapsed_sec"])
        self.assertEqual(len(saved["steps"]), 2)
        self.assertEqual(len(policy.calls), 2)

    def test_forced_inspection_never_counts_as_a_training_prefix(self):
        policy = ScriptPolicy([self.retrieve])
        output = self._run(policy, max_steps=1, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(len(policy.calls), 1)
        self.assertEqual(output.factual_records, [])
        self.assertTrue(output.trajectory["steps"][-1]["observation"]["forced"])
        self.assertNotIn("training_prefix_stop", output.result)

    def test_final_step_prefix_preserves_original_forced_fallback_boundary(self):
        policy = ScriptPolicy([self.retrieve] * 15 + [self.inspect])
        self.inspection_outputs[:] = [self._inspection(), self._inspection(decisive=True)]
        output = self._run(policy, max_steps=16, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(len(policy.calls), 16)
        self.assertEqual(len(output.trajectory["steps"]), 17)
        self.assertTrue(output.trajectory["steps"][-1]["observation"]["forced"])
        self.assertEqual(len(output.factual_records), 1)
        self.assertEqual(output.factual_records[0]["step_index"], 15)
        self.assertEqual(output.result["final"], "C")
        self.assertNotIn("training_prefix_stop", output.result)

    def test_prefix_accepts_record_that_old_future_failure_would_discard(self):
        class FutureFailure(ScriptPolicy):
            def sample(self, messages, **kwargs):
                if self.calls:
                    raise ValueError("future context overflow")
                return super().sample(messages, **kwargs)

        self.inspection_outputs[:] = [self._inspection()]
        with self.assertRaisesRegex(ValueError, "future context overflow"):
            self._run(FutureFailure([self.inspect]))
        self.inspection_outputs[:] = [self._inspection()]
        policy = FutureFailure([self.inspect])
        output = self._run(policy, max_factual_records=1, record_filter=self._eligible)
        self.assertEqual(len(output.factual_records), 1)
        self.assertEqual(len(policy.calls), 1)
        self.assertTrue(output.result["training_prefix_stop"]["partial"])

    def test_prefix_does_not_swallow_cancellation_or_filter_errors(self):
        for exception in (KeyboardInterrupt, RuntimeError):
            with self.subTest(exception=exception.__name__):
                class FailingPolicy:
                    def sample(self, messages, **kwargs):
                        raise exception("unrelated failure")

                with patch.dict(os.environ, {"AGENT_LLM_TEMPERATURE": "9"}):
                    with self.assertRaisesRegex(exception, "unrelated failure"):
                        self._run(FailingPolicy(), max_factual_records=1, record_filter=self._eligible)
                    self.assertEqual(os.environ["AGENT_LLM_TEMPERATURE"], "9")
        self.inspection_outputs[:] = [self._inspection()]

        def bad_filter(record):
            raise RuntimeError("filter failed")

        with self.assertRaisesRegex(RuntimeError, "filter failed"):
            self._run(ScriptPolicy([self.inspect]), max_factual_records=1, record_filter=bad_filter)

    def test_prefix_limit_must_be_positive(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_factual_records"):
                    self._run(ScriptPolicy([]), max_factual_records=value)
        self.assertEqual(self.tool_calls, [])


if __name__ == "__main__":
    unittest.main()
