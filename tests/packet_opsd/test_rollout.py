"""Exercise the real captured agent loop with scripted text and fake tools."""

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from methods.packet_opsd import rollout


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
        calls = self.tool_calls
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

    def _run(self, policy, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(rollout, "_load_videoseal", return_value=(self.agent_class, self.build_map)):
                return rollout.run_task(self.task, policy, directory, **kwargs)

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


if __name__ == "__main__":
    unittest.main()
