"""Stopping-contract checks using scripted token streams, with no model assets."""

import importlib.util
from types import SimpleNamespace
import unittest

from methods.packet_opsd.objective import INSPECTION_PREFIX, candidate_spans


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
if AVAILABLE:
    import torch
    from methods.packet_opsd.policy import HFPolicy


class ScriptTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pieces = {
        2: "<|im_end|>", 3: "<|endoftext|>",
        4: '{"spans":[{"start_time":"00:00:01","end_time":"00:00:03"}]}}',
        5: "</tool_", 6: "call>", 7: "UNWANTED_TAIL", 9: "</tool_call>",
    }

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        return "ORIGINAL_USER_HISTORY<assistant><think>\n"

    def encode(self, text, *, add_special_tokens=False):
        if text == "</tool_call>":
            return [9]
        self.last_prompt_text = text
        return [1]

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(self.pieces.get(int(token_id), "?") for token_id in token_ids
                       if not skip_special_tokens or int(token_id) not in (2, 3, 9))


@unittest.skipUnless(AVAILABLE, "torch/transformers are not installed")
class CandidateStoppingTest(unittest.TestCase):
    def policy(self, stream):
        class ScriptModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 2)
                self.config = SimpleNamespace(eos_token_id=2)
                self.generation_config = SimpleNamespace(eos_token_id=[2, 3])
                self.calls = []

            def get_input_embeddings(self):
                return self.embedding

            def set_adapter(self, name):
                self.active_adapter = name

            def generate(self, *, input_ids, attention_mask, **kwargs):
                self.calls.append(kwargs)
                output = input_ids
                eos = kwargs["eos_token_id"]
                for token_id in stream[:kwargs["max_new_tokens"]]:
                    output = torch.cat((output, torch.tensor([[token_id]], device=output.device)), dim=1)
                    if token_id in eos:
                        break
                    criteria = kwargs.get("stopping_criteria")
                    if criteria is not None and bool(criteria(output, None)):
                        break
                return output

        return HFPolicy(ScriptModel(), ScriptTokenizer(), max_context=32)

    def test_student_and_teacher_stop_at_configured_end_of_text(self):
        for adapter in ("student", "teacher"):
            with self.subTest(adapter=adapter):
                policy = self.policy([4, 3, 7, 7, 7])
                completion = policy.sample([], adapter=adapter, prefix=INSPECTION_PREFIX, max_new_tokens=8)
                self.assertEqual(completion.token_ids, [4, 3])
                self.assertTrue(completion.complete)
                self.assertEqual(completion.stop_reason, "eos")
                self.assertEqual(completion.stop_token_id, 3)
                self.assertEqual(policy.model.calls[0]["eos_token_id"], [2, 3])
                self.assertIsNone(candidate_spans(completion))  # EOS is not a fabricated tool ending
                self.assertEqual(len(policy.model.calls), 1)

    def test_special_tool_marker_is_preserved_and_no_tail_is_sampled(self):
        policy = self.policy([4, 9, 7, 7, 2])
        completion = policy.sample([], prefix=INSPECTION_PREFIX, max_new_tokens=8)
        self.assertEqual(completion.token_ids, [4, 9])
        self.assertTrue(completion.complete)
        self.assertEqual(completion.stop_reason, "tool_end")
        self.assertTrue(completion.text.endswith("</tool_call>"))
        self.assertNotIn("UNWANTED_TAIL", completion.text)
        self.assertIsNotNone(candidate_spans(completion))
        self.assertEqual(len(policy.model.calls), 1)
        rendered = policy.tokenizer.last_prompt_text
        self.assertEqual(rendered.count("<think>"), 1)
        self.assertEqual(rendered.count("</think>"), 1)

    def test_tokenizer_eos_is_unioned_with_configured_or_fallback_eos(self):
        for configured in ([3], 3, None):
            with self.subTest(configured=configured):
                policy = self.policy([4, 2, 7])
                policy.model.generation_config.eos_token_id = configured
                policy.model.config.eos_token_id = 3
                completion = policy.sample([], prefix=INSPECTION_PREFIX, max_new_tokens=8)
                self.assertEqual(policy.model.calls[0]["eos_token_id"], [3, 2])
                self.assertEqual(completion.token_ids, [4, 2])
                self.assertEqual(completion.stop_reason, "eos")
                self.assertEqual(completion.stop_token_id, 2)

    def test_contextual_multitoken_marker_also_stops(self):
        policy = self.policy([4, 5, 6, 7, 2])
        completion = policy.sample([], prefix=INSPECTION_PREFIX, max_new_tokens=8)
        self.assertEqual(completion.token_ids, [4, 5, 6])
        self.assertEqual(completion.stop_reason, "tool_end")
        self.assertIsNotNone(candidate_spans(completion))

    def test_hard_cap_remains_invalid_without_repair_or_resampling(self):
        policy = self.policy([4, 5, 6, 9, 2])
        completion = policy.sample([], prefix=INSPECTION_PREFIX, max_new_tokens=2)
        self.assertEqual(completion.token_ids, [4, 5])
        self.assertFalse(completion.complete)
        self.assertEqual(completion.stop_reason, "token_cap")
        self.assertIsNone(candidate_spans(completion))
        self.assertEqual(len(policy.model.calls), 1)

    def test_factual_planner_keeps_original_no_prefix_decoding(self):
        policy = self.policy([4, 9, 7, 2])
        completion = policy.sample([], max_new_tokens=8)
        self.assertEqual(completion.token_ids, [4, 9, 7, 2])
        self.assertNotIn("stopping_criteria", policy.model.calls[0])
        self.assertNotIn("</tool_call>", completion.text)
        self.assertEqual(completion.stop_reason, "eos")


if __name__ == "__main__":
    unittest.main()
