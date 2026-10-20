"""Exact-ID region and training-reference checks without models or assets."""
import json
from types import SimpleNamespace
import unittest

from methods.packet_opsd.completion_regions import CompletionRegionError, completion_region
from methods.packet_opsd.train import teacher_adapter_name


class OffsetTokenizer:
    is_fast = True
    pieces = {100000: "<EOS>", 100001: ":{"}

    def encode(self, text, **kwargs):
        return self(text, **kwargs)["input_ids"]

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(i, chr(i)) for i in ids)

    def __call__(self, text, **kwargs):
        ids, offsets, at = [], [], 0
        while at < len(text):
            found = next(((token, piece) for token, piece in self.pieces.items()
                          if text.startswith(piece, at)), None)
            token, piece = found if found else (ord(text[at]), text[at])
            ids.append(token); offsets.append((at, at + len(piece))); at += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.tok = OffsetTokenizer()
        self.args = {"spans": [{"start_time": "00:00:01", "end_time": "00:00:03"}],
                     "context": 'quoted {brace} and "name" and nested JSON'}
        self.call = '<tool_call>' + json.dumps({"name": "visual_inspect", "arguments": self.args}) + '</tool_call>'
        self.text = 'reasoning</think>\n' + self.call + '<EOS>'

    def completion(self, text=None, **changes):
        text = self.text if text is None else text
        values = dict(token_ids=self.tok.encode(text), complete=True, stop_reason="eos")
        values.update(changes)
        return SimpleNamespace(**values)

    def select(self, region, completion=None, tokenizer=None):
        return completion_region(tokenizer or self.tok, completion or self.completion(),
                                 region=region, eos_token_ids=(100000,))

    def test_full_default_has_no_new_parser_or_fast_tokenizer_requirement(self):
        selection = self.select("full", self.completion("unstructured factual text"), SimpleNamespace())
        self.assertIsNone(selection.mask)
        self.assertEqual(selection.selected_tokens, selection.full_tokens)

    def test_action_is_complete_actual_call_and_excludes_reasoning_and_eos(self):
        selection = self.select("action")
        chosen = [token for token, yes in zip(self.completion().token_ids, selection.mask) if yes]
        self.assertEqual(self.tok.decode(chosen), self.call)
        self.assertFalse(selection.mask[-1])
        self.assertEqual(selection.full_tokens, len(self.completion().token_ids))

    def test_arguments_use_exact_nested_json_value_not_name_or_substring(self):
        selection = self.select("arguments")
        chosen = [token for token, yes in zip(self.completion().token_ids, selection.mask) if yes]
        self.assertEqual(self.tok.decode(chosen), json.dumps(self.args))
        self.assertLess(selection.selected_tokens, selection.full_tokens)

    def test_token_crossing_argument_boundary_is_excluded_without_retokenizing(self):
        text = self.text.replace('"arguments": {', '"arguments":{')
        completion = self.completion(text)
        self.assertIn(100001, completion.token_ids)
        arguments = self.select("arguments", completion)
        action = self.select("action", completion)
        at = completion.token_ids.index(100001)
        self.assertFalse(arguments.mask[at])
        self.assertTrue(action.mask[at])
        self.assertEqual(arguments.boundary_tokens_excluded, 1)
        self.assertEqual(arguments.full_tokens, len(completion.token_ids))

    def test_tool_example_before_think_close_refuses_ambiguous_harness_action(self):
        example = '<tool_call>{"name":"visual_retrieve","arguments":{"query":"example"}}</tool_call>'
        completion = self.completion(example + ' is only a quotation</think>\n' + self.call + '<EOS>')
        with self.assertRaisesRegex(CompletionRegionError, 'exactly-one-tool-marker-pair'):
            self.select("action", completion)

    def test_bad_first_call_is_never_skipped_to_supervise_later_valid_call(self):
        first_calls = ('<tool_call>{bad json}</tool_call>',
                       '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
                       '<tool_call>{"name":"visual_inspect","arguments":{}}')
        for first in first_calls:
            for region in ("action", "arguments"):
                with self.subTest(first=first, region=region), self.assertRaises(CompletionRegionError):
                    self.select(region, self.completion(first + self.call + '<EOS>'))

    def test_extra_close_marker_is_rejected_even_with_one_opener(self):
        with self.assertRaisesRegex(CompletionRegionError, 'exactly-one-tool-marker-pair'):
            self.select("arguments", self.completion(self.text + '</tool_call>'))

    def test_multiple_actual_calls_are_rejected(self):
        for text in ('</think>' + self.call + self.call + '<EOS>',
                     self.call + self.call + '<EOS>'):
            with self.subTest(text=text), self.assertRaises(CompletionRegionError):
                self.select("arguments", self.completion(text))

    def test_truncation_and_unclosed_or_nonterminal_suffix_are_rejected(self):
        variants = [self.completion(complete=False), self.completion(stop_reason="token_cap"),
                    self.completion(self.text.replace('</tool_call>', '')),
                    self.completion(self.text + 'unwanted tail')]
        for completion in variants:
            with self.subTest(completion=completion), self.assertRaises(CompletionRegionError):
                self.select("action", completion)

    def test_duplicate_keys_and_wrong_argument_type_are_rejected(self):
        for obj in ('{"name":"visual_inspect","name":"visual_retrieve","arguments":{}}',
                    '{"name":"visual_inspect","arguments":[]}'):
            with self.subTest(obj=obj), self.assertRaises(CompletionRegionError):
                self.select("arguments", self.completion('</think><tool_call>' + obj + '</tool_call><EOS>'))

    def test_non_fast_roundtrip_and_invalid_offsets_refuse_nonfull(self):
        self.tok.is_fast = False
        with self.assertRaisesRegex(CompletionRegionError, 'fast-tokenizer'):
            self.select("arguments")
        self.tok.is_fast = True
        class BrokenTokenizer(OffsetTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if kwargs.get('return_offsets_mapping'):
                    result['input_ids'][0] += 1
                return result
        with self.assertRaisesRegex(CompletionRegionError, 'roundtrip'):
            self.select("arguments", tokenizer=BrokenTokenizer())
        class BrokenOffset(OffsetTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result['offset_mapping'][0] = (-1, 1)
                return result
        with self.assertRaisesRegex(CompletionRegionError, 'offset-bounds'):
            self.select("arguments", tokenizer=BrokenOffset())
        class UnsupportedDecode(OffsetTokenizer):
            def decode(self, ids, **kwargs):
                raise NotImplementedError("no exact full-string decode")
        with self.assertRaisesRegex(CompletionRegionError, 'invalid-kl-offset'):
            self.select("arguments", tokenizer=UnsupportedDecode())

    def test_same_teacher_selector_for_both_KL_and_candidate_reference(self):
        self.assertEqual(teacher_adapter_name("ema"), "teacher")
        self.assertEqual(teacher_adapter_name("fixed-base"), "base")
        with self.assertRaises(ValueError):
            teacher_adapter_name("unknown")


if __name__ == "__main__":
    unittest.main()
