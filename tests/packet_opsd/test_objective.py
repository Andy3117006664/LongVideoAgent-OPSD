"""Exact finite-space checks; no model, media or numerical packages needed."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from methods.packet_opsd.objective import (
    candidate_spans, hindsight_messages, packet_coefficient,
    symmetric_packet_coefficients,
)


def softmax(logits):
    weights = [math.exp(value - max(logits)) for value in logits]
    return [value / sum(weights) for value in weights]


def kernel(left, right):
    # Actions 0 and 1 emit the same complete packet. Action 3 is the single
    # invalid-packet symbol, which still has probability and a score gradient.
    if left is None or right is None:
        return float(left is None and right is None)
    return math.exp(-sum((x - y) ** 2 for x, y in zip(left, right)) / 2)


PACKETS = [(0.0, 1.0), (0.0, 1.0), (1.0, 0.0), None]


def mmd(logits, teacher):
    student = softmax(logits)
    return sum(
        (student[i] - teacher[i]) * (student[j] - teacher[j])
        * kernel(PACKETS[i], PACKETS[j])
        for i in range(len(student)) for j in range(len(student))
    )


def exact_score_gradient(logits, teacher):
    student = softmax(logits)
    gradient = [0.0] * len(student)
    for i, pi in enumerate(student):
        for j, pj in enumerate(student):
            for k, qk in enumerate(teacher):
                coefficient = packet_coefficient(
                    PACKETS[i], PACKETS[j], PACKETS[k], kernel=kernel,
                ).value
                for coordinate in range(len(student)):
                    score = float(i == coordinate) - student[coordinate]
                    gradient[coordinate] += pi * pj * qk * coefficient * score
    return gradient


def exact_symmetric_score_gradient(logits, teacher):
    student = softmax(logits)
    gradient = [0.0] * len(student)
    for i, pi in enumerate(student):
        for j, pj in enumerate(student):
            for k, qk in enumerate(teacher):
                coefficient = symmetric_packet_coefficients(
                    PACKETS[i], PACKETS[j], PACKETS[k], kernel=kernel,
                )
                for coordinate in range(len(student)):
                    first_score = float(i == coordinate) - student[coordinate]
                    second_score = float(j == coordinate) - student[coordinate]
                    gradient[coordinate] += pi * pj * qk * (
                        coefficient.first_value * first_score
                        + coefficient.second_value * second_score
                    )
    return gradient


class PacketObjectiveTest(unittest.TestCase):
    def test_symmetric_score_gradient_matches_finite_difference_and_asymmetric_expectation(self):
        logits = [0.4, -0.2, 0.8, -0.7]
        teacher = [0.1, 0.4, 0.2, 0.3]
        actual = exact_symmetric_score_gradient(logits, teacher)
        asymmetric = exact_score_gradient(logits, teacher)
        epsilon = 1e-5
        for coordinate in range(len(logits)):
            plus, minus = list(logits), list(logits)
            plus[coordinate] += epsilon
            minus[coordinate] -= epsilon
            expected = (mmd(plus, teacher) - mmd(minus, teacher)) / (2 * epsilon)
            self.assertAlmostEqual(actual[coordinate], expected, places=9)
            self.assertAlmostEqual(actual[coordinate], asymmetric[coordinate], places=12)

    def test_symmetric_estimator_keeps_second_score_when_second_packet_matches_teacher(self):
        value = symmetric_packet_coefficients(PACKETS[0], PACKETS[2], PACKETS[2], kernel=kernel)
        self.assertEqual(value.first_value, 0.0)
        self.assertLess(value.second_value, 0.0)
        self.assertEqual(value.second_teacher_similarity, 1.0)
        self.assertEqual(value.second_value, value.student_similarity - 1.0)

    def test_symmetric_coefficients_are_swap_equivariant(self):
        value = symmetric_packet_coefficients(PACKETS[0], PACKETS[2], PACKETS[0], kernel=kernel)
        swapped = symmetric_packet_coefficients(PACKETS[2], PACKETS[0], PACKETS[0], kernel=kernel)
        self.assertEqual(value.first_value, swapped.second_value)
        self.assertEqual(value.second_value, swapped.first_value)
        self.assertEqual(value.student_similarity, swapped.student_similarity)
        self.assertEqual(value.first_teacher_similarity, swapped.second_teacher_similarity)
        self.assertEqual(value.second_teacher_similarity, swapped.first_teacher_similarity)

    def test_symmetrization_reduces_exact_gradient_variance_without_new_draws(self):
        logits, teacher = [0.4, -0.2, 0.8, -0.7], [0.1, 0.4, 0.2, 0.3]
        student = softmax(logits)
        asymmetric_second_moment, symmetric_second_moment = 0.0, 0.0
        for i, pi in enumerate(student):
            for j, pj in enumerate(student):
                for k, qk in enumerate(teacher):
                    old = packet_coefficient(PACKETS[i], PACKETS[j], PACKETS[k], kernel=kernel)
                    new = symmetric_packet_coefficients(PACKETS[i], PACKETS[j], PACKETS[k], kernel=kernel)
                    first = [float(i == coordinate) - student[coordinate] for coordinate in range(len(student))]
                    second = [float(j == coordinate) - student[coordinate] for coordinate in range(len(student))]
                    asymmetric_second_moment += pi * pj * qk * sum((old.value * score) ** 2 for score in first)
                    symmetric_second_moment += pi * pj * qk * sum(
                        (new.first_value * a + new.second_value * ap) ** 2 for a, ap in zip(first, second)
                    )
        # The means were checked equal above, so comparing second moments is
        # exactly comparing the trace of the gradient covariance matrices.
        self.assertLess(symmetric_second_moment, asymmetric_second_moment)

    def test_exact_mmd_score_gradient_matches_finite_difference(self):
        logits = [0.4, -0.2, 0.8, -0.7]
        teacher = [0.1, 0.4, 0.2, 0.3]
        actual = exact_score_gradient(logits, teacher)
        epsilon = 1e-5
        for coordinate in range(len(logits)):
            plus, minus = list(logits), list(logits)
            plus[coordinate] += epsilon
            minus[coordinate] -= epsilon
            expected = (mmd(plus, teacher) - mmd(minus, teacher)) / (2 * epsilon)
            self.assertAlmostEqual(actual[coordinate], expected, places=9)
        self.assertAlmostEqual(sum(actual), 0.0, places=12)
        after_descent = [value - 0.1 * grad for value, grad in zip(logits, actual)]
        self.assertLess(mmd(after_descent, teacher), mmd(logits, teacher))

    def test_alias_action_mass_is_irrelevant_when_packet_laws_match(self):
        student = [0.2, 0.3, 0.25, 0.25]
        teacher = [0.4, 0.1, 0.25, 0.25]
        logits = [math.log(value) for value in student]
        self.assertAlmostEqual(mmd(logits, teacher), 0.0, places=12)
        for coordinate in exact_score_gradient(logits, teacher):
            self.assertAlmostEqual(coordinate, 0.0, places=12)
        for coordinate in exact_symmetric_score_gradient(logits, teacher):
            self.assertAlmostEqual(coordinate, 0.0, places=12)

    def test_coefficient_can_be_negative_and_is_not_an_mmd_value(self):
        value = packet_coefficient(PACKETS[0], PACKETS[2], PACKETS[0], kernel=kernel)
        self.assertLess(value.value, 0)
        self.assertEqual(value.value, 2 * (value.student_similarity - value.teacher_similarity))

    def test_parser_preserves_one_valid_draw_and_rejects_invalid_draws(self):
        spans = '[{"start_time":"00:00:01","end_time":"00:00:03"}]'
        valid = '{"spans":' + spans + '}}</tool_call>'
        parsed = candidate_spans(SimpleNamespace(text=valid, complete=True))
        self.assertEqual(parsed, [{"start_time": "00:00:01", "end_time": "00:00:03"}])
        for text, complete in (
            (valid, False),
            (valid + "extra rationale", True),
            (valid.replace("</tool_call>", ""), True),
            ('{"spans":[]}}</tool_call>', True),
            ('{"spans":{}}}</tool_call>', True),
            ('{"context":"no spans"}}</tool_call>', True),
            ('not json</tool_call>', True),
            ('{"spans":' + spans + '},"name":"visual_retrieve"}</tool_call>', True),
        ):
            with self.subTest(text=text, complete=complete):
                self.assertIsNone(candidate_spans(SimpleNamespace(text=text, complete=complete)))

    def test_hindsight_uses_natural_feedback_without_external_gold_fields(self):
        messages = [{"role": "system", "content": "ORIGINAL SYSTEM"},
                    {"role": "user", "content": "ORIGINAL HISTORY AND QUESTION"}]
        observation = {
            "name": "visual_inspect", "ok": True,
            "output": {"answer": "MODEL_GENERATED_C"}, "error": None,
            "metadata": {"timestamps": [1, 1.5],
                         "exclusive_inspector_output": {"raw_answer": "MODEL_GENERATED_C"}},
            "ground_truth": "EXTERNAL_GOLD_SENTINEL",
            "gt": "EXTERNAL_GT_SENTINEL", "correct": True,
            "reward": "EXTERNAL_REWARD_SENTINEL",
        }
        old_messages, old_observation = deepcopy(messages), deepcopy(observation)
        teacher = hindsight_messages(messages, observation)
        self.assertEqual(messages, old_messages)
        self.assertEqual(observation, old_observation)
        self.assertEqual(teacher[0], messages[0])
        self.assertTrue(teacher[-1]["content"].startswith(messages[-1]["content"]))
        self.assertIn("MODEL_GENERATED_C", teacher[-1]["content"])
        self.assertIn('"timestamps": [1, 1.5]', teacher[-1]["content"])
        self.assertNotIn("EXTERNAL_", str(teacher))
        self.assertNotIn('"correct"', str(teacher))
        self.assertEqual(len(teacher), len(messages))

    def test_training_manifest_rejects_gold_and_fixed300_uid_aliases(self):
        from methods.packet_opsd.train import load_tasks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            index = root / "index"
            index.mkdir()
            heldout, manifest = root / "held.json", root / "tasks.jsonl"
            heldout.write_text(json.dumps([
                {"ordinal": i, "sample_uid": f"lvbench:{qid}", "qid": qid,
                 "video_id": "same-video" if i == 0 else f"held-video-{i}",
                 "question_sha256": hashlib.sha256(f"held question {i}".encode()).hexdigest()}
                for i, qid in enumerate(["5256"] + [f"other-{i}" for i in range(299)])
            ]))
            task = {"uid": "fresh", "video_id": "same-video", "question": "toy question",
                    "video_path": str(video), "visual_index": str(index)}
            manifest.write_text(json.dumps({**task, "ground_truth": "FORBIDDEN"}) + "\n")
            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                load_tasks(manifest, heldout, allow_shared_videos=True)
            for uid in ("5256", "lvbench:5256", "another-runtime:5256"):
                manifest.write_text(json.dumps({**task, "uid": uid}) + "\n")
                with self.subTest(uid=uid):
                    with self.assertRaisesRegex(ValueError, "UID overlaps"):
                        load_tasks(manifest, heldout, allow_shared_videos=True)
            manifest.write_text(json.dumps(task) + "\n")
            self.assertEqual(len(load_tasks(manifest, heldout, allow_shared_videos=True)), 1)


if __name__ == "__main__":
    unittest.main()
