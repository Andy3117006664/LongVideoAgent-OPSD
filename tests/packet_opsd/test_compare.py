"""Terminal scoring tests use synthetic choices, never benchmark gold."""
import json
from pathlib import Path
import tempfile
import unittest

from methods.packet_opsd.compare import (
    compare_fixed300, exact_mcnemar, main, paired_bootstrap, paired_video_bootstrap,
)


class PairedComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mapping = self.root / "mapping.json"
        self.mapping.write_text(json.dumps([
            {"ordinal": i, "qid": str(i), "sample_uid": f"lvbench:{i}",
             "video_id": f"video{i}", "original_video_id": f"oldvideo{i}"}
            for i in range(300)
        ]))
        self.gold = [{"uid": str(i), "video_id": f"video{i}", "gold": "A"} for i in range(300)]
        self.base_search = [self.search(i) for i in range(300)]
        self.trained_search = [self.search(i) for i in range(300)]
        self.base_final = [self.final(row) for row in self.base_search]
        self.trained_final = [self.final(row) for row in self.trained_search]

    @staticmethod
    def search(i, prediction="A", state="completed"):
        return {"uid": str(i), "video_id": f"video{i}", "ordinal": i,
                "prediction": prediction, "state": state}

    @staticmethod
    def final(search, *, forced=False, observer="", status=None):
        adopted = forced and observer in {"A", "B", "C", "D"}
        return {"uid": search["uid"], "video_id": search["video_id"], "ordinal": search["ordinal"],
                "original_prediction": search["prediction"], "search_state": search["state"],
                "has_forced_full_video": forced, "observer_prediction": observer,
                "observer_status": status,
                "final_source": "DirectObserver" if adopted else "OriginalSearch",
                "final_prediction": observer if adopted else search["prediction"]}

    def files(self):
        paths = {"identity_mapping": self.mapping}
        for label, rows in (("baseline_search", self.base_search), ("baseline_final", self.base_final),
                            ("trained_search", self.trained_search), ("trained_final", self.trained_final),
                            ("gold", self.gold)):
            path = self.root / f"{label}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[label] = path
        return paths

    def compare(self):
        return compare_fixed300(**self.files())

    def test_direction_pairing_and_order_independence(self):
        # Twelve new correct answers and three regressions give +9, not +15.
        for i in range(12):
            self.base_search[i]["prediction"] = "B"
        for i in range(12, 15):
            self.trained_search[i]["prediction"] = "C"
        self.base_final = [self.final(row) for row in self.base_search]
        self.trained_final = [self.final(row) for row in self.trained_search]
        self.trained_search.reverse()
        self.base_final.reverse()
        self.gold.reverse()
        report = self.compare()
        for stage in ("search", "final"):
            scores = report[stage]
            self.assertEqual(scores["baseline"]["correct"], 288)
            self.assertEqual(scores["trained"]["correct"], 297)
            self.assertEqual(scores["trained_only_gain"], 12)
            self.assertEqual(scores["baseline_only_loss"], 3)
            self.assertEqual(scores["net_correct"], 9)
            self.assertAlmostEqual(scores["accuracy_difference_percentage_points"], 3)
            self.assertAlmostEqual(scores["mcnemar"]["p_value"], .03515625)
            bootstrap = scores["paired_bootstrap"]
            self.assertEqual(bootstrap["replicates"], 10_000)
            self.assertEqual(bootstrap["seed"], 20261018)
        serialized = json.dumps(report)
        for forbidden in ("video0", '"uid"', '"gold":', str(self.root)):
            self.assertNotIn(forbidden, serialized)

    def test_failed_search_counts_wrong_even_with_correct_prediction_observer_can_rescue(self):
        self.base_search[0]["state"] = "failed"
        self.base_final[0] = self.final(self.base_search[0])
        self.trained_search[0]["state"] = "timed_out"
        self.trained_final[0] = self.final(self.trained_search[0], forced=True, observer="A", status="ok")
        report = self.compare()
        self.assertEqual(report["denominator"], 300)
        self.assertEqual(report["search"]["baseline"]["correct"], 299)
        self.assertEqual(report["search"]["trained"]["correct"], 299)
        self.assertEqual(report["final"]["baseline"]["correct"], 299)
        self.assertEqual(report["final"]["trained"]["correct"], 300)
        self.assertEqual(report["final"]["trained_only_gain"], 1)
        self.assertEqual(report["arms"]["trained"]["failed_search_rows"], 1)

    def test_all_missing_and_duplicate_input_variants_refuse_partial_report(self):
        for attribute in ("base_search", "base_final", "trained_search", "trained_final", "gold"):
            original = getattr(self, attribute)
            with self.subTest(input=attribute, problem="missing"):
                setattr(self, attribute, original[:-1])
                with self.assertRaisesRegex(ValueError, "exactly 300"):
                    self.compare()
            with self.subTest(input=attribute, problem="duplicate"):
                setattr(self, attribute, original[:-1] + [original[0]])
                with self.assertRaisesRegex(ValueError, "duplicate normalized"):
                    self.compare()
            setattr(self, attribute, original)

    def test_final_cannot_be_mixed_with_different_search_arm(self):
        for key, changed, message in (
            ("original_prediction", "D", "original_prediction"),
            ("search_state", "failed", "search_state"),
            ("final_source", "DirectObserver", "adoption rule"),
            ("final_prediction", "D", "adoption rule"),
        ):
            with self.subTest(key=key):
                self.trained_final[0] = {**self.final(self.trained_search[0]), key: changed}
                with self.assertRaisesRegex(ValueError, message):
                    self.compare()
        self.trained_final[0] = self.final(self.trained_search[0], forced=False, observer="B")
        # A valid unforced Observer answer must not be adopted.
        self.trained_final[0]["final_source"] = "DirectObserver"
        self.trained_final[0]["final_prediction"] = "B"
        with self.assertRaisesRegex(ValueError, "adoption rule"):
            self.compare()

    def test_original_adoption_uses_exact_letter_and_forced_flag_not_status(self):
        self.trained_final[0] = self.final(self.trained_search[0], forced=True, observer="A", status="error")
        self.trained_final[1] = self.final(self.trained_search[1], forced=True, observer="a", status="ok")
        self.trained_final[2] = self.final(self.trained_search[2], forced=True, observer="Answer A", status="ok")
        report = self.compare()
        self.assertEqual(report["arms"]["trained"]["observer_adopted"], 1)
        self.assertEqual(report["final"]["trained"]["correct"], 300)

    def test_gold_is_never_opened_until_both_arms_are_complete_and_consistent(self):
        paths = self.files()
        paths["gold"].unlink()
        self.trained_final[0]["original_prediction"] = "D"
        paths["trained_final"].write_text("".join(json.dumps(row) + "\n" for row in self.trained_final))
        with self.assertRaisesRegex(ValueError, "original_prediction"):
            compare_fixed300(**paths)

    def test_explicit_aliases_normalize_but_wrong_video_or_ordinal_is_rejected(self):
        self.gold[0].update(uid="lvbench:0", video_id="oldvideo0")
        self.trained_search[0].update(uid="lvbench:0", video_id="oldvideo0")
        self.trained_final[0].update(uid="lvbench:0", video_id="oldvideo0")
        self.assertEqual(self.compare()["final"]["trained"]["correct"], 300)
        self.trained_search[0]["video_id"] = "video1"
        with self.assertRaisesRegex(ValueError, "video identity"):
            self.compare()
        self.trained_search[0]["video_id"] = "video0"
        self.trained_final[0]["ordinal"] = 1
        with self.assertRaisesRegex(ValueError, "ordinal"):
            self.compare()

    def test_invalid_gold_schema_or_choice_rejected(self):
        for changed in ({"question": "must not be ingested"}, {"gold": "a"},
                        {"gold": "A. red"}, {"gold": []}):
            with self.subTest(changed=changed):
                self.gold[0] = {"uid": "0", "video_id": "video0", "gold": "A", **changed}
                with self.assertRaisesRegex(ValueError, "Gold"):
                    self.compare()

    def test_mapping_must_be_unique_complete_and_unambiguous(self):
        original = json.loads(self.mapping.read_text())
        for key, value, message in (("qid", "1", "Duplicate qid"),
                                    ("ordinal", 1, "Duplicate qid or ordinal"),
                                    ("ordinal", 300, "cover 0 through 299"),
                                    ("ordinal", False, "integer ordinal"),
                                    ("sample_uid", "lvbench:1", "Ambiguous UID alias")):
            with self.subTest(key=key, value=value):
                changed = [dict(row) for row in original]
                changed[0][key] = value
                self.mapping.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, message):
                    self.compare()
        self.mapping.write_text(json.dumps(original[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly 300"):
            self.compare()

    def test_completed_invalid_choice_is_wrong_without_reparsing(self):
        for i, text in enumerate(("a", "Answer A", "", "A. red")):
            self.trained_search[i]["prediction"] = text
            self.trained_final[i] = self.final(self.trained_search[i])
        report = self.compare()
        self.assertEqual(report["search"]["trained"]["correct"], 296)
        self.assertEqual(report["final"]["trained"]["correct"], 296)
        self.assertEqual(report["arms"]["trained"]["failed_search_rows"], 0)

    def test_video_cluster_report_uses_mapping_canonical_video_not_input_alias(self):
        mapping = json.loads(self.mapping.read_text())
        for i, row in enumerate(mapping):
            cluster = i // 150
            row.update(video_id=f"cluster{cluster}", original_video_id=f"oldcluster{cluster}")
            # The same canonical video deliberately uses both accepted aliases.
            alias = f"cluster{cluster}" if i % 2 else f"oldcluster{cluster}"
            for collection in (self.base_search, self.base_final, self.trained_search,
                               self.trained_final, self.gold):
                collection[i]["video_id"] = alias
        self.mapping.write_text(json.dumps(mapping))
        report = self.compare()
        for stage in ("search", "final"):
            cluster = report[stage]["paired_video_bootstrap"]
            self.assertEqual(cluster["video_clusters"], 2)
            self.assertEqual(cluster["replicates"], 10_000)
            self.assertEqual(cluster["seed"], 20261018)
            self.assertEqual(cluster["accuracy_difference_ci"], [0, 0])
            self.assertIn("independent question", report[stage]["mcnemar"]["assumption"])
        self.assertNotIn("cluster0", json.dumps(report))

    def test_smoke_flag_and_duplicate_json_key_rejected(self):
        self.base_search[0]["smoke"] = True
        with self.assertRaisesRegex(ValueError, "smoke or partial"):
            self.compare()
        del self.base_search[0]["smoke"]
        paths = self.files()
        text = paths["baseline_search"].read_text()
        paths["baseline_search"].write_text(text.replace('"state": "completed"', '"state": "failed", "state": "completed"', 1))
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            compare_fixed300(**paths)

    def test_cli_writes_aggregate_report_and_refuses_overwrite(self):
        paths = self.files()
        output = self.root / "aggregate.json"
        arguments = [value for key, path in paths.items() for value in ("--" + key.replace("_", "-"), str(path))]
        arguments.extend(("--output", str(output)))
        self.assertEqual(main(arguments), 0)
        self.assertEqual(json.loads(output.read_text())["final"]["net_correct"], 0)
        with self.assertRaises(SystemExit):
            main(arguments)


class PairedStatisticsTest(unittest.TestCase):
    def test_exact_two_sided_binomial_known_values_and_symmetry(self):
        for gain, loss, expected in ((0, 0, 1), (1, 0, 1), (2, 0, .5),
                                     (5, 0, .0625), (4, 1, .375), (3, 3, 1),
                                     (12, 3, .03515625)):
            self.assertAlmostEqual(exact_mcnemar(gain, loss), expected)
            self.assertEqual(exact_mcnemar(gain, loss), exact_mcnemar(loss, gain))
        for pair in ((-1, 2), (1.5, 2), (True, 1)):
            with self.assertRaises(ValueError):
                exact_mcnemar(*pair)

    def test_bootstrap_resamples_question_pairs_and_is_reproducible(self):
        self.assertEqual(paired_bootstrap([0] * 300), [0, 0])
        self.assertEqual(paired_bootstrap([1] * 300), [1, 1])
        self.assertEqual(paired_bootstrap([-1] * 300), [-1, -1])
        differences = [1] * 10 + [-1] * 5 + [0] * 285
        first = paired_bootstrap(differences, seed=19, iterations=300)
        self.assertEqual(first, paired_bootstrap(differences, seed=19, iterations=300))
        negated = paired_bootstrap([-value for value in differences], seed=19, iterations=300)
        self.assertAlmostEqual(first[0], -negated[1])
        self.assertAlmostEqual(first[1], -negated[0])

    def test_video_bootstrap_retains_correlated_questions_and_is_reproducible(self):
        differences = [1] * 100 + [-1] * 100
        videos = ["positive"] * 100 + ["negative"] * 100
        question = paired_bootstrap(differences)
        video = paired_video_bootstrap(differences, videos)
        # Two perfectly correlated videos contain far less information than
        # 200 independent questions; whole-cluster draws can be all +/-1.
        self.assertEqual(video, [-1, 1])
        self.assertLess(video[0], question[0])
        self.assertGreater(video[1], question[1])
        self.assertEqual(video, paired_video_bootstrap(differences, videos))
        self.assertEqual(paired_video_bootstrap([0] * 4, ["one", "one", "two", "two"]), [0, 0])

    def test_video_bootstrap_weights_by_drawn_questions_not_mean_video_accuracy(self):
        # Seed 4 draws the two distinct clusters once each. Their sizes 3 and 1
        # imply (3 - 1) / 4 = .5, rather than (1 - 1) / 2 = 0.
        self.assertEqual(paired_video_bootstrap([1, 1, 1, -1], ["long"] * 3 + ["short"],
                                               seed=4, iterations=1), [.5, .5])
        self.assertEqual(paired_video_bootstrap([1, 0, -1], ["one"] * 3), [0, 0])
        for differences, videos in (([], []), ([0], []), ([2], ["one"]), ([0], [""])):
            with self.assertRaises(ValueError):
                paired_video_bootstrap(differences, videos)


if __name__ == "__main__":
    unittest.main()
