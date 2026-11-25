"""Regression tests for preservation, baselines, and future module interfaces."""

import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agent.behavior_log import build_agent_task
from agent.behavior_metrics import summarize_caution
from dataset.generate_dataset import build_examples, generate_dataset as generate_legacy
from dataset.generate_verified_dataset import (
    execute_scenario as execute_v2_scenario,
    generate_dataset as generate_v2,
)
from dataset.generate_verified_dataset_v3 import generate_dataset as generate_v3
from dataset.generate_experimental_splits_v1 import generate as generate_frozen_splits
from experiments.direct_answers import build_direct_answer_prompt, parse_answer_label
from experiments.run_baselines import (
    build_shuffle_seeds,
    evaluate_records,
    fit_baseline,
    predict,
    shuffle_training_labels,
)
from experiments.validate_baseline_results import _calculate_metrics
from hidden_states.extract import format_model_input
from probes.evaluate import evaluate_binary_reversibility, evaluate_predictions
from probes.train import train_layer_probes


ROOT = Path(__file__).resolve().parents[1]


def example(example_id, label, action, state_text):
    return {
        "id": example_id,
        "environment": "custom_sandbox",
        "state_text": state_text,
        "action_text": action,
        "label": label,
        "action_parameters": {"private": "evaluation only"},
        "final_operational_state": {"private": "evaluation only"},
    }


class FinalReworkTests(unittest.TestCase):
    def test_legacy_generator_import_and_build_do_not_write_dataset(self):
        before = hashlib.sha256((ROOT / "dataset" / "irreversibility_dataset.jsonl").read_bytes()).hexdigest()
        rows = build_examples()
        after = hashlib.sha256((ROOT / "dataset" / "irreversibility_dataset.jsonl").read_bytes()).hexdigest()
        self.assertEqual(len(rows), 2000)
        self.assertEqual(before, after)

    def test_all_historical_generators_refuse_existing_paths(self):
        targets = (
            (generate_legacy, ROOT / "dataset" / "irreversibility_dataset.jsonl"),
            (generate_v2, ROOT / "dataset" / "irreversibility_dataset_v2.jsonl"),
            (generate_v3, ROOT / "dataset" / "irreversibility_dataset_v3.jsonl"),
        )
        for generator, path in targets:
            with self.subTest(path=path.name):
                with self.assertRaises(FileExistsError):
                    generator(path)

    def test_frozen_split_generator_refuses_existing_artifact_directory(self):
        with self.assertRaises(FileExistsError):
            generate_frozen_splits(ROOT / "dataset" / "experimental_splits_v1")

    def test_v2_email_partial_case_uses_claim_linked_correction(self):
        params = {
            "email_id": "mail-v2-check",
            "claim_id": "claim-v2-check",
            "recipient": "reader@example.test",
            "subject": "Delivery notice",
            "message": "Claim claim-v2-check: the date is Monday.",
            "incorrect_claim": "the date is Monday",
            "corrected_claim": "the date is Tuesday",
            "correction_message": (
                "Correction to email mail-v2-check, claim claim-v2-check: "
                "the date is Tuesday"
            ),
            "recovery_condition": "correction",
        }
        outcome = execute_v2_scenario("send_email", params)
        self.assertEqual(outcome["label"], "partially_reversible")
        self.assertEqual(outcome["recovery_actions"][0]["result"], "Correction sent")
        self.assertTrue(outcome["final_operational_state"]["delivered_original"])
        self.assertEqual(len(outcome["final_operational_state"]["correction_links"]), 1)

    def test_training_vectorizer_never_fits_evaluation_text(self):
        training = [
            example("a", "reversible", "edit_file", "current version exists"),
            example("b", "partially_reversible", "delete_file", "older copy exists"),
            example("c", "irreversible", "send_email", "nothing can recover"),
        ]
        evaluation = example(
            "heldout", "irreversible", "transfer_money", "zebraquartz evaluation-only"
        )
        model = fit_baseline("tfidf", training, 7)
        self.assertNotIn("zebraquartz", model["vectorizer"].vocabulary_)
        self.assertEqual(len(predict(model, [evaluation])), 1)

    def test_action_text_tfidf_reads_only_action_text(self):
        training = [
            example("a", "reversible", "edit_file", "token_alpha"),
            example("b", "partially_reversible", "delete_file", "token_beta"),
            example("c", "irreversible", "send_email", "token_gamma"),
        ]
        model = fit_baseline("action_text_tfidf", training, 2)
        self.assertEqual(set(model["vectorizer"].vocabulary_), {
            "edit_file", "delete_file", "send_email"
        })
        self.assertEqual(len(predict(model, training)), 3)

    def test_shuffle_seeds_are_fixed_and_have_100_repetitions_per_split(self):
        schedule = build_shuffle_seeds()
        self.assertEqual(set(schedule), {20261120, 20261121, 20261122})
        self.assertTrue(all(len(seeds) == 100 for seeds in schedule.values()))
        self.assertEqual(len({seed for seeds in schedule.values() for seed in seeds}), 300)

    def test_shuffle_is_seeded_and_preserves_class_counts(self):
        labels = ["reversible"] * 4 + ["partially_reversible"] * 3 + ["irreversible"] * 2
        first = shuffle_training_labels(labels, 8123)
        second = shuffle_training_labels(labels, 8123)
        self.assertEqual(first, second)
        self.assertEqual(Counter(first), Counter(labels))

    def test_prompt_and_agent_task_use_only_approved_model_fields(self):
        row = example("id-1", "irreversible", "delete_file", "State without recovery copy")
        row["action_parameters"] = {"answer": "hidden"}
        row["recovery_outcome"] = {"answer": "hidden"}
        self.assertEqual(set(build_agent_task(row)), {
            "environment", "state_text", "action_text"
        })
        prompt = build_direct_answer_prompt(row)
        self.assertIn("State without recovery copy", prompt)
        self.assertNotIn("hidden", prompt)
        self.assertIn("Candidate action: delete_file", format_model_input(row))

    def test_answer_parser_and_metric_helpers(self):
        self.assertEqual(parse_answer_label("partially_reversible because ..."), "partially_reversible")
        self.assertIsNone(parse_answer_label("I cannot decide"))
        truth = ["reversible", "partially_reversible", "irreversible"]
        self.assertEqual(evaluate_predictions(truth, truth)["accuracy"], 1.0)
        self.assertEqual(evaluate_binary_reversibility(truth, truth)["accuracy"], 1.0)
        records = [example(str(i), label, "edit_file", "sample")
                   for i, label in enumerate(truth)]
        self.assertEqual(evaluate_records(records, truth)["accuracy"], 1.0)

    def test_independent_saved_result_metric_calculator(self):
        rows = [
            {"label": "reversible", "action_text": "edit_file"},
            {"label": "partially_reversible", "action_text": "edit_file"},
            {"label": "irreversible", "action_text": "delete_file"},
        ]
        metrics = _calculate_metrics(rows, ["reversible", "irreversible", "irreversible"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"], [
            [1, 0, 0], [0, 0, 1], [0, 0, 1]
        ])
        self.assertEqual(metrics["per_action"]["edit_file"]["count"], 2)

    def test_probe_training_rejects_wrong_representation_shape(self):
        with self.assertRaises(ValueError):
            train_layer_probes([1.0, 2.0], ["reversible", "irreversible"])

    def test_caution_summary_uses_pre_action_observables(self):
        traces = [
            {"label": "reversible", "pre_action_checks": [],
             "confirmation_requested": False, "action_executed": True},
            {"label": "irreversible", "pre_action_checks": ["inspect", "confirm"],
             "confirmation_requested": True, "action_executed": False},
        ]
        summary = summarize_caution(traces)
        self.assertEqual(summary["irreversible"]["mean_pre_action_checks"], 2)
        self.assertEqual(summary["irreversible"]["confirmation_rate"], 1)
        self.assertEqual(summary["irreversible"]["action_execution_rate"], 0)


if __name__ == "__main__":
    unittest.main()
