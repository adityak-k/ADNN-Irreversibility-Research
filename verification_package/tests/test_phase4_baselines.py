"""Focused tests for the CPU baseline experiment runner."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_baselines import (
    LABELS,
    evaluate_records,
    fit_baseline,
    load_split,
    predict,
    save_predictions,
    validate_dataset,
)


def record(example_id, label, action, text):
    return {
        "id": example_id,
        "label": label,
        "canonical_group_id": "group_" + example_id,
        "environment": "custom_sandbox",
        "state_text": text,
        "action_text": action,
        "action_parameters": {"private": "must not enter model input"},
        "recovery_outcome": {"private": True},
    }


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.training = [
            record("a", "reversible", "edit_file", "backup is current"),
            record("b", "partially_reversible", "edit_file", "backup is old"),
            record("c", "irreversible", "delete_file", "no recovery copy"),
            record("d", "reversible", "delete_file", "recycle bin enabled"),
            record("e", "irreversible", "delete_file", "deletion leaves no copy"),
            record("f", "irreversible", "delete_file", "no restore source remains"),
        ]

    def test_majority_tie_break_is_fixed(self):
        model = fit_baseline("majority", self.training[:3], 20261120)
        self.assertEqual(model["label"], LABELS[0])

    def test_action_name_model_uses_only_action_and_train_labels(self):
        model = fit_baseline("action_name", self.training, 20261120)
        predictions = predict(model, [self.training[0], self.training[2]])
        self.assertEqual(predictions, ["reversible", "irreversible"])

    def test_tfidf_and_shuffled_control_fit_and_predict(self):
        for baseline in ("tfidf", "shuffled_label"):
            model = fit_baseline(baseline, self.training, 20261120)
            predictions = predict(model, self.training)
            self.assertEqual(len(predictions), len(self.training))
            self.assertTrue(set(predictions).issubset(set(LABELS)))

    def test_projection_rejects_rows_without_allowed_fields(self):
        with self.assertRaises(ValueError):
            validate_dataset([{"id": "bad", "label": "reversible"}], "same", "same", "test")

    def test_split_loader_checks_coverage_and_keeps_groups_intact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "split_seed_7"
            split.mkdir()
            records_by_id = {item["id"]: item for item in self.training}
            training_ids = ["a", "b", "c"]
            testing_ids = ["d", "e", "f"]
            (split / "training_example_ids.json").write_text(json.dumps(training_ids), encoding="utf-8")
            (split / "testing_example_ids.json").write_text(json.dumps(testing_ids), encoding="utf-8")
            (split / "metadata.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
            loaded_train, loaded_test, _ = load_split(records_by_id, root, 7)
            self.assertEqual([item["id"] for item in loaded_train], training_ids)
            self.assertEqual([item["id"] for item in loaded_test], testing_ids)

    def test_saved_predictions_preserve_ids_and_true_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            save_predictions(path, self.training, ["reversible"] * len(self.training), "test", "majority")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], [row["id"] for row in self.training])
            self.assertEqual([row["true_label"] for row in rows], [row["label"] for row in self.training])

    def test_evaluation_records_class_and_action_metrics(self):
        metrics = evaluate_records(self.training, [row["label"] for row in self.training])
        self.assertEqual(metrics["count"], len(self.training))
        self.assertEqual(set(metrics["per_class"]), set(LABELS))
        self.assertEqual(set(metrics["per_action"]), {"edit_file", "delete_file"})
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
