"""Checks for the separate, execution-backed Phase 3.5 challenge set."""
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from dataset.model_input import MODEL_INPUT_FIELDS
from dataset.reference_oracle import validate_outcome_record
from dataset.generate_challenge_dataset_v1 import generate


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "irreversibility_challenge_v1.jsonl"
ACTIONS = ("edit_file", "delete_file", "send_email", "transfer_money")
LABELS = ("reversible", "partially_reversible", "irreversible")


class ChallengeDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DATASET.open(encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle if line.strip()]

    def test_size_balance_and_unique_ids(self):
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(len({row["id"] for row in self.rows}), 240)
        counts = Counter((row["action_text"], row["label"]) for row in self.rows)
        for action in ACTIONS:
            for label in LABELS:
                self.assertEqual(counts[(action, label)], 20)

    def test_all_records_match_independent_reference_outcomes(self):
        for row in self.rows:
            with self.subTest(example=row["id"]):
                result = validate_outcome_record(row)
                self.assertTrue(result["passed"], result["errors"])
                self.assertEqual(row["source"], "sandbox_execution_challenge_v1")
                self.assertLessEqual(len(row["recovery_attempts"]), 3)
                self.assertEqual(row["state_phase"], "pre_action")

    def test_pairs_change_only_the_recovery_condition(self):
        pairs = defaultdict(list)
        for row in self.rows:
            pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 120)
        for pair_id, members in pairs.items():
            with self.subTest(pair=pair_id):
                self.assertEqual(len(members), 2)
                first, second = members
                self.assertEqual(first["action_text"], second["action_text"])
                self.assertEqual(first["scenario_id"], second["scenario_id"])
                self.assertEqual(first["scenario_group_id"], second["scenario_group_id"])
                self.assertEqual(first["canonical_group_id"], second["canonical_group_id"])
                self.assertEqual(first["wording_template_group_id"],
                                 second["wording_template_group_id"])
                self.assertNotEqual(first["label"], second["label"])
                key = ("transfer_condition" if first["action_text"] == "transfer_money"
                       else "recovery_condition")
                left, right = first["action_parameters"], second["action_parameters"]
                self.assertNotEqual(left[key], right[key])
                self.assertEqual({k: v for k, v in left.items() if k != key},
                                 {k: v for k, v in right.items() if k != key})

    def test_every_action_template_occurs_with_all_labels(self):
        grouped = defaultdict(set)
        for row in self.rows:
            grouped[(row["action_text"], row["wording_template_group_id"])].add(row["label"])
        self.assertEqual(len(grouped), 12)
        for key, labels in grouped.items():
            with self.subTest(template=key):
                self.assertEqual(labels, set(LABELS))

    def test_model_input_is_pre_action_whitelist_and_has_no_explicit_label_words(self):
        forbidden = ("reversible", "irreversible", "partial recovery", "permanent action")
        for row in self.rows:
            self.assertEqual(set(row["model_input"]), set(MODEL_INPUT_FIELDS))
            self.assertEqual(row["model_input"], {
                field: row[field] for field in MODEL_INPUT_FIELDS
            })
            lowered = row["state_text"].lower()
            self.assertFalse(any(term in lowered for term in forbidden))

    def test_generator_refuses_to_overwrite_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "already_frozen.jsonl"
            existing.write_text("preserve this", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate(existing)
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve this")


if __name__ == "__main__":
    unittest.main(verbosity=2)
