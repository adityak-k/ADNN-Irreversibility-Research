"""Regression checks for repaired v3 partial recovery and input controls."""
import copy
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from dataset.independent_outcomes import derive_expected_label, validate_example
from dataset.model_input import MODEL_INPUT_FIELDS, build_model_input
from dataset.validate_verified_dataset_v3 import duplicate_input_groups


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "irreversibility_dataset_v3.jsonl"


class Phase32DatasetQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DATASET.open(encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle]

    def test_genuine_partial_file_recovery_retains_shared_version_content(self):
        for action in ("edit_file", "delete_file"):
            row = next(item for item in self.rows
                       if item["action_text"] == action
                       and item["label"] == "partially_reversible")
            params = row["action_parameters"]
            recovered = row["final_operational_state"]["content"]
            original = row["initial_operational_state"]["content"]
            self.assertIn(params["shared_original_unit"], original)
            self.assertIn(params["shared_original_unit"], recovered)
            self.assertIn(params["current_only_unit"], original)
            self.assertNotIn(params["current_only_unit"], recovered)
            self.assertIn(params["old_only_unit"], recovered)
            self.assertNotEqual(recovered, original)
            self.assertEqual(derive_expected_label(row), "partially_reversible")

    def test_partial_email_correction_is_claim_linked_and_separately_delivered(self):
        row = next(item for item in self.rows
                   if item["action_text"] == "send_email"
                   and item["label"] == "partially_reversible")
        params = row["action_parameters"]
        final = row["final_operational_state"]
        original = final["delivered_original"]
        link = final["correction_links"][0]
        correction = final["delivered_corrections"][link["correction_email_id"]]
        self.assertEqual(original["message"], params["message"])
        self.assertEqual(link["original_email_id"], params["email_id"])
        self.assertEqual(link["claim_id"], params["claim_id"])
        self.assertEqual(link["original_claim"], params["incorrect_claim"])
        self.assertEqual(link["corrected_claim"], params["corrected_claim"])
        self.assertIn(params["email_id"], correction["message"])
        self.assertIn(params["claim_id"], correction["message"])
        self.assertIn(params["corrected_claim"], correction["message"])
        self.assertEqual(correction["recipient"], original["recipient"])
        self.assertEqual(derive_expected_label(row), "partially_reversible")
        self.assertTrue(validate_example(row)["passed"])
        corrupted = copy.deepcopy(row)
        corrupted["final_operational_state"]["correction_links"][0]["corrected_claim"] = "unrelated claim"
        self.assertFalse(validate_example(corrupted, check_replay=False)["passed"])

    def test_independent_checker_rejects_deliberately_incorrect_label(self):
        row = copy.deepcopy(next(item for item in self.rows if item["label"] == "partially_reversible"))
        row["label"] = "irreversible"
        result = validate_example(row)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Stored label" in error for error in result["errors"]))

    def test_model_input_is_exactly_the_whitelist(self):
        row = next(item for item in self.rows if "stale_content" in item["action_parameters"])
        projected = build_model_input(row)
        self.assertEqual(tuple(projected), MODEL_INPUT_FIELDS)
        self.assertEqual(set(projected), {"environment", "state_text", "action_text"})
        self.assertNotIn("stale_content", projected)
        self.assertNotIn("correction_message", projected)
        self.assertNotIn("label", projected)
        self.assertNotIn("audit_history", projected)
        self.assertIn("action_parameters", row["evaluation_only_fields"])
        self.assertIn("recovery_attempts", row["evaluation_only_fields"])

    def test_duplicate_prompt_detector_groups_identical_observations(self):
        example = self.rows[0]
        duplicate = copy.deepcopy(example)
        duplicate["id"] = "TEST_DUPLICATE"
        duplicate["label"] = example["label"]
        groups = duplicate_input_groups([example, duplicate])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_pairs_hold_scenario_constant_and_change_only_recovery_condition(self):
        pairs = defaultdict(list)
        for row in self.rows:
            if row["pair_id"]:
                pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 720)
        for members in pairs.values():
            self.assertEqual(len(members), 2)
            first, second = members
            self.assertEqual(first["action_text"], second["action_text"])
            self.assertEqual(first["scenario_group_id"], second["scenario_group_id"])
            field = "transfer_condition" if first["action_text"] == "transfer_money" else "recovery_condition"
            left, right = first["action_parameters"], second["action_parameters"]
            self.assertNotEqual(left[field], right[field])
            self.assertEqual({k: v for k, v in left.items() if k != field},
                             {k: v for k, v in right.items() if k != field})

    def test_dataset_is_balanced_for_each_action_and_label(self):
        counts = Counter((row["action_text"], row["label"]) for row in self.rows)
        for action in ("edit_file", "delete_file", "send_email", "transfer_money"):
            for label in ("reversible", "partially_reversible", "irreversible"):
                self.assertEqual(counts[(action, label)], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
