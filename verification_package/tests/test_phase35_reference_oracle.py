"""Tests for the reference outcome calculations independent of Sandbox."""
import copy
import json
import unittest
from pathlib import Path

from dataset.reference_oracle import validate_outcome_record
from sandbox.environment import Sandbox


V3_DATASET = Path(__file__).resolve().parents[1] / "dataset" / "irreversibility_dataset_v3.jsonl"


class ReferenceOracleTests(unittest.TestCase):
    def _run_file(self, action, condition):
        env = Sandbox()
        params = {
            "file_name": "report.txt", "original_content": "code=A; qty=8",
            "new_content": "code=A; qty=9", "stale_content": "code=A; qty=7",
            "shared_original_unit": "code=A", "current_only_unit": "qty=8",
            "old_only_unit": "qty=7", "recovery_condition": condition,
        }
        env.files[params["file_name"]] = params["original_content"]
        if action == "edit_file":
            env.backup_enabled = condition == "current"
            if condition == "stale":
                env.backups[params["file_name"]] = params["stale_content"]
            initial = {"exists": True, "content": env.files[params["file_name"]]}
            env.edit_file(params["file_name"], params["new_content"])
            post = {"exists": True, "content": env.files[params["file_name"]]}
            if condition in {"current", "stale", "none"}:
                env.restore_backup(params["file_name"])
        else:
            env.recycle_bin_enabled = condition == "recycle"
            if condition == "stale_backup":
                env.backups[params["file_name"]] = params["stale_content"]
            initial = {"exists": True, "content": env.files[params["file_name"]]}
            env.delete_file(params["file_name"])
            post = {"exists": False, "content": None}
            if condition == "recycle":
                env.restore_file(params["file_name"])
            elif condition == "stale_backup":
                env.restore_backup(params["file_name"])
            else:
                env.restore_file(params["file_name"])
                env.restore_backup(params["file_name"])
        final = {"exists": params["file_name"] in env.files,
                 "content": env.files.get(params["file_name"])}
        return {"action_text": action, "action_parameters": params,
                "initial_operational_state": initial,
                "post_action_operational_state": post,
                "final_operational_state": final}

    def _run_email(self, condition):
        env = Sandbox()
        params = {
            "email_id": "mail-1", "claim_id": "claim-1",
            "incorrect_claim": "date is Monday", "corrected_claim": "date is Tuesday",
            "recipient": "reader@example.test", "subject": "Schedule",
            "message": "Claim claim-1: date is Monday",
            "correction_message": "Correction to email mail-1, claim claim-1: date is Tuesday",
            "recovery_condition": condition,
        }
        env.email_undo_enabled = condition == "recall"
        env.email_correction_enabled = condition == "correction"
        env.create_email(params["email_id"], params["recipient"], params["subject"],
                         params["message"], claim_id=params["claim_id"],
                         claim_text=params["incorrect_claim"],
                         corrected_claim=params["corrected_claim"])
        initial = {"outbox": copy.deepcopy(env.outbox.get(params["email_id"]),),
                   "pending_send": None, "delivered_original": None,
                   "delivered_corrections": {}, "correction_links": []}
        env.send_email(params["email_id"])
        post = self._email_state(env, params["email_id"])
        if condition == "recall":
            env.undo_email(params["email_id"])
        elif condition == "correction":
            env.send_correction(params["email_id"], params["correction_message"],
                                claim_id=params["claim_id"],
                                corrected_claim=params["corrected_claim"])
        final = self._email_state(env, params["email_id"])
        return {"action_text": "send_email", "action_parameters": params,
                "initial_operational_state": initial,
                "post_action_operational_state": post,
                "final_operational_state": final}

    @staticmethod
    def _email_state(env, email_id):
        links = [copy.deepcopy(item) for item in env.email_corrections
                 if item["original_email_id"] == email_id]
        correction_ids = {item["correction_email_id"] for item in links}
        return {
            "outbox": copy.deepcopy(env.outbox.get(email_id)),
            "pending_send": copy.deepcopy(env.sent_emails.get(email_id)),
            "delivered_original": copy.deepcopy(env.delivered_emails.get(email_id)),
            "delivered_corrections": {
                key: copy.deepcopy(env.delivered_emails.get(key)) for key in sorted(correction_ids)},
            "correction_links": links,
        }

    def _run_transfer(self, condition):
        env = Sandbox()
        params = {"recipient": "supplier", "amount": 100,
                  "wallet_balance": 1000, "recipient_balance": 40,
                  "refund_percentage": 100, "transfer_condition": condition}
        env.wallet_balance = params["wallet_balance"]
        env.recipient_balances[params["recipient"]] = params["recipient_balance"]
        initial = env.transfer_operational_state()
        env.transfer_money(params["recipient"], params["amount"],
                           refundable=condition != "settled_no_refund",
                           refund_percentage=params["refund_percentage"],
                           settle_immediately=condition != "pending")
        post = env.transfer_operational_state()
        if condition == "pending":
            env.cancel_transfer(0)
        elif condition == "settled_refundable":
            env.refund_transfer(0)
        else:
            env.refund_transfer(0)
            env.cancel_transfer(0)
        final = env.transfer_operational_state()
        for state in (initial, post, final):
            state["settled_transfer_ids"] = sorted(state["settled_transfer_ids"])
        return {"action_text": "transfer_money", "action_parameters": params,
                "initial_operational_state": initial,
                "post_action_operational_state": post,
                "final_operational_state": final}

    def test_reference_oracle_accepts_actual_action_outcomes(self):
        records = [
            self._run_file("edit_file", "current"),
            self._run_file("delete_file", "stale_backup"),
            self._run_email("correction"),
            self._run_transfer("settled_refundable"),
        ]
        records[0]["label"] = "reversible"
        records[1]["label"] = "partially_reversible"
        records[2]["label"] = "partially_reversible"
        records[3]["label"] = "partially_reversible"
        for record in records:
            with self.subTest(action=record["action_text"]):
                result = validate_outcome_record(record)
                self.assertTrue(result["passed"], result["errors"])

    def test_reference_oracle_rejects_incorrect_final_state(self):
        record = self._run_file("edit_file", "current")
        record["final_operational_state"]["content"] = "invented value"
        result = validate_outcome_record(record)
        self.assertFalse(result["passed"])
        self.assertIn("final_operational_state disagrees with independent reference outcome",
                      result["errors"])

    def test_reference_oracle_rejects_incorrect_expected_label(self):
        record = self._run_transfer("settled_refundable")
        record["label"] = "reversible"
        result = validate_outcome_record(record)
        self.assertFalse(result["passed"])
        self.assertIn("label disagrees with independent reference outcome", result["errors"])

    def test_reference_oracle_agrees_with_every_frozen_v3_outcome(self):
        with V3_DATASET.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(records), 2400)
        errors = []
        for record in records:
            result = validate_outcome_record(record)
            if not result["passed"]:
                errors.append(f"{record['id']}: {result['errors']}")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
