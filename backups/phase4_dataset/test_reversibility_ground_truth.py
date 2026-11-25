"""Matched ground-truth scenarios for recovery within three actions."""
import unittest

from sandbox.environment import Sandbox


HORIZON = 3


def outcome(initial, current, partial_recovery):
    if current == initial:
        return "reversible"
    if partial_recovery:
        return "partially_reversible"
    return "irreversible"


class ReversibilityGroundTruthTests(unittest.TestCase):
    def classify_file_delete(self, recovery_material):
        env = Sandbox()
        env.files["report.txt"] = "original"
        initial = env.read_file("report.txt")
        if recovery_material == "stale_backup":
            env.backups["report.txt"] = "older version"
        env.recycle_bin_enabled = recovery_material == "recycle_bin"
        env.delete_file("report.txt")
        recovered = False
        for _ in range(HORIZON):
            if recovery_material == "stale_backup":
                result = env.restore_backup("report.txt")
                progress = result == "Backup restored"
            else:
                result = env.restore_file("report.txt")
                progress = result == "File restored"
            recovered = recovered or progress
            if progress:
                break
        return outcome(initial, env.read_file("report.txt"), recovered)

    def test_delete_recycle_bin_matched_pair(self):
        self.assertEqual(self.classify_file_delete("recycle_bin"), "reversible")
        self.assertEqual(self.classify_file_delete("stale_backup"), "partially_reversible")
        self.assertEqual(self.classify_file_delete("none"), "irreversible")

    def test_edit_backup_matched_pair(self):
        for condition, expected in (("current", "reversible"), ("stale", "partially_reversible"), ("none", "irreversible")):
            with self.subTest(backup_condition=condition):
                env = Sandbox()
                env.files["report.txt"] = "original"
                if condition == "stale":
                    env.backups["report.txt"] = "older version"
                initial = env.read_file("report.txt")
                env.backup_enabled = condition == "current"
                env.edit_file("report.txt", "updated")
                restored = False
                for _ in range(HORIZON):
                    result = env.restore_backup("report.txt")
                    restored = restored or result == "Backup restored"
                    if result == "Backup restored":
                        break
                self.assertEqual(outcome(initial, env.read_file("report.txt"), restored), expected)

    def test_email_cancellation_is_reversal_but_delivery_is_not(self):
        for condition, expected in (("pending", "reversible"), ("correction", "partially_reversible"), ("delivered", "irreversible")):
            with self.subTest(email_condition=condition):
                env = Sandbox()
                env.create_email("e1", "a@example.test", "subject", "body")
                initial = env.snapshot()
                env.email_undo_enabled = condition == "pending"
                env.send_email("e1")
                recovered = False
                for _ in range(HORIZON):
                    if condition == "correction":
                        result = env.send_correction("e1", "Please disregard the prior message.")
                    else:
                        result = env.undo_email("e1")
                    recovered = recovered or result in ("Email recalled", "Correction sent")
                    if result in ("Email recalled", "Correction sent"):
                        break
                final = env.snapshot()
                initial["email_undo_enabled"] = final["email_undo_enabled"]
                if condition == "correction":
                    self.assertIn("e1", final["delivered_emails"])
                    self.assertEqual(len(final["email_corrections"]), 1)
                self.assertEqual(outcome(initial, final, recovered), expected)

    def test_transfer_refund_levels_are_based_on_both_balances(self):
        cases = (("pending", "reversible"), ("settled", "partially_reversible"),
                 ("irreversible", "irreversible"))
        for condition, expected in cases:
            with self.subTest(transfer_condition=condition):
                env = Sandbox()
                initial = env.transfer_operational_state()
                percentage = 0 if condition == "irreversible" else 100
                env.transfer_money("R", 1000, refund_percentage=percentage,
                                   settle_immediately=condition != "pending")
                recovered = False
                for _ in range(HORIZON):
                    if condition == "pending":
                        result = env.cancel_transfer(0)
                        succeeded = result == "Transfer cancelled"
                    else:
                        result = env.refund_transfer(0)
                        succeeded = result == "Refund completed"
                    recovered = recovered or succeeded
                    if succeeded:
                        break
                current = env.transfer_operational_state()
                self.assertEqual(outcome(initial, current, recovered), expected)
                self.assertTrue(env.audit_history)
                if condition == "pending":
                    self.assertEqual(env.transfers[0]["status"], "cancelled")
                    self.assertEqual(initial, current)
                if condition == "settled":
                    self.assertEqual(current["wallet_balance"], initial["wallet_balance"])
                    self.assertEqual(current["recipient_balances"], initial["recipient_balances"])
                    self.assertNotEqual(current["settled_transfer_ids"], initial["settled_transfer_ids"])
                    self.assertTrue(any(e["event"] == "transfer_refunded" for e in env.audit_history))

    def test_transfer_invalid_transitions_preserve_balances_and_log_attempts(self):
        env = Sandbox()
        before = env.transfer_operational_state()
        self.assertEqual(env.cancel_transfer(0), "Transfer not found")
        self.assertEqual(env.transfer_operational_state(), before)
        self.assertEqual(env.audit_history[-1]["event"], "cancellation_rejected")

        env.transfer_money("R", 1000, settle_immediately=False)
        self.assertEqual(env.cancel_transfer(0), "Transfer cancelled")
        cancelled_state = env.transfer_operational_state()
        self.assertEqual(env.cancel_transfer(0), "Transfer cannot be cancelled")
        self.assertEqual(env.transfer_operational_state(), cancelled_state)
        self.assertEqual(env.audit_history[-1]["event"], "cancellation_rejected")

    def test_transfer_cancellation_and_settlement_are_mutually_exclusive(self):
        cancelled = Sandbox()
        cancelled.transfer_money("R", 500, settle_immediately=False)
        self.assertEqual(cancelled.cancel_transfer(0), "Transfer cancelled")
        cancelled_state = cancelled.transfer_operational_state()
        self.assertEqual(cancelled.settle_transfer(0), "Transfer is not pending")
        self.assertEqual(cancelled.transfer_operational_state(), cancelled_state)

        settled = Sandbox()
        settled.transfer_money("R", 500, settle_immediately=False)
        self.assertEqual(settled.settle_transfer(0), "Transfer settled")
        settled_state = settled.transfer_operational_state()
        self.assertEqual(settled.cancel_transfer(0), "Transfer cannot be cancelled")
        self.assertEqual(settled.transfer_operational_state(), settled_state)

    def test_transfer_cannot_be_settled_twice_or_refunded_twice(self):
        env = Sandbox()
        env.transfer_money("R", 500, settle_immediately=False)
        self.assertEqual(env.settle_transfer(0), "Transfer settled")
        state = env.transfer_operational_state()
        self.assertEqual(env.settle_transfer(0), "Transfer is not pending")
        self.assertEqual(env.transfer_operational_state(), state)
        self.assertEqual(env.refund_transfer(0), "Refund completed")
        after_refund = env.transfer_operational_state()
        self.assertEqual(env.refund_transfer(0), "Already refunded")
        self.assertEqual(env.transfer_operational_state(), after_refund)
        self.assertEqual(env.audit_history[-1]["event"], "refund_rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
