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
        cases = ((100, "reversible"), (80, "partially_reversible"), (0, "irreversible"))
        for percentage, expected in cases:
            with self.subTest(refund_percentage=percentage):
                env = Sandbox()
                initial = (env.wallet_balance, env.recipient_balances.get("R", 0), env.transfers.copy())
                env.transfer_money("R", 1000, refund_percentage=percentage)
                recovered = False
                for _ in range(HORIZON):
                    result = env.refund_transfer(0)
                    recovered = recovered or result == "Refund completed"
                    if result != "Refund completed":
                        break
                current = (env.wallet_balance, env.recipient_balances.get("R", 0), env.transfers.copy())
                # Even a 100% refund leaves transfer and refund audit data; it is
                # economic compensation, not exact restoration of transaction state.
                if percentage == 100:
                    self.assertEqual(current[:2], initial[:2])
                    self.assertNotEqual(current[2], initial[2])
                    expected = "partially_reversible"
                self.assertEqual(outcome(initial, current, recovered), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
