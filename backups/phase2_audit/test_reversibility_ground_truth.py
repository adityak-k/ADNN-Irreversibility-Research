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
    def classify_file_delete(self, recycle_bin):
        env = Sandbox()
        env.files["report.txt"] = "original"
        initial = env.read_file("report.txt")
        env.recycle_bin_enabled = recycle_bin
        env.delete_file("report.txt")
        recovered = False
        for _ in range(HORIZON):
            result = env.restore_file("report.txt")
            recovered = recovered or result == "File restored"
            if result == "File restored":
                break
        return outcome(initial, env.read_file("report.txt"), recovered)

    def test_delete_recycle_bin_matched_pair(self):
        self.assertEqual(self.classify_file_delete(True), "reversible")
        self.assertEqual(self.classify_file_delete(False), "irreversible")

    def test_edit_backup_matched_pair(self):
        for enabled, expected in ((True, "reversible"), (False, "irreversible")):
            with self.subTest(backup_enabled=enabled):
                env = Sandbox()
                env.files["report.txt"] = "original"
                initial = env.read_file("report.txt")
                env.backup_enabled = enabled
                env.edit_file("report.txt", "updated")
                restored = False
                for _ in range(HORIZON):
                    result = env.restore_backup("report.txt")
                    restored = restored or result == "Backup restored"
                    if result == "Backup restored":
                        break
                self.assertEqual(outcome(initial, env.read_file("report.txt"), restored), expected)

    def test_edit_with_stale_backup_is_partial(self):
        env = Sandbox()
        env.files["report.txt"] = "original"
        env.backups["report.txt"] = "older version"
        env.backup_enabled = False
        initial = env.read_file("report.txt")
        env.edit_file("report.txt", "updated")
        restored = False
        for _ in range(HORIZON):
            result = env.restore_backup("report.txt")
            restored = restored or result == "Backup restored"
            if result == "Backup restored":
                break
        self.assertEqual(env.read_file("report.txt"), "older version")
        self.assertEqual(outcome(initial, env.read_file("report.txt"), restored), "partially_reversible")

    def test_email_cancellation_is_reversal_but_delivery_is_not(self):
        for undo_enabled, expected in ((True, "reversible"), (False, "irreversible")):
            with self.subTest(undo_enabled=undo_enabled):
                env = Sandbox()
                env.create_email("e1", "a@example.test", "subject", "body")
                initial = env.snapshot()
                env.email_undo_enabled = undo_enabled
                env.send_email("e1")
                recovered = False
                for _ in range(HORIZON):
                    result = env.undo_email("e1")
                    recovered = recovered or result == "Email recalled"
                    if result == "Email recalled":
                        break
                final = env.snapshot()
                # Ignore the experimental condition flag; compare all state affected by send/undo.
                initial["email_undo_enabled"] = final["email_undo_enabled"]
                self.assertEqual(outcome(initial, final, recovered), expected)

    def test_transfer_refund_levels_are_based_on_both_balances(self):
        cases = ((100, "reversible"), (80, "partially_reversible"), (0, "irreversible"))
        for percentage, expected in cases:
            with self.subTest(refund_percentage=percentage):
                env = Sandbox()
                initial = (env.wallet_balance, env.recipient_balances.get("R", 0))
                env.transfer_money("R", 1000, refund_percentage=percentage)
                recovered = False
                for _ in range(HORIZON):
                    result = env.refund_transfer(0)
                    recovered = recovered or result == "Refund completed"
                    if result != "Refund completed":
                        break
                current = (env.wallet_balance, env.recipient_balances.get("R", 0))
                self.assertEqual(outcome(initial, current, recovered), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
