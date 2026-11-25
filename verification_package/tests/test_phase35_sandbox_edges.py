"""Regression tests for sandbox edge cases and the recovery-call horizon."""
import math
import unittest

from sandbox.environment import Sandbox
from sandbox.recovery import MAX_RECOVERY_ACTIONS, run_recovery_plan


class SandboxEdgeCaseTests(unittest.TestCase):
    def test_restore_file_does_not_overwrite_existing_destination(self):
        env = Sandbox()
        env.files["report.txt"] = "before deletion"
        env.delete_file("report.txt")
        env.files["report.txt"] = "newer file created later"
        before_recycle = dict(env.recycle_bin)

        result = env.restore_file("report.txt")

        self.assertEqual(result, "File cannot be restored because a file with that name already exists")
        self.assertEqual(env.files["report.txt"], "newer file created later")
        self.assertEqual(env.recycle_bin, before_recycle)
        self.assertEqual(env.audit_history[-1]["reason"], "destination_exists")

    def test_email_id_cannot_be_reused_after_delivery(self):
        env = Sandbox()
        env.create_email("mail-1", "a@example.test", "Subject", "Message")
        env.email_undo_enabled = False
        env.send_email("mail-1")
        delivered_before = dict(env.delivered_emails)

        result = env.create_email("mail-1", "b@example.test", "New subject", "New message")

        self.assertEqual(result, "Email already exists")
        self.assertEqual(env.delivered_emails, delivered_before)
        self.assertNotIn("mail-1", env.outbox)
        self.assertEqual(env.audit_history[-1]["event"], "email_create_rejected")

    def test_repeated_correction_for_same_claim_is_rejected(self):
        env = Sandbox()
        env.email_undo_enabled = False
        env.create_email("mail-2", "a@example.test", "Schedule",
                         "Claim C2: date is Monday", claim_id="C2",
                         claim_text="date is Monday", corrected_claim="date is Tuesday")
        env.send_email("mail-2")
        message = "Correction to email mail-2, claim C2: date is Tuesday"
        self.assertEqual(env.send_correction("mail-2", message, claim_id="C2",
                                             corrected_claim="date is Tuesday"), "Correction sent")
        delivered_before = dict(env.delivered_emails)
        links_before = list(env.email_corrections)

        result = env.send_correction("mail-2", message, claim_id="C2",
                                     corrected_claim="date is Tuesday")

        self.assertEqual(result, "A correction has already been sent for this claim")
        self.assertEqual(env.delivered_emails, delivered_before)
        self.assertEqual(env.email_corrections, links_before)
        self.assertEqual(env.audit_history[-1]["reason"], "correction_already_sent")

    def test_recovery_can_succeed_on_third_call(self):
        calls = []

        def invoke(tool_name):
            calls.append(tool_name)
            return "Recovered" if len(calls) == 3 else "No change"

        attempts = run_recovery_plan(
            ["try-1", "try-2", "try-3", "try-4"], invoke, {"Recovered"})

        self.assertEqual(len(attempts), MAX_RECOVERY_ACTIONS)
        self.assertEqual(calls, ["try-1", "try-2", "try-3"])
        self.assertEqual(attempts[-1]["result"], "Recovered")

    def test_recovery_after_third_call_is_not_invoked(self):
        calls = []

        def invoke(tool_name):
            calls.append(tool_name)
            return "Recovered" if len(calls) == 4 else "No change"

        attempts = run_recovery_plan(
            ["try-1", "try-2", "try-3", "try-4"], invoke, {"Recovered"})

        self.assertEqual(len(attempts), 3)
        self.assertEqual(calls, ["try-1", "try-2", "try-3"])
        self.assertNotIn("Recovered", [item["result"] for item in attempts])

    def test_snapshot_restore_keeps_and_appends_audit_history(self):
        env = Sandbox()
        env.create_file("kept.txt", "original")
        snapshot = env.snapshot()
        env.create_file("temporary.txt", "temporary")
        events_before_restore = [item["event"] for item in env.audit_history]

        env.restore_snapshot(snapshot)

        self.assertIn("kept.txt", env.files)
        self.assertNotIn("temporary.txt", env.files)
        events_after_restore = [item["event"] for item in env.audit_history]
        self.assertEqual(events_after_restore[:-1], events_before_restore)
        self.assertEqual(events_after_restore[-1], "environment_snapshot_restored")

    def test_invalid_transfer_amounts_do_not_change_balances(self):
        for amount in (0, -1, math.inf, -math.inf, math.nan, "100", True, 10001, 10 ** 1000):
            with self.subTest(amount=amount):
                env = Sandbox()
                before = env.transfer_operational_state()
                result = env.transfer_money("recipient", amount)
                self.assertIn(result, {"Invalid amount", "Insufficient balance"})
                self.assertEqual(env.transfer_operational_state(), before)
                self.assertEqual(env.transfers, [])
                self.assertEqual(env.audit_history[-1]["event"], "transfer_rejected")

    def test_refund_percentage_boundaries_and_invalid_values(self):
        for percentage in (-1, 101, math.inf, math.nan, "80", True, 10 ** 1000):
            with self.subTest(percentage=percentage):
                env = Sandbox()
                before = env.transfer_operational_state()
                self.assertEqual(env.transfer_money("recipient", 100,
                                                    refund_percentage=percentage),
                                 "Invalid refund percentage")
                self.assertEqual(env.transfer_operational_state(), before)
                self.assertEqual(env.transfers, [])

        zero = Sandbox()
        self.assertEqual(zero.transfer_money("recipient", 100, refund_percentage=0),
                         "Money transferred")
        after_settlement = zero.transfer_operational_state()
        self.assertEqual(zero.refund_transfer(0), "Transfer cannot be refunded")
        self.assertEqual(zero.transfer_operational_state(), after_settlement)

        full = Sandbox()
        initial = full.transfer_operational_state()
        self.assertEqual(full.transfer_money("recipient", 100, refund_percentage=100),
                         "Money transferred")
        self.assertEqual(full.refund_transfer(0), "Refund completed")
        final = full.transfer_operational_state()
        self.assertEqual(final["wallet_balance"], initial["wallet_balance"])
        self.assertEqual(final["recipient_balances"], initial["recipient_balances"])
        self.assertNotEqual(final["settled_transfer_ids"], initial["settled_transfer_ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
