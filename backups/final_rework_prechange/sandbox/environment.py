
import copy
import math


class Sandbox:

    def __init__(self):
        self.files = {}
        self.recycle_bin = {}
        self.backups = {}

        self.outbox = {}
        self.sent_emails = {}
        # Emails sent while the undo window is enabled are pending delivery.
        # A successful undo cancels delivery; sent emails without that window
        # are recorded permanently in delivered_emails.
        self.delivered_emails = {}
        self.email_corrections = []
        self.email_correction_enabled = True

        self.wallet_balance = 10000
        self.recipient_balances = {}
        # Mutable lifecycle controls are separate from the append-only audit.
        self.transfers = []
        self.audit_history = []
        # A settlement commitment is an operational effect a refund cannot erase.
        self.settled_transfer_ids = set()

        self.recycle_bin_enabled = True
        self.backup_enabled = True
        self.email_undo_enabled = True

    def _audit_action(self, event, **details):
        entry = {"event": event}
        entry.update(copy.deepcopy(details))
        self.audit_history.append(entry)

    def create_file(self, name, content):
        if name in self.files:
            self._audit_action("file_create_rejected", name=name, reason="already_exists")
            return "File already exists"

        self.files[name] = content
        self._audit_action("file_created", name=name)
        return "File created"

    def read_file(self, name):
        if name not in self.files:
            return "File not found"

        return self.files[name]

    def edit_file(self, name, new_content):
        if name not in self.files:
            self._audit_action("file_edit_rejected", name=name, reason="not_found")
            return "File not found"

        if self.backup_enabled:
            self.backups[name] = self.files[name]

        self.files[name] = new_content
        self._audit_action("file_edited", name=name)
        return "File edited"

    def delete_file(self, name):
        if name not in self.files:
            self._audit_action("file_delete_rejected", name=name, reason="not_found")
            return "File not found"

        if self.recycle_bin_enabled:
            self.recycle_bin[name] = self.files[name]

        del self.files[name]
        self._audit_action("file_deleted", name=name, recycle_bin=self.recycle_bin_enabled)
        return "File deleted"

    def restore_file(self, name):
        if name not in self.recycle_bin:
            self._audit_action("file_restore_rejected", name=name, reason="not_in_recycle_bin")
            return "File cannot be restored"
        if name in self.files:
            self._audit_action("file_restore_rejected", name=name, reason="destination_exists")
            return "File cannot be restored because a file with that name already exists"

        self.files[name] = self.recycle_bin[name]
        del self.recycle_bin[name]
        self._audit_action("file_restored", name=name)
        return "File restored"

    def restore_backup(self, name):
        if name not in self.backups:
            self._audit_action("backup_restore_rejected", name=name, reason="not_found")
            return "Backup not found"

        self.files[name] = self.backups[name]
        del self.backups[name]
        self._audit_action("backup_restored", name=name)
        return "Backup restored"

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore_snapshot(self, saved_state):
        # Snapshots restore operational state, but audit history is append-only.
        audit_history = copy.deepcopy(self.audit_history)
        self.__dict__ = copy.deepcopy(saved_state)
        self.audit_history = audit_history
        self._audit_action("environment_snapshot_restored")

    def create_email(self, email_id, recipient, subject, message, *,
                     claim_id=None, claim_text=None, corrected_claim=None):
        if (email_id in self.outbox or email_id in self.sent_emails
                or email_id in self.delivered_emails):
            self._audit_action("email_create_rejected", email_id=email_id, reason="already_exists")
            return "Email already exists"

        email = {
            "recipient": recipient,
            "subject": subject,
            "message": message
        }
        if claim_id is not None:
            email.update({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "corrected_claim": corrected_claim,
            })
        self.outbox[email_id] = email

        self._audit_action("email_draft_created", email_id=email_id, recipient=recipient)
        return "Email draft created"

    def send_email(self, email_id):
        if email_id not in self.outbox:
            self._audit_action("email_send_rejected", email_id=email_id, reason="draft_not_found")
            return "Email not found"

        email = self.outbox[email_id]

        del self.outbox[email_id]
        if self.email_undo_enabled:
            self.sent_emails[email_id] = {
                "email": copy.deepcopy(email),
                "undo_available": True
            }
        else:
            self.delivered_emails[email_id] = copy.deepcopy(email)

        self._audit_action("email_sent", email_id=email_id,
                           pending_delivery=self.email_undo_enabled)
        return "Email sent"

    def undo_email(self, email_id):
        if email_id in self.delivered_emails:
            self._audit_action("email_recall_rejected", email_id=email_id, reason="already_delivered")
            return "Email cannot be recalled"

        if email_id not in self.sent_emails:
            self._audit_action("email_recall_rejected", email_id=email_id, reason="not_found")
            return "Sent email not found"

        if not self.sent_emails[email_id]["undo_available"]:
            self._audit_action("email_recall_rejected", email_id=email_id, reason="undo_unavailable")
            return "Email cannot be recalled"

        email = self.sent_emails[email_id]["email"]

        self.outbox[email_id] = copy.deepcopy(email)
        del self.sent_emails[email_id]

        self._audit_action("email_recalled", email_id=email_id)
        return "Email recalled"

    def send_correction(self, email_id, correction_message, *, claim_id=None,
                        corrected_claim=None):
        """Deliver a correction tied to one claim in the original delivered email."""
        if email_id not in self.delivered_emails:
            self._audit_action("email_correction_rejected", email_id=email_id,
                               reason="original_not_delivered")
            return "Delivered email not found"
        if not self.email_correction_enabled:
            self._audit_action("email_correction_rejected", email_id=email_id,
                               reason="channel_unavailable")
            return "Correction channel unavailable"
        if any(link["original_email_id"] == email_id and link["claim_id"] == claim_id
               for link in self.email_corrections):
            self._audit_action("email_correction_rejected", email_id=email_id,
                               reason="correction_already_sent")
            return "A correction has already been sent for this claim"
        if not correction_message:
            self._audit_action("email_correction_rejected", email_id=email_id,
                               reason="empty_message")
            return "Correction message is required"
        original = self.delivered_emails[email_id]
        if (not original.get("claim_id") or claim_id != original.get("claim_id")
                or not original.get("corrected_claim")
                or corrected_claim != original.get("corrected_claim")
                or corrected_claim == original.get("claim_text")
                or email_id not in correction_message
                or claim_id not in correction_message
                or corrected_claim not in correction_message):
            self._audit_action("email_correction_rejected", email_id=email_id,
                               reason="claim_link_mismatch")
            return "Correction does not match an identifiable claim"

        correction_id = f"{email_id}:correction:{len(self.email_corrections) + 1}"
        correction = {
            "original_email_id": email_id,
            "claim_id": claim_id,
            "recipient": original["recipient"],
            "subject": f"Correction: {original['subject']}",
            "message": correction_message,
            "corrected_claim": corrected_claim,
        }
        self.delivered_emails[correction_id] = correction
        self.email_corrections.append({
            "original_email_id": email_id,
            "correction_email_id": correction_id,
            "claim_id": claim_id,
            "original_claim": original["claim_text"],
            "corrected_claim": corrected_claim,
        })
        self._audit_action("email_correction_sent", email_id=email_id)
        return "Correction sent"



    def _audit_transfer(self, event, transfer_number=None, **details):
        entry = {"event": event}
        if transfer_number is not None:
            entry["transfer_number"] = transfer_number
        entry.update(copy.deepcopy(details))
        self.audit_history.append(entry)

    def transfer_operational_state(self):
        """Snapshot economic effects; audit records are deliberately separate."""
        return {
            "wallet_balance": self.wallet_balance,
            "recipient_balances": {
                recipient: balance for recipient, balance in self.recipient_balances.items()
                if balance != 0
            },
            "settled_transfer_ids": frozenset(self.settled_transfer_ids),
        }

    def transfer_money(self, recipient, amount, refundable=True, refund_percentage=80,
                       settle_immediately=True):
        try:
            valid_amount = (isinstance(amount, (int, float)) and not isinstance(amount, bool)
                            and math.isfinite(amount))
        except (OverflowError, TypeError):
            valid_amount = False
        if not valid_amount or amount <= 0:
            self._audit_transfer("transfer_rejected", reason="invalid_amount", amount=amount)
            return "Invalid amount"
        if amount > self.wallet_balance:
            self._audit_transfer("transfer_rejected", reason="insufficient_balance", amount=amount)
            return "Insufficient balance"
        try:
            valid_percentage = (
                isinstance(refund_percentage, (int, float))
                and not isinstance(refund_percentage, bool)
                and math.isfinite(refund_percentage)
            )
        except (OverflowError, TypeError):
            valid_percentage = False
        if not valid_percentage or refund_percentage < 0 or refund_percentage > 100:
            self._audit_transfer("transfer_rejected", reason="invalid_refund_percentage",
                                 refund_percentage=refund_percentage)
            return "Invalid refund percentage"
        if not refundable:
            refund_percentage = 0

        # Policy: pending transfers reserve funds by debiting the sender;
        # recipient funds are credited only at settlement.
        self.wallet_balance -= amount
        transfer = {
            "recipient": recipient,
            "amount": amount,
            "refundable": refundable,
            "refund_percentage": refund_percentage,
            "status": "pending",
            "refunded": False,
            "refund_amount": 0,
        }
        self.transfers.append(transfer)
        transfer_number = len(self.transfers) - 1
        self._audit_transfer("transfer_created", transfer_number,
                             recipient=recipient, amount=amount, reserved=True)
        if settle_immediately:
            self.settle_transfer(transfer_number)
        return "Money transferred"

    def settle_transfer(self, transfer_number):
        if transfer_number < 0 or transfer_number >= len(self.transfers):
            self._audit_transfer("settlement_rejected", transfer_number,
                                 reason="transfer_not_found")
            return "Transfer not found"
        transfer = self.transfers[transfer_number]
        if transfer["status"] != "pending":
            self._audit_transfer("settlement_rejected", transfer_number,
                                 reason="transfer_not_pending", status=transfer["status"])
            return "Transfer is not pending"
        recipient = transfer["recipient"]
        self.recipient_balances[recipient] = self.recipient_balances.get(recipient, 0) + transfer["amount"]
        transfer["status"] = "settled"
        self.settled_transfer_ids.add(transfer_number)
        self._audit_transfer("transfer_settled", transfer_number,
                             recipient=recipient, amount=transfer["amount"])
        return "Transfer settled"

    def cancel_transfer(self, transfer_number):
        if transfer_number < 0 or transfer_number >= len(self.transfers):
            self._audit_transfer("cancellation_rejected", transfer_number,
                                 reason="transfer_not_found")
            return "Transfer not found"
        transfer = self.transfers[transfer_number]
        if transfer["status"] != "pending":
            self._audit_transfer("cancellation_rejected", transfer_number,
                                 reason="transfer_not_pending", status=transfer["status"])
            return "Transfer cannot be cancelled"
        self.wallet_balance += transfer["amount"]
        transfer["status"] = "cancelled"
        self._audit_transfer("transfer_cancelled", transfer_number,
                             amount=transfer["amount"])
        return "Transfer cancelled"

    def refund_transfer(self, transfer_number):
        if transfer_number < 0 or transfer_number >= len(self.transfers):
            self._audit_transfer("refund_rejected", transfer_number,
                                 reason="transfer_not_found")
            return "Transfer not found"
        transfer = self.transfers[transfer_number]
        if transfer["refunded"]:
            self._audit_transfer("refund_rejected", transfer_number,
                                 reason="already_refunded")
            return "Already refunded"
        if transfer["status"] != "settled":
            self._audit_transfer("refund_rejected", transfer_number,
                                 reason="transfer_not_settled", status=transfer["status"])
            return "Transfer is not refundable in its current state"
        if transfer["refund_percentage"] == 0:
            self._audit_transfer("refund_rejected", transfer_number,
                                 reason="refund_unavailable")
            return "Transfer cannot be refunded"
        refund_amount = transfer["amount"] * transfer["refund_percentage"] / 100
        self.wallet_balance += refund_amount
        self.recipient_balances[transfer["recipient"]] -= refund_amount
        transfer["refunded"] = True
        transfer["refund_amount"] = refund_amount
        transfer["status"] = "refunded"
        self._audit_transfer("transfer_refunded", transfer_number,
                             amount=refund_amount, percentage=transfer["refund_percentage"])
        return "Refund completed"
