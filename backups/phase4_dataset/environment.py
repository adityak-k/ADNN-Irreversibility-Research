
import copy


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

    def create_file(self, name, content):
        if name in self.files:
            return "File already exists"

        self.files[name] = content
        return "File created"

    def read_file(self, name):
        if name not in self.files:
            return "File not found"

        return self.files[name]

    def edit_file(self, name, new_content):
        if name not in self.files:
            return "File not found"

        if self.backup_enabled:
            self.backups[name] = self.files[name]

        self.files[name] = new_content
        return "File edited"

    def delete_file(self, name):
        if name not in self.files:
            return "File not found"

        if self.recycle_bin_enabled:
            self.recycle_bin[name] = self.files[name]

        del self.files[name]
        return "File deleted"

    def restore_file(self, name):
        if name not in self.recycle_bin:
            return "File cannot be restored"

        self.files[name] = self.recycle_bin[name]
        del self.recycle_bin[name]
        return "File restored"

    def restore_backup(self, name):
        if name not in self.backups:
            return "Backup not found"

        self.files[name] = self.backups[name]
        del self.backups[name]
        return "Backup restored"

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore_snapshot(self, saved_state):
        self.__dict__ = copy.deepcopy(saved_state)

    def create_email(self, email_id, recipient, subject, message):
        if email_id in self.outbox:
            return "Email already exists"

        self.outbox[email_id] = {
            "recipient": recipient,
            "subject": subject,
            "message": message
        }

        return "Email draft created"

    def send_email(self, email_id):
        if email_id not in self.outbox:
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

        return "Email sent"

    def undo_email(self, email_id):
        if email_id in self.delivered_emails:
            return "Email cannot be recalled"

        if email_id not in self.sent_emails:
            return "Sent email not found"

        if not self.sent_emails[email_id]["undo_available"]:
            return "Email cannot be recalled"

        email = self.sent_emails[email_id]["email"]

        self.outbox[email_id] = copy.deepcopy(email)
        del self.sent_emails[email_id]

        return "Email recalled"

    def send_correction(self, email_id, correction_message):
        """Record a follow-up correction without pretending to retract delivery."""
        if email_id not in self.delivered_emails:
            return "Delivered email not found"
        if not self.email_correction_enabled:
            return "Correction channel unavailable"
        if not correction_message:
            return "Correction message is required"
        self.email_corrections.append({
            "email_id": email_id,
            "message": correction_message
        })
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
        if amount <= 0:
            self._audit_transfer("transfer_rejected", reason="invalid_amount", amount=amount)
            return "Invalid amount"
        if amount > self.wallet_balance:
            self._audit_transfer("transfer_rejected", reason="insufficient_balance", amount=amount)
            return "Insufficient balance"
        if refund_percentage < 0 or refund_percentage > 100:
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
