
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

        self.wallet_balance = 10000
        self.recipient_balances = {}
        self.transfers = []

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



    def transfer_money(self, recipient, amount, refundable=True, refund_percentage=80):
        if amount <= 0:
            return "Invalid amount"

        if amount > self.wallet_balance:
            return "Insufficient balance"

        if refund_percentage < 0 or refund_percentage > 100:
            return "Invalid refund percentage"

        if not refundable:
            refund_percentage = 0

        self.wallet_balance -= amount
        self.recipient_balances[recipient] = self.recipient_balances.get(recipient, 0) + amount

        transfer = {
            "recipient": recipient,
            "amount": amount,
            "refundable": refundable,
            "refund_percentage": refund_percentage,
            "refunded": False
        }

        self.transfers.append(transfer)

        return "Money transferred"

    def refund_transfer(self, transfer_number):
        if transfer_number < 0 or transfer_number >= len(self.transfers):
            return "Transfer not found"

        transfer = self.transfers[transfer_number]

        if transfer["refund_percentage"] == 0:
            return "Transfer cannot be refunded"

        if transfer["refunded"]:
            return "Already refunded"

        refund_amount = transfer["amount"] * transfer["refund_percentage"] / 100

        self.wallet_balance += refund_amount
        self.recipient_balances[transfer["recipient"]] -= refund_amount
        transfer["refunded"] = True
        transfer["refund_amount"] = refund_amount

        return "Refund completed"
