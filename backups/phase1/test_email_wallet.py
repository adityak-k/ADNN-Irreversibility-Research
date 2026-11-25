
from sandbox.environment import Sandbox


env = Sandbox()


print("TEST 1: Creating an email")

result = env.create_email(
    "email1",
    "student@example.com",
    "Project Report",
    "Please find the report attached"
)

print(result)
assert "email1" in env.outbox


print("\nTEST 2: Sending an email with undo enabled")

result = env.send_email("email1")
print(result)

assert "email1" in env.sent_emails
assert "email1" not in env.outbox


print("\nTEST 3: Recalling an email")

result = env.undo_email("email1")
print(result)

assert "email1" in env.outbox
assert "email1" not in env.sent_emails


print("\nTEST 4: Sending an email without undo")

env.email_undo_enabled = False

env.send_email("email1")

result = env.undo_email("email1")
print(result)

assert result == "Email cannot be recalled"


print("\nTEST 5: Making a refundable money transfer")

env = Sandbox()

result = env.transfer_money("Person A", 1000, refundable=True)
print(result)

assert env.wallet_balance == 9000


print("\nTEST 6: Receiving a partial refund")

result = env.refund_transfer(0)
print(result)

print("Wallet balance:", env.wallet_balance)

assert env.wallet_balance == 9800
assert env.wallet_balance != 10000


print("\nTEST 7: Making a non-refundable transfer")

result = env.transfer_money("Person B", 500, refundable=False)
print(result)

assert env.wallet_balance == 9300


print("\nTEST 8: Trying to refund a non-refundable transfer")

result = env.refund_transfer(1)
print(result)

assert result == "Transfer cannot be refunded"


print("\nALL EMAIL AND WALLET TESTS PASSED")
