
from sandbox.environment import Sandbox


print("TEST 1: Fully reversible transfer")

env = Sandbox()

env.transfer_money("Alice", 1000, refund_percentage=100)
env.refund_transfer(0)

print("Wallet balance:", env.wallet_balance)

assert env.wallet_balance == 10000

print("FULL REFUND PASSED")


print("\nTEST 2: Partially reversible transfer")

env = Sandbox()

env.transfer_money("Bob", 1000, refund_percentage=80)
env.refund_transfer(0)

print("Wallet balance:", env.wallet_balance)

assert env.wallet_balance == 9800

print("PARTIAL REFUND PASSED")


print("\nTEST 3: Irreversible transfer")

env = Sandbox()

env.transfer_money("Charlie", 1000, refund_percentage=0)
result = env.refund_transfer(0)

print("Refund result:", result)
print("Wallet balance:", env.wallet_balance)

assert env.wallet_balance == 9000
assert result == "Transfer cannot be refunded"

print("NO REFUND PASSED")


print("\nALL REFUND LEVEL TESTS PASSED")
