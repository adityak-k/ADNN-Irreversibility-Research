
from sandbox.environment import Sandbox


env = Sandbox()

print("TEST 1: Creating a file")

result = env.create_file("report.txt", "Original report")
print(result)
assert env.read_file("report.txt") == "Original report"


print("\nTEST 2: Editing a file")

result = env.edit_file("report.txt", "Updated report")
print(result)
assert env.read_file("report.txt") == "Updated report"


print("\nTEST 3: Restoring a backup")

result = env.restore_backup("report.txt")
print(result)
assert env.read_file("report.txt") == "Original report"


print("\nTEST 4: Deleting a file with recycle bin enabled")

result = env.delete_file("report.txt")
print(result)
assert env.read_file("report.txt") == "File not found"


print("\nTEST 5: Restoring a deleted file")

result = env.restore_file("report.txt")
print(result)
assert env.read_file("report.txt") == "Original report"


print("\nTEST 6: Deleting a file without recycle bin")

env.recycle_bin_enabled = False
result = env.delete_file("report.txt")
print(result)

assert env.restore_file("report.txt") == "File cannot be restored"


print("\nTEST 7: Saving and restoring environment state")

saved_state = env.snapshot()

env.create_file("temporary.txt", "Temporary data")
env.restore_snapshot(saved_state)

assert env.read_file("temporary.txt") == "File not found"

print("Environment state restored successfully")


print("\nALL TESTS PASSED")
