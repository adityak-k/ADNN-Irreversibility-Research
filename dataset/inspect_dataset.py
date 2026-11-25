
import json
from collections import Counter, defaultdict


examples = []

with open("dataset/irreversibility_dataset.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))


print("DATASET INSPECTION")
print("------------------")

print("Total examples:", len(examples))

labels = Counter()
actions = Counter()
pairs = defaultdict(list)
ids = set()

errors = []

for example in examples:

    labels[example["label"]] += 1
    actions[example["action_text"]] += 1

    pairs[example["pair_id"]].append(example)

    if example["id"] in ids:
        errors.append("Duplicate ID: " + example["id"])

    ids.add(example["id"])


print("\nCLASS DISTRIBUTION")

for label, count in labels.items():
    print(label, ":", count)


print("\nACTION DISTRIBUTION")

for action, count in actions.items():
    print(action, ":", count)


print("\nPAIR VALIDATION")

valid_pairs = 0

for pair_id, pair in pairs.items():

    if len(pair) != 2:
        errors.append(pair_id + " does not contain two examples")
        continue

    if pair[0]["action_text"] != pair[1]["action_text"]:
        errors.append(pair_id + " has different actions")

    if pair[0]["label"] == pair[1]["label"]:
        errors.append(pair_id + " has identical labels")

    if pair[0]["environment"] != pair[1]["environment"]:
        errors.append(pair_id + " has different environments")

    valid_pairs += 1


print("Total pairs:", len(pairs))
print("Valid pairs:", valid_pairs)
print("Unique IDs:", len(ids))


print("\nVALIDATION RESULT")

if len(examples) == 2000 and len(ids) == 2000 and not errors:
    print("STRUCTURAL CHECK PASSED")
else:
    print("STRUCTURAL CHECK FAILED")

    for error in errors:
        print(error)


print("\nSAMPLE EXAMPLES")

for example in examples[:3]:
    print(json.dumps(example, indent=2))
