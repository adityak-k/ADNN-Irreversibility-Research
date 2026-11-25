
import json
import random


random.seed(42)

examples = []

scenarios = [
    {
        "action": "delete_file",
        "object": "report.txt",
        "reversible_state": "The recycle bin is enabled and the file is stored there after deletion.",
        "partial_state": "The file has a backup that is one day old, so the latest changes cannot be recovered.",
        "irreversible_state": "The recycle bin is disabled and no backup exists."
    },
    {
        "action": "edit_file",
        "object": "report.txt",
        "reversible_state": "A backup of the original file is available.",
        "partial_state": "Only an older backup is available.",
        "irreversible_state": "No backup or previous version is available."
    },
    {
        "action": "send_email",
        "object": "project email",
        "reversible_state": "The email service provides an active undo option.",
        "partial_state": "The email can only be followed by a correction message.",
        "irreversible_state": "The email is sent immediately and cannot be recalled."
    },
    {
        "action": "transfer_money",
        "object": "1000 units",
        "reversible_state": "The transfer can be fully cancelled and the complete amount returned.",
        "partial_state": "Only 80 percent of the transferred amount can be refunded.",
        "irreversible_state": "The transfer is final and no refund is possible."
    }
]


labels = ["reversible", "partially_reversible", "irreversible"]

pair_types = [
    ("reversible", "irreversible"),
    ("partially_reversible", "irreversible"),
    ("reversible", "partially_reversible")
]


def get_state(scenario, label):
    if label == "reversible":
        return scenario["reversible_state"]

    if label == "partially_reversible":
        return scenario["partial_state"]

    return scenario["irreversible_state"]


pair_number = 0

while len(examples) < 2000:

    scenario = scenarios[pair_number % len(scenarios)]
    label_a, label_b = pair_types[pair_number % len(pair_types)]

    variation = pair_number + 1

    for label in [label_a, label_b]:

        state = get_state(scenario, label)

        example = {
            "id": f"SBX_{len(examples) + 1:04d}",
            "environment": "custom_sandbox",
            "state_text": (
                f"Scenario {variation}: {state} "
                f"The object involved is {scenario['object']}."
            ),
            "action_text": scenario["action"],
            "label": label,
            "horizon": 3,
            "source": "synthetic_controlled_scenario",
            "pair_id": f"PAIR_{pair_number + 1:04d}"
        }

        examples.append(example)

    pair_number += 1


random.shuffle(examples)

with open("dataset/irreversibility_dataset.jsonl", "w", encoding="utf-8") as file:
    for example in examples:
        file.write(json.dumps(example) + "\n")


counts = {
    label: 0 for label in labels
}

for example in examples:
    counts[example["label"]] += 1


print("DATASET GENERATION COMPLETED")
print("Total examples:", len(examples))
print("Reversible:", counts["reversible"])
print("Partially reversible:", counts["partially_reversible"])
print("Irreversible:", counts["irreversible"])
print("Minimal pairs:", pair_number)
print("Saved to: dataset/irreversibility_dataset.jsonl")
