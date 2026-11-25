"""Legacy text-only dataset generator.

This file is retained for historical comparison. Its examples are not execution-
backed ground truth and must not be used as the project's verified dataset.
Running or importing the module never writes files. The command refuses to
overwrite an existing dataset.
"""

import json
import random
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "irreversibility_dataset.jsonl"
SEED = 42

SCENARIOS = [
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


LABELS = ["reversible", "partially_reversible", "irreversible"]

PAIR_TYPES = [
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


def build_examples():
    """Build the old text-only examples without executing sandbox actions."""
    rng = random.Random(SEED)
    examples = []
    pair_number = 0
    while len(examples) < 2000:
        scenario = SCENARIOS[pair_number % len(SCENARIOS)]
        label_a, label_b = PAIR_TYPES[pair_number % len(PAIR_TYPES)]
        variation = pair_number + 1
        for label in (label_a, label_b):
            examples.append({
                "id": f"SBX_{len(examples) + 1:04d}",
                "environment": "custom_sandbox",
                "state_text": (
                    f"Scenario {variation}: {get_state(scenario, label)} "
                    f"The object involved is {scenario['object']}."
                ),
                "action_text": scenario["action"],
                "label": label,
                "horizon": 3,
                "source": "synthetic_controlled_scenario",
                "pair_id": f"PAIR_{pair_number + 1:04d}",
            })
        pair_number += 1
    rng.shuffle(examples)
    return examples


def generate_dataset(output_path=OUTPUT_PATH):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        for example in examples:
            output_file.write(json.dumps(example) + "\n")
    counts = {label: sum(row["label"] == label for row in examples) for label in LABELS}
    print(f"Generated {len(examples)} legacy examples at {output_path}")
    print(json.dumps(counts, indent=2))
    return examples


if __name__ == "__main__":
    generate_dataset()
