"""Structural, balance, pair, conflict, and execution-replay validator."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from generate_verified_dataset import ACTIONS, LABELS, OUTPUT_PATH, execute_scenario


REQUIRED_FIELDS = {
    "id", "environment", "state_text", "action_text", "label", "horizon",
    "source", "pair_id", "scenario_id", "pair_role",
    "initial_operational_state", "final_operational_state", "recovery_actions",
    "recovery_outcome", "action_parameters", "execution_trace", "audit_history",
    "initial_audit_history", "state_phase", "evaluation_only_fields",
}
FORBIDDEN_TEXT = ("reversible", "irreversible", "partial recovery", "permanent action")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def validate(path=OUTPUT_PATH):
    path = Path(path)
    examples = []
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON: {exc}")

    ids = [row.get("id") for row in examples]
    id_counts = Counter(ids)
    duplicates = [item for item, count in id_counts.items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate IDs: {duplicates[:5]}")
    if len(examples) < 2000:
        errors.append(f"Expected at least 2,000 examples; found {len(examples)}")

    labels = Counter(row.get("label") for row in examples)
    actions = Counter(row.get("action_text") for row in examples)
    for label in LABELS:
        if not labels[label]:
            errors.append(f"Missing class: {label}")
    for action in ACTIONS:
        if not actions[action]:
            errors.append(f"Missing action: {action}")
    if labels and max(labels[label] for label in LABELS) - min(labels[label] for label in LABELS) > 1:
        errors.append("Class distribution differs by more than one example")
    if actions and max(actions[action] for action in ACTIONS) - min(actions[action] for action in ACTIONS) > 1:
        errors.append("Action distribution differs by more than one example")

    pairs = defaultdict(list)
    seen_examples = set()
    scenario_labels = defaultdict(set)
    replay_checked = 0
    text_leaks = []

    for index, row in enumerate(examples, start=1):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"Example {row.get('id', index)} missing fields: {sorted(missing)}")
            continue
        if row["label"] not in LABELS or row["action_text"] not in ACTIONS:
            errors.append(f"Example {row['id']} has invalid label/action")
            continue
        if row["environment"] != "custom_sandbox" or row["horizon"] != 3:
            errors.append(f"Example {row['id']} has invalid environment/horizon")
        if row["state_phase"] != "pre_action":
            errors.append(f"Example {row['id']} state_text is not marked pre-action")
        required_evaluation_fields = {
            "initial_operational_state", "final_operational_state", "recovery_actions",
            "recovery_outcome", "execution_trace", "initial_audit_history", "audit_history",
        }
        if not required_evaluation_fields.issubset(set(row["evaluation_only_fields"])):
            errors.append(f"Example {row['id']} does not mark evidence fields as evaluation-only")
        if not isinstance(row["state_text"], str) or not row["state_text"].strip():
            errors.append(f"Example {row['id']} has empty state_text")
        lower_text = row["state_text"].lower()
        if any(phrase in lower_text for phrase in FORBIDDEN_TEXT):
            text_leaks.append(row["id"])

        if row["pair_id"]:
            pairs[row["pair_id"]].append(row)
        duplicate_key = canonical({
            key: row[key] for key in (
                "action_text", "state_text", "initial_operational_state", "action_parameters"
            )
        })
        if duplicate_key in seen_examples:
            errors.append(f"Duplicate substantive example: {row['id']}")
        seen_examples.add(duplicate_key)
        scenario_signature = canonical({
            "action": row["action_text"],
            "initial_operational_state": row["initial_operational_state"],
            "action_parameters": row["action_parameters"],
        })
        scenario_labels[scenario_signature].add(row["label"])

        if len(row["recovery_actions"]) > 3:
            errors.append(f"Example {row['id']} exceeds recovery horizon")
        if not row["execution_trace"] or row["execution_trace"][0].get("tool") != row["action_text"]:
            errors.append(f"Example {row['id']} lacks initiating-action execution evidence")
            continue
        if row["execution_trace"][1:] != row["recovery_actions"]:
            errors.append(f"Example {row['id']} recovery trace mismatch")
            continue
        initial = row["initial_operational_state"]
        final = row["final_operational_state"]
        exact = initial == final
        progress = row["recovery_outcome"].get("meaningful_recovery_progress")
        expected_label = "reversible" if exact else (
            "partially_reversible" if progress else "irreversible"
        )
        if row["recovery_outcome"].get("exact_operational_match") != exact:
            errors.append(f"Example {row['id']} has incorrect operational comparison flag")
        if row["label"] != expected_label:
            errors.append(f"Example {row['id']} label disagrees with operational recovery comparison")

        try:
            replay = execute_scenario(row["action_text"], row["action_parameters"])
        except Exception as exc:  # validation must report a replay failure with its example ID
            errors.append(f"Example {row['id']} replay failed: {type(exc).__name__}: {exc}")
            continue
        fields = (
            ("label", "label"),
            ("initial_operational_state", "initial_operational_state"),
            ("final_operational_state", "final_operational_state"),
            ("recovery_actions", "recovery_actions"),
            ("recovery_outcome", "recovery_outcome"),
            ("execution_trace", "execution_trace"),
            ("initial_audit_history", "initial_audit_history"),
            ("audit_history", "audit_history"),
            ("transfer_lifecycle", "transfer_lifecycle"),
        )
        mismatch = [record_key for record_key, replay_key in fields
                    if row.get(record_key) != replay.get(replay_key)]
        if mismatch:
            errors.append(f"Example {row['id']} does not match sandbox replay: {mismatch}")
        else:
            replay_checked += 1

    for signature, conflicting in scenario_labels.items():
        if len(conflicting) > 1:
            errors.append(f"Conflicting labels for identical operational scenario: {signature[:140]}")

    paired_example_count = sum(len(members) for members in pairs.values())
    for pair_id, members in pairs.items():
        if len(members) != 2:
            errors.append(f"Pair {pair_id} has {len(members)} examples, expected 2")
            continue
        first, second = members
        if first["action_text"] != second["action_text"]:
            errors.append(f"Pair {pair_id} contains different actions")
        if first["scenario_id"] != second["scenario_id"]:
            errors.append(f"Pair {pair_id} contains different base scenarios")
        if first["label"] == second["label"]:
            errors.append(f"Pair {pair_id} does not change the verified outcome")
        if first["state_text"] == second["state_text"]:
            errors.append(f"Pair {pair_id} has identical state descriptions")
        p1 = first["action_parameters"]
        p2 = second["action_parameters"]
        condition_key = "transfer_condition" if first["action_text"] == "transfer_money" else "recovery_condition"
        normalized_1 = {k: v for k, v in p1.items() if k != condition_key}
        normalized_2 = {k: v for k, v in p2.items() if k != condition_key}
        if condition_key not in p1 or condition_key not in p2 or p1[condition_key] == p2[condition_key]:
            errors.append(f"Pair {pair_id} lacks one changed environmental condition")
        if normalized_1 != normalized_2:
            errors.append(f"Pair {pair_id} differs in more than its controlled condition")
    paired_fraction = paired_example_count / len(examples) if examples else 0
    if paired_fraction < 0.30:
        errors.append(f"Only {paired_fraction:.1%} of examples belong to minimal pairs")
    if text_leaks:
        errors.append(f"Explicit label wording appears in state_text for {text_leaks[:10]}")

    report = {
        "path": str(path),
        "total_examples": len(examples),
        "class_distribution": {label: labels[label] for label in LABELS},
        "action_distribution": {action: actions[action] for action in ACTIONS},
        "unique_ids": len(set(ids)),
        "pair_count": len(pairs),
        "paired_examples": paired_example_count,
        "paired_example_percentage": round(paired_fraction * 100, 2),
        "execution_replays_checked": replay_checked,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    validate()
