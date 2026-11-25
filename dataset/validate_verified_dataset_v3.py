"""Independent validation for v3; does not import the v3 generator."""
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.grouping import assign_group_ids
from dataset.independent_outcomes import ACTIONS, LABELS, validate_example
from dataset.model_input import MODEL_INPUT_FIELDS


DATA_PATH = ROOT / "dataset" / "irreversibility_dataset_v3.jsonl"
REQUIRED = {
    "id", "environment", "state_text", "action_text", "label", "horizon", "source",
    "state_phase", "model_input", "action_parameters", "initial_operational_state",
    "post_action_operational_state", "final_operational_state", "recovery_attempts",
    "execution_trace", "initial_audit_history", "audit_history", "transfer_lifecycle",
    "scenario_id", "pair_id", "pair_role", "scenario_group_id", "input_group_id",
    "canonical_group_id", "wording_template_group_id", "evaluation_only_fields",
}
COMPARISONS = {
    ("irreversible", "partially_reversible"),
    ("irreversible", "reversible"),
    ("partially_reversible", "reversible"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def duplicate_input_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = canonical({field: row.get(field) for field in MODEL_INPUT_FIELDS})
        groups[key].append(row)
    return [members for members in groups.values() if len(members) > 1]


def _label_pair(members):
    return tuple(sorted((row["label"] for row in members)))


def validate_dataset(path=DATA_PATH):
    path = Path(path)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    errors = []
    ids = [row.get("id") for row in rows]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate IDs: {duplicate_ids[:10]}")
    if len(rows) < 2000:
        errors.append(f"Dataset has only {len(rows)} rows")

    class_counts = Counter(row.get("label") for row in rows)
    action_counts = Counter(row.get("action_text") for row in rows)
    cross_tab = {action: {label: sum(row.get("action_text") == action and row.get("label") == label
                                    for row in rows) for label in sorted(LABELS)}
                 for action in sorted(ACTIONS)}
    if set(class_counts) != LABELS or set(action_counts) != ACTIONS:
        errors.append("Dataset does not contain exactly the required label/action sets")
    if any(cross_tab[action][label] != 200 for action in ACTIONS for label in LABELS):
        errors.append("Action-by-class counts are not exactly 200 per cell")

    pair_groups = defaultdict(list)
    input_groups = duplicate_input_groups(rows)
    for row in rows:
        missing = REQUIRED - set(row)
        if missing:
            errors.append(f"{row.get('id')}: missing schema keys {sorted(missing)}")
            continue
        if row["environment"] != "custom_sandbox" or row["horizon"] != 3:
            errors.append(f"{row['id']}: wrong environment or horizon")
        if row["state_phase"] != "pre_action":
            errors.append(f"{row['id']}: state description is not pre-action")
        if row["model_input"] != {field: row[field] for field in MODEL_INPUT_FIELDS}:
            errors.append(f"{row['id']}: model input violates exact whitelist")
        hidden_required = {
            "label", "action_parameters", "initial_operational_state",
            "post_action_operational_state", "final_operational_state",
            "recovery_attempts", "execution_trace", "audit_history",
            "transfer_lifecycle",
        }
        if not hidden_required.issubset(set(row["evaluation_only_fields"])):
            errors.append(f"{row['id']}: evaluation-only field declaration is incomplete")
        if row["action_text"] not in ACTIONS or row["label"] not in LABELS:
            errors.append(f"{row['id']}: invalid action or label")
        if row["pair_id"]:
            pair_groups[row["pair_id"]].append(row)
        if any(term in row["state_text"].lower()
               for term in ("reversible", "irreversible", "partial recovery", "permanent action")):
            errors.append(f"{row['id']}: explicit class wording in state_text")
        check = validate_example(row)
        errors.extend(f"{row['id']}: {issue}" for issue in check["errors"])

    pair_combinations = Counter()
    pair_by_action = defaultdict(Counter)
    for pair_id, members in pair_groups.items():
        if len(members) != 2:
            errors.append(f"{pair_id}: expected two members, got {len(members)}")
            continue
        first, second = members
        if first["action_text"] != second["action_text"]:
            errors.append(f"{pair_id}: action mismatch")
        if first["scenario_id"] != second["scenario_id"]:
            errors.append(f"{pair_id}: scenario mismatch")
        if first["scenario_group_id"] != second["scenario_group_id"]:
            errors.append(f"{pair_id}: scenario group mismatch")
        if first["state_text"] == second["state_text"]:
            errors.append(f"{pair_id}: identical descriptions")
        field = "transfer_condition" if first["action_text"] == "transfer_money" else "recovery_condition"
        p1, p2 = first["action_parameters"], second["action_parameters"]
        if field not in p1 or field not in p2 or p1[field] == p2[field]:
            errors.append(f"{pair_id}: recovery condition did not change")
        if {k: v for k, v in p1.items() if k != field} != {k: v for k, v in p2.items() if k != field}:
            errors.append(f"{pair_id}: unrelated scenario parameters changed")
        combo = tuple(sorted((first["label"], second["label"])))
        pair_combinations[combo] += 1
        pair_by_action[first["action_text"]][combo] += 1
        if combo not in COMPARISONS:
            errors.append(f"{pair_id}: unsupported label combination {combo}")
    for action in ACTIONS:
        for combo in COMPARISONS:
            if pair_by_action[action][combo] != 60:
                errors.append(f"{action}: expected 60 pairs for {combo}, found {pair_by_action[action][combo]}")

    paired_rows = sum(map(len, pair_groups.values()))
    paired_percentage = paired_rows / len(rows) * 100 if rows else 0
    if paired_percentage < 30:
        errors.append(f"Only {paired_percentage:.2f}% of examples are paired")
    input_label_conflicts = [members for members in input_groups
                             if len({row["label"] for row in members}) > 1]
    if input_label_conflicts:
        errors.append(f"{len(input_label_conflicts)} identical model inputs have conflicting labels")

    # Independently recompute scenario/input union groups and verify stored IDs.
    regrouped = assign_group_ids(copy.deepcopy(rows))
    group_id_mismatches = sum(
        before.get("canonical_group_id") != after.get("canonical_group_id")
        or before.get("scenario_group_id") != after.get("scenario_group_id")
        or before.get("input_group_id") != after.get("input_group_id")
        for before, after in zip(rows, regrouped)
    )
    if group_id_mismatches:
        errors.append(f"{group_id_mismatches} stored canonical group identifiers are incorrect")

    template_groups = len({row.get("wording_template_group_id") for row in rows})
    canonical_groups = len({row.get("canonical_group_id") for row in rows})
    scenario_groups = len({row.get("scenario_group_id") for row in rows})
    template_sizes = Counter(row["wording_template_group_id"] for row in rows)
    normalized_text = set()
    for row in rows:
        text = row["state_text"].lower()
        for key, value in sorted(row["action_parameters"].items(),
                                 key=lambda item: -len(str(item[1]))):
            if key in {"recovery_condition", "transfer_condition"}:
                continue
            text = text.replace(str(value).lower(), "<value>")
        text = re.sub(r"\d+", "<number>", text)
        normalized_text.add(" ".join(text.split()))
    exact_input_unique = len(rows) - sum(len(group) - 1 for group in input_groups)
    report = {
        "dataset": str(path),
        "total_examples": len(rows),
        "cross_tab": cross_tab,
        "class_counts": {label: class_counts[label] for label in sorted(LABELS)},
        "action_counts": {action: action_counts[action] for action in sorted(ACTIONS)},
        "unique_ids": len(set(ids)),
        "pair_count": len(pair_groups),
        "pair_label_combinations": {" / ".join(k): v for k, v in sorted(pair_combinations.items())},
        "pairs_by_action": {action: {" / ".join(k): v for k, v in sorted(pair_by_action[action].items())}
                            for action in sorted(ACTIONS)},
        "paired_examples": paired_rows,
        "paired_percentage": round(paired_percentage, 2),
        "unique_model_inputs": exact_input_unique,
        "duplicate_model_input_groups": len(input_groups),
        "conflicting_duplicate_input_groups": len(input_label_conflicts),
        "unique_state_descriptions": len({row["state_text"] for row in rows}),
        "normalized_wording_patterns": len(normalized_text),
        "wording_template_groups": template_groups,
        "rows_per_wording_template": {
            "min": min(template_sizes.values()) if template_sizes else 0,
            "median": sorted(template_sizes.values())[len(template_sizes) // 2] if template_sizes else 0,
            "max": max(template_sizes.values()) if template_sizes else 0,
        },
        "scenario_groups": scenario_groups,
        "canonical_leakage_groups": canonical_groups,
        "independent_examples_checked": len(rows),
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    validate_dataset()
