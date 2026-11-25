"""Independent validation of challenge outcomes, pairs, groups, and split."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.reference_oracle import validate_outcome_record
from dataset.model_input import MODEL_INPUT_FIELDS


DATASET = ROOT / "dataset" / "irreversibility_challenge_v1.jsonl"
SPLIT_ROOT = ROOT / "dataset" / "experimental_challenge_v1"
SEED = 20261125
ACTIONS = ("edit_file", "delete_file", "send_email", "transfer_money")
LABELS = ("reversible", "partially_reversible", "irreversible")
COMPARISONS = (
    ("reversible", "partially_reversible"),
    ("reversible", "irreversible"),
    ("partially_reversible", "irreversible"),
)
FORBIDDEN = ("reversible", "irreversible", "partial recovery", "permanent action")
SUCCESS_RESULTS = {
    "Backup restored", "File restored", "Email recalled", "Correction sent",
    "Transfer cancelled", "Refund completed",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stable_id(prefix, value):
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _derive_group_ids(rows):
    """Recompute grouping without importing dataset.grouping."""
    scenario_ids, input_ids = [], []
    for row in rows:
        params = copy.deepcopy(row["action_parameters"])
        params.pop("recovery_condition", None)
        params.pop("transfer_condition", None)
        scenario_ids.append(_stable_id("SCENARIO", {
            "action": row["action_text"], "parameters": params}))
        input_ids.append(_stable_id("INPUT", {
            "environment": row["environment"], "state_text": row["state_text"],
            "action_text": row["action_text"],
        }))

    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, keys in enumerate(zip(scenario_ids, input_ids)):
        for key in keys:
            if key in owners:
                left, right = find(index), find(owners[key])
                if left != right:
                    parent[right] = left
            else:
                owners[key] = index

    component_scenarios = defaultdict(set)
    for index, scenario_id in enumerate(scenario_ids):
        component_scenarios[find(index)].add(scenario_id)
    canonical_ids = {}
    for root, component in component_scenarios.items():
        canonical_ids[root] = _stable_id("GROUP", sorted(component))
    return [scenario_ids, input_ids,
            [canonical_ids[find(index)] for index in range(len(rows))]]


def _counts(rows):
    return {
        "total": len(rows),
        "class": {label: sum(row["label"] == label for row in rows) for label in LABELS},
        "action": {action: sum(row["action_text"] == action for row in rows) for action in ACTIONS},
        "action_class": {
            action: {
                label: sum(row["action_text"] == action and row["label"] == label for row in rows)
                for label in LABELS
            }
            for action in ACTIONS
        },
    }


def _expected_recovery_tools(row):
    action = row["action_text"]
    condition = row["action_parameters"].get(
        "recovery_condition", row["action_parameters"].get("transfer_condition"))
    plans = {
        "edit_file": ["restore_backup"],
        "delete_file": ["restore_file", "restore_backup"],
        "send_email": ["undo_email", "send_correction"],
        "transfer_money": ["cancel_transfer", "refund_transfer"],
    }
    # A successful call stops the plan; failed calls before it consume the horizon.
    results = row["recovery_attempts"]
    expected = []
    for tool, attempt in zip(plans[action], results):
        expected.append(tool)
        if attempt.get("tool") != tool:
            break
        if attempt.get("result") in SUCCESS_RESULTS:
            break
    return expected


def _expected_audit_events(row):
    action = row["action_text"]
    events = [item.get("event") for item in row["initial_audit_history"]]
    if action == "send_email":
        events.append("email_sent")
    elif action == "edit_file":
        events.append("file_edited")
    elif action == "delete_file":
        events.append("file_deleted")
    else:
        events.append("transfer_created")
        if row["action_parameters"]["transfer_condition"] != "pending":
            events.append("transfer_settled")

    success_event = {
        "restore_backup": ("Backup restored", "backup_restored"),
        "restore_file": ("File restored", "file_restored"),
        "undo_email": ("Email recalled", "email_recalled"),
        "send_correction": ("Correction sent", "email_correction_sent"),
        "cancel_transfer": ("Transfer cancelled", "transfer_cancelled"),
        "refund_transfer": ("Refund completed", "transfer_refunded"),
    }
    rejected_event = {
        "restore_backup": "backup_restore_rejected",
        "restore_file": "file_restore_rejected",
        "undo_email": "email_recall_rejected",
        "send_correction": "email_correction_rejected",
        "cancel_transfer": "cancellation_rejected",
        "refund_transfer": "refund_rejected",
    }
    for attempt in row["recovery_attempts"]:
        tool = attempt["tool"]
        success_result, success_name = success_event[tool]
        events.append(success_name if attempt["result"] == success_result else rejected_event[tool])
    return events


def _expected_test_groups(rows):
    """Independently reconstruct seed-stratified pair selection."""
    pair_rows = defaultdict(list)
    for row in rows:
        pair_rows[row["pair_id"]].append(row)
    strata = defaultdict(list)
    pair_group = {}
    for pair_id, members in pair_rows.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} has an invalid member count")
        first, second = members
        if first["canonical_group_id"] != second["canonical_group_id"]:
            raise ValueError(f"{pair_id} crosses canonical groups")
        pair_group[pair_id] = first["canonical_group_id"]
        combination = tuple(label for label in LABELS
                            if label in {first["label"], second["label"]})
        strata[(first["action_text"], combination)].append(pair_id)
    randomizer = random.Random(SEED)
    selected_groups = set()
    for action in ACTIONS:
        for combination in COMPARISONS:
            candidates = sorted(strata[(action, combination)])
            if len(candidates) != 10:
                raise ValueError(f"Expected ten pairs for {action}/{combination}")
            randomizer.shuffle(candidates)
            selected_groups.update(pair_group[pair] for pair in candidates[:2])
    return selected_groups


def validate():
    raw = DATASET.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256(raw).hexdigest()
    errors = []
    if len(rows) != 240:
        errors.append(f"Expected 240 examples; found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Example IDs are not unique")
    if set(row.get("action_text") for row in rows) != set(ACTIONS):
        errors.append("Action set mismatch")
    if set(row.get("label") for row in rows) != set(LABELS):
        errors.append("Label set mismatch")
    full_counts = _counts(rows)
    if any(full_counts["action_class"][a][y] != 20 for a in ACTIONS for y in LABELS):
        errors.append("Action/class cells are not exactly 20 each")

    required = {
        "id", "environment", "state_text", "action_text", "label", "horizon", "source",
        "state_phase", "model_input", "action_parameters", "initial_operational_state",
        "post_action_operational_state", "final_operational_state", "recovery_attempts",
        "recovery_outcome", "execution_trace", "initial_audit_history", "audit_history",
        "transfer_lifecycle", "scenario_id", "pair_id", "pair_role",
        "wording_template_group_id", "scenario_group_id", "input_group_id",
        "canonical_group_id", "evaluation_only_fields",
    }
    pair_rows = defaultdict(list)
    input_labels = defaultdict(set)
    template_labels = defaultdict(set)
    oracle_checks = 0
    for row in rows:
        missing = required - set(row)
        if missing:
            errors.append(f"{row.get('id')}: missing keys {sorted(missing)}")
            continue
        if row["environment"] != "custom_sandbox" or row["horizon"] != 3:
            errors.append(f"{row['id']}: incorrect environment or horizon")
        if row["source"] != "sandbox_execution_challenge_v1" or row["state_phase"] != "pre_action":
            errors.append(f"{row['id']}: provenance or state phase mismatch")
        if row["model_input"] != {field: row[field] for field in MODEL_INPUT_FIELDS}:
            errors.append(f"{row['id']}: model input is not the exact whitelist")
        if len(row["recovery_attempts"]) > 3:
            errors.append(f"{row['id']}: recovery horizon exceeded")
        if [attempt.get("tool") for attempt in row["recovery_attempts"]] != _expected_recovery_tools(row):
            errors.append(f"{row['id']}: recovery tools do not follow the declared bounded plan")
        if not row["execution_trace"] or row["execution_trace"][0].get("tool") != row["action_text"]:
            errors.append(f"{row['id']}: missing initiating action evidence")
        elif row["execution_trace"][1:] != row["recovery_attempts"]:
            errors.append(f"{row['id']}: execution trace/recovery evidence mismatch")
        observed_events = [item.get("event") for item in row["audit_history"]]
        if observed_events != _expected_audit_events(row):
            errors.append(f"{row['id']}: audit events do not match executed action/recovery trace")
        if any(term in row["state_text"].lower() for term in FORBIDDEN):
            errors.append(f"{row['id']}: explicit label wording in model-visible state")
        oracle = validate_outcome_record(row)
        if not oracle["passed"]:
            errors.extend(f"{row['id']}: {issue}" for issue in oracle["errors"])
        else:
            oracle_checks += 1
        pair_rows[row["pair_id"]].append(row)
        input_labels[_canonical(row["model_input"])].add(row["label"])
        template_labels[(row["action_text"], row["wording_template_group_id"])].add(row["label"])

    if len(pair_rows) != 120:
        errors.append(f"Expected 120 pairs; found {len(pair_rows)}")
    pair_combinations = Counter()
    pairs_by_action_comparison = Counter()
    for pair_id, members in pair_rows.items():
        if len(members) != 2:
            errors.append(f"{pair_id}: expected two examples")
            continue
        first, second = members
        if first["action_text"] != second["action_text"]:
            errors.append(f"{pair_id}: action mismatch")
        if (first["scenario_id"] != second["scenario_id"]
                or first["scenario_group_id"] != second["scenario_group_id"]
                or first["canonical_group_id"] != second["canonical_group_id"]):
            errors.append(f"{pair_id}: scenario/canonical grouping mismatch")
        if first["wording_template_group_id"] != second["wording_template_group_id"]:
            errors.append(f"{pair_id}: wording template mismatch")
        key = ("transfer_condition" if first["action_text"] == "transfer_money"
               else "recovery_condition")
        params_left, params_right = first["action_parameters"], second["action_parameters"]
        if params_left.get(key) == params_right.get(key):
            errors.append(f"{pair_id}: recovery condition did not change")
        if ({k: v for k, v in params_left.items() if k != key}
                != {k: v for k, v in params_right.items() if k != key}):
            errors.append(f"{pair_id}: unrelated scenario parameters changed")
        combination = tuple(label for label in LABELS
                            if label in {first["label"], second["label"]})
        pair_combinations[combination] += 1
        pairs_by_action_comparison[(first["action_text"], combination)] += 1
        if first["label"] == second["label"] or combination not in COMPARISONS:
            errors.append(f"{pair_id}: invalid label contrast")
    for action in ACTIONS:
        for combination in COMPARISONS:
            if pairs_by_action_comparison[(action, combination)] != 10:
                errors.append(f"{action}/{combination}: expected ten matched pairs")
    if any(len(labels) > 1 for labels in input_labels.values()):
        errors.append("Identical model-visible inputs have conflicting labels")
    if len(template_labels) != 12 or any(values != set(LABELS) for values in template_labels.values()):
        errors.append("Each action/wording template must occur with all three labels")

    expected_scenario, expected_input, expected_canonical = _derive_group_ids(rows)
    group_mismatches = 0
    for index, row in enumerate(rows):
        if (row["scenario_group_id"] != expected_scenario[index]
                or row["input_group_id"] != expected_input[index]
                or row["canonical_group_id"] != expected_canonical[index]):
            group_mismatches += 1
    if group_mismatches:
        errors.append(f"{group_mismatches} stored group IDs disagree with independent recomputation")

    manifest_path = SPLIT_ROOT / "manifest.json"
    split_name = f"split_seed_{SEED}"
    split_dir = SPLIT_ROOT / split_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    train_ids = json.loads((split_dir / "training_example_ids.json").read_text(encoding="utf-8"))
    test_ids = json.loads((split_dir / "testing_example_ids.json").read_text(encoding="utf-8"))
    train_group_list = json.loads((split_dir / "training_canonical_group_ids.json").read_text(encoding="utf-8"))
    test_group_list = json.loads((split_dir / "testing_canonical_group_ids.json").read_text(encoding="utf-8"))
    row_by_id = {row["id"]: row for row in rows}
    if manifest.get("dataset_sha256") != digest or metadata.get("dataset_sha256") != digest:
        errors.append("Challenge split dataset hash mismatch")
    all_partition_ids = train_ids + test_ids
    if len(all_partition_ids) != len(set(all_partition_ids)) or set(all_partition_ids) != set(row_by_id):
        errors.append("Challenge split IDs do not form an exact, nonduplicated dataset partition")
    train_rows = [row_by_id[item] for item in train_ids if item in row_by_id]
    test_rows = [row_by_id[item] for item in test_ids if item in row_by_id]
    if (len(train_rows), len(test_rows)) != (192, 48):
        errors.append("Challenge split size is not 192/48")
    train_groups = {row["canonical_group_id"] for row in train_rows}
    test_groups = {row["canonical_group_id"] for row in test_rows}
    for field in ("canonical_group_id", "scenario_group_id", "input_group_id", "pair_id"):
        left = {row[field] for row in train_rows}
        right = {row[field] for row in test_rows}
        if left & right:
            errors.append(f"Challenge {field} crosses the evaluation boundary")
    if len(train_group_list) != len(set(train_group_list)) or set(train_group_list) != train_groups:
        errors.append("Saved train canonical-group list is invalid")
    if len(test_group_list) != len(set(test_group_list)) or set(test_group_list) != test_groups:
        errors.append("Saved test canonical-group list is invalid")
    train_counts, test_counts = _counts(train_rows), _counts(test_rows)
    if any(train_counts["action_class"][a][y] != 16 or test_counts["action_class"][a][y] != 4
           for a in ACTIONS for y in LABELS):
        errors.append("Challenge split action/class counts are incorrect")
    expected_test_groups = _expected_test_groups(rows)
    if expected_test_groups != test_groups:
        errors.append("Seeded challenge split replay differs from saved test groups")
    if metadata.get("training_counts") != train_counts or metadata.get("testing_counts") != test_counts:
        errors.append("Challenge split metadata counts disagree with the partitions")
    if metadata.get("training_group_count") != len(train_groups) or metadata.get("testing_group_count") != len(test_groups):
        errors.append("Challenge split metadata group counts disagree with partitions")
    if metadata.get("seed") != SEED:
        errors.append("Challenge split seed metadata mismatch")

    report = {
        "dataset": "irreversibility_challenge_v1.jsonl",
        "dataset_sha256": digest,
        "examples": len(rows),
        "action_class_counts": full_counts["action_class"],
        "pairs": len(pair_rows),
        "pair_combinations": {" / ".join(key): value for key, value in pair_combinations.items()},
        "independent_reference_outcomes_checked": oracle_checks,
        "recomputed_group_ids": len(rows),
        "split": {
            "seed": SEED,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "train_action_class": train_counts["action_class"],
            "test_action_class": test_counts["action_class"],
            "test_group_replay": expected_test_groups == test_groups,
            "crossings": {
                field: len({row[field] for row in train_rows} & {row[field] for row in test_rows})
                for field in ("canonical_group_id", "scenario_group_id", "input_group_id", "pair_id")
            },
        },
        "passed": not errors,
        "errors": errors,
    }
    (SPLIT_ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    return report


def _write_markdown(report):
    split = report["split"]
    lines = [
        "# Challenge Dataset v1 Validation",
        "",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Validation: **{'PASSED' if report['passed'] else 'FAILED'}**",
        f"Examples: {report['examples']}; matched pairs: {report['pairs']}",
        f"Independent reference outcomes checked: {report['independent_reference_outcomes_checked']}",
        "",
        f"Canonical-group split seed {split['seed']}: {split['train_examples']} train / {split['test_examples']} test; "
        f"{split['train_groups']} train groups / {split['test_groups']} test groups.",
        "",
        "| Action | Train reversible | Train partial | Train irreversible | Test reversible | Test partial | Test irreversible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action in ACTIONS:
        tr, te = split["train_action_class"][action], split["test_action_class"][action]
        lines.append(f"| {action} | {tr['reversible']} | {tr['partially_reversible']} | {tr['irreversible']} | "
                     f"{te['reversible']} | {te['partially_reversible']} | {te['irreversible']} |")
    lines.extend(["", "## Leakage checks", ""])
    for key, value in split["crossings"].items():
        lines.append(f"- `{key}` crossings: {value}")
    lines.extend([f"- Seed replay: {'PASS' if split['test_group_replay'] else 'FAIL'}", "",
                  "The challenge dataset is separate from v3. Labels were checked with the pure reference oracle; the challenge split does not replace the frozen v3 splits.", ""])
    (SPLIT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    validate()
