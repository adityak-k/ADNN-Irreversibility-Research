"""Independent validation and deterministic replay for saved v3 splits.

This module intentionally does not import the split generator. It reconstructs
group-level constraints and seeded assignments from the source dataset.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "irreversibility_dataset_v3.jsonl"
SPLITS = ROOT / "dataset" / "experimental_splits_v1"
EXPECTED_SEEDS = (20261120, 20261121, 20261122)
ACTION_ORDER = ("edit_file", "delete_file", "send_email", "transfer_money")
LABEL_ORDER = ("reversible", "partially_reversible", "irreversible")


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cross_tab(rows):
    return {
        action: {
            label: sum(r["action_text"] == action and r["label"] == label for r in rows)
            for label in LABEL_ORDER
        }
        for action in ACTION_ORDER
    }


def _partition_counts(rows):
    return {
        "total": len(rows),
        "class": {label: sum(r["label"] == label for r in rows) for label in LABEL_ORDER},
        "action": {action: sum(r["action_text"] == action for r in rows) for action in ACTION_ORDER},
        "action_class": _cross_tab(rows),
    }


def _independent_expected_groups(rows, seed):
    """Rebuild the seeded MILP without calling generator code."""
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[row["canonical_group_id"]].append(row)
    keys = sorted(group_rows.keys())
    strata = [(a, y) for a in ACTION_ORDER for y in LABEL_ORDER]
    coefficients = np.asarray([
        [sum(r["action_text"] == action and r["label"] == label for r in group_rows[g])
         for g in keys]
        for action, label in strata
    ], dtype=np.float64)
    costs = np.random.default_rng(seed).random(len(keys))
    answer = milp(
        costs,
        integrality=np.ones(len(keys), dtype=np.int8),
        bounds=Bounds(np.zeros(len(keys)), np.ones(len(keys))),
        constraints=LinearConstraint(
            coefficients,
            np.full(12, 40.0),
            np.full(12, 40.0),
        ),
        options={"disp": False},
    )
    if not answer.success or answer.x is None:
        raise AssertionError(f"Independent replay failed for seed {seed}: {answer.message}")
    return {keys[i] for i, x in enumerate(answer.x) if x > 0.5}


def validate():
    raw = DATASET.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    dataset_hash = hashlib.sha256(raw).hexdigest()
    manifest = _load_json(SPLITS / "manifest.json")
    errors = []
    results = []
    assignment_signatures = []
    if manifest.get("dataset_sha256") != dataset_hash:
        errors.append("Manifest dataset hash does not match v3 dataset")
    if tuple(manifest.get("seeds", [])) != EXPECTED_SEEDS:
        errors.append("Manifest seed list does not match the three approved seeds")
    expected_directories = [f"split_seed_{seed}" for seed in EXPECTED_SEEDS]
    if manifest.get("split_directories") != expected_directories:
        errors.append("Manifest split directories do not match the approved seeds")
    row_by_id = {r["id"]: r for r in rows}
    if len(row_by_id) != len(rows):
        errors.append("Source dataset has duplicate IDs")

    for seed in EXPECTED_SEEDS:
        name = f"split_seed_{seed}"
        directory = SPLITS / name
        metadata = _load_json(directory / "metadata.json")
        train_ids = _load_json(directory / "training_example_ids.json")
        test_ids = _load_json(directory / "testing_example_ids.json")
        train_saved_group_list = _load_json(directory / "training_canonical_group_ids.json")
        test_saved_group_list = _load_json(directory / "testing_canonical_group_ids.json")
        train_saved_groups = set(train_saved_group_list)
        test_saved_groups = set(test_saved_group_list)
        local_errors = []
        if len(train_saved_group_list) != len(train_saved_groups):
            local_errors.append("duplicate training canonical group IDs in artifact")
        if len(test_saved_group_list) != len(test_saved_groups):
            local_errors.append("duplicate testing canonical group IDs in artifact")
        all_ids = train_ids + test_ids
        if len(all_ids) != len(set(all_ids)):
            local_errors.append("duplicate ID within/across partitions")
        if set(all_ids) != set(row_by_id):
            local_errors.append("partition IDs do not form an exact cover of the dataset")
        train_rows = [row_by_id[i] for i in train_ids if i in row_by_id]
        test_rows = [row_by_id[i] for i in test_ids if i in row_by_id]
        if len(train_rows) != 1920 or len(test_rows) != 480:
            local_errors.append("train/test target sizes differ from 1920/480")

        # Independently derive leakage sets from source rows, not saved group lists.
        def groups_of(partition, field):
            return {r[field] for r in partition if r.get(field)}
        for field in ("canonical_group_id", "scenario_group_id", "input_group_id"):
            if groups_of(train_rows, field) & groups_of(test_rows, field):
                local_errors.append(f"{field} crosses train/test")
        pair_train = {r["pair_id"] for r in train_rows if r.get("pair_id")}
        pair_test = {r["pair_id"] for r in test_rows if r.get("pair_id")}
        if pair_train & pair_test:
            local_errors.append("matched pair crosses train/test")

        actual_train_groups = groups_of(train_rows, "canonical_group_id")
        actual_test_groups = groups_of(test_rows, "canonical_group_id")
        if actual_train_groups != train_saved_groups or actual_test_groups != test_saved_groups:
            local_errors.append("saved canonical group lists do not match ID partitions")
        train_cross, test_cross = _cross_tab(train_rows), _cross_tab(test_rows)
        if any(train_cross[a][y] != 160 or test_cross[a][y] != 40
               for a in ACTION_ORDER for y in LABEL_ORDER):
            local_errors.append("action/class target cells are not exact")

        # Replay the seeded solver independently and compare assignment membership.
        expected_test_groups = _independent_expected_groups(rows, seed)
        if expected_test_groups != actual_test_groups:
            local_errors.append("same-seed independent replay produced different test groups")
        if metadata.get("dataset_sha256") != dataset_hash:
            local_errors.append("split metadata dataset hash mismatch")
        if metadata.get("seed") != seed:
            local_errors.append("split metadata seed mismatch")
        if metadata.get("dataset_version") != "irreversibility_dataset_v3":
            local_errors.append("split metadata dataset version mismatch")
        if metadata.get("training_counts") != _partition_counts(train_rows):
            local_errors.append("training metadata counts do not match partition rows")
        if metadata.get("testing_counts") != _partition_counts(test_rows):
            local_errors.append("testing metadata counts do not match partition rows")
        if metadata.get("training_group_count") != len(actual_train_groups):
            local_errors.append("training group count metadata mismatch")
        if metadata.get("testing_group_count") != len(actual_test_groups):
            local_errors.append("testing group count metadata mismatch")
        config = metadata.get("generation_configuration", {})
        if config.get("seed") != seed or config.get("algorithm") != "scipy.optimize.milp":
            local_errors.append("split generation configuration is incomplete or inconsistent")
        assignment_signatures.append(tuple(sorted(actual_test_groups)))

        results.append({
            "split": name,
            "seed": seed,
            "passed": not local_errors,
            "errors": local_errors,
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "train_groups": len(actual_train_groups),
            "test_groups": len(actual_test_groups),
            "train_action_class": train_cross,
            "test_action_class": test_cross,
            "leakage_checks": {
                "canonical_group_crossings": len(actual_train_groups & actual_test_groups),
                "scenario_group_crossings": len(groups_of(train_rows, "scenario_group_id") & groups_of(test_rows, "scenario_group_id")),
                "input_group_crossings": len(groups_of(train_rows, "input_group_id") & groups_of(test_rows, "input_group_id")),
                "matched_pair_crossings": len(pair_train & pair_test),
            },
            "reproducibility_check": "passed" if expected_test_groups == actual_test_groups else "failed",
        })
        errors.extend(f"{name}: {issue}" for issue in local_errors)

    if len(set(assignment_signatures)) != len(EXPECTED_SEEDS):
        errors.append("Different seeds produced duplicate test group assignments")

    report = {
        "dataset_path": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": "irreversibility_dataset_v3",
        "dataset_sha256": dataset_hash,
        "rows": len(rows),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "splits": results,
        "passed": not errors,
        "errors": errors,
    }
    (SPLITS / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report_markdown(report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    return report


def _write_report_markdown(report):
    lines = [
        "# Experimental Split Validation Report",
        "",
        f"Dataset: `{report['dataset_path']}` ({report['dataset_version']})",
        f"SHA-256: `{report['dataset_sha256']}`",
        f"Validation: **{'PASSED' if report['passed'] else 'FAILED'}**",
        "",
        "| Seed | Train rows | Test rows | Train groups | Test groups | Cell counts | Leakage | Replay |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for split in report["splits"]:
        cells = all(v == 160 for counts in split["train_action_class"].values() for v in counts.values()) and all(
            v == 40 for counts in split["test_action_class"].values() for v in counts.values())
        leak = all(value == 0 for value in split["leakage_checks"].values())
        lines.append(
            f"| {split['seed']} | {split['train_examples']} | {split['test_examples']} | "
            f"{split['train_groups']} | {split['test_groups']} | {'PASS' if cells else 'FAIL'} | "
            f"{'PASS' if leak else 'FAIL'} | {split['reproducibility_check'].upper()} |"
        )
    lines.extend(["", "## Action/class counts", ""])
    for split in report["splits"]:
        lines.extend([f"### Seed {split['seed']}", "", "Training (target 160 per cell):", "", "| Action | Reversible | Partial | Irreversible |", "|---|---:|---:|---:|"])
        for action, counts in split["train_action_class"].items():
            lines.append(f"| {action} | {counts['reversible']} | {counts['partially_reversible']} | {counts['irreversible']} |")
        lines.extend(["", "Testing (target 40 per cell):", "", "| Action | Reversible | Partial | Irreversible |", "|---|---:|---:|---:|"])
        for action, counts in split["test_action_class"].items():
            lines.append(f"| {action} | {counts['reversible']} | {counts['partially_reversible']} | {counts['irreversible']} |")
        lines.append("")
    lines.extend(["## Validation scope", "", "Independent checks covered exact ID partitioning, duplicates/missing IDs, canonical/scenario/input group crossings, matched-pair crossings, action/class targets, dataset hashes, metadata, and deterministic same-seed MILP replay. No models or embeddings were used.", ""])
    (SPLITS / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    validate()
