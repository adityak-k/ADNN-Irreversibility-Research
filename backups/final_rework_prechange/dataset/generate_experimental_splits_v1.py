"""Generate three deterministic, canonical-grouped v3 train/test splits."""
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
OUTPUT = ROOT / "dataset" / "experimental_splits_v1"
SEEDS = (20261120, 20261121, 20261122)
ACTIONS = ("edit_file", "delete_file", "send_email", "transfer_money")
LABELS = ("reversible", "partially_reversible", "irreversible")
TEST_PER_CELL = 40


def _read_dataset(path: Path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset contains duplicate example IDs")
    return raw, rows


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


def _select_test_groups(rows, seed):
    """Use a seeded MILP objective with exact 40-example action/class cells."""
    members = defaultdict(list)
    for row in rows:
        members[row["canonical_group_id"]].append(row)
    group_ids = sorted(members)
    cells = [(action, label) for action in ACTIONS for label in LABELS]
    matrix = np.zeros((len(cells), len(group_ids)), dtype=np.float64)
    for column, group_id in enumerate(group_ids):
        for row in members[group_id]:
            matrix[cells.index((row["action_text"], row["label"])), column] += 1

    # Seeded random costs select among feasible whole-group assignments.
    objective = np.random.default_rng(seed).random(len(group_ids))
    result = milp(
        c=objective,
        integrality=np.ones(len(group_ids), dtype=np.int8),
        bounds=Bounds(np.zeros(len(group_ids)), np.ones(len(group_ids))),
        constraints=LinearConstraint(
            matrix,
            np.full(len(cells), TEST_PER_CELL, dtype=np.float64),
            np.full(len(cells), TEST_PER_CELL, dtype=np.float64),
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"No exact grouped split found for seed {seed}: {result.message}")
    selected = {group_ids[i] for i, value in enumerate(result.x) if value > 0.5}
    test_rows = [row for row in rows if row["canonical_group_id"] in selected]
    if _counts(test_rows)["total"] != TEST_PER_CELL * len(cells):
        raise RuntimeError(f"Solver result for seed {seed} did not yield 480 test examples")
    return selected


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate():
    raw, rows = _read_dataset(DATASET)
    digest = hashlib.sha256(raw).hexdigest()
    dataset_version = "irreversibility_dataset_v3"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "dataset_path": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": dataset_version,
        "dataset_sha256": digest,
        "seeds": list(SEEDS),
        "split_directories": [],
        "split_algorithm": "Seeded SciPy MILP over canonical_group_id groups; exact 40 test rows per action/class cell.",
    }

    for seed in SEEDS:
        test_groups = _select_test_groups(rows, seed)
        test_rows = [r for r in rows if r["canonical_group_id"] in test_groups]
        train_rows = [r for r in rows if r["canonical_group_id"] not in test_groups]
        split_name = f"split_seed_{seed}"
        directory = OUTPUT / split_name
        directory.mkdir(parents=True, exist_ok=True)
        train_group_ids = sorted({r["canonical_group_id"] for r in train_rows})
        test_group_ids = sorted({r["canonical_group_id"] for r in test_rows})
        config = {
            "algorithm": "scipy.optimize.milp",
            "selection_unit": "canonical_group_id",
            "group_order": "lexicographically sorted canonical_group_id",
            "objective": "independent uniform random costs from numpy.random.default_rng(seed)",
            "constraints": "exactly 40 test examples for each of 4 actions x 3 labels",
            "integrality": "binary per group",
            "seed": seed,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        }
        _write_json(directory / "training_example_ids.json", [r["id"] for r in train_rows])
        _write_json(directory / "testing_example_ids.json", [r["id"] for r in test_rows])
        _write_json(directory / "training_canonical_group_ids.json", train_group_ids)
        _write_json(directory / "testing_canonical_group_ids.json", test_group_ids)
        metadata = {
            "split_name": split_name,
            "seed": seed,
            "dataset_version": dataset_version,
            "dataset_sha256": digest,
            "generation_configuration": config,
            "training_counts": _counts(train_rows),
            "testing_counts": _counts(test_rows),
            "training_group_count": len(train_group_ids),
            "testing_group_count": len(test_group_ids),
        }
        _write_json(directory / "metadata.json", metadata)
        manifest["split_directories"].append(split_name)

    _write_json(OUTPUT / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
