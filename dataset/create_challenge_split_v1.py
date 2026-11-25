"""Create a canonical-grouped 80/20 evaluation split for the challenge set."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "irreversibility_challenge_v1.jsonl"
OUTPUT = ROOT / "dataset" / "experimental_challenge_v1"
SEED = 20261125
ACTIONS = ("edit_file", "delete_file", "send_email", "transfer_money")
LABEL_ORDER = ("reversible", "partially_reversible", "irreversible")
PAIR_COMPARISONS = (
    ("reversible", "partially_reversible"),
    ("reversible", "irreversible"),
    ("partially_reversible", "irreversible"),
)
TEST_PAIRS_PER_STRATUM = 2


def _read_rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _counts(rows):
    return {
        "total": len(rows),
        "class": {label: sum(row["label"] == label for row in rows) for label in LABEL_ORDER},
        "action": {action: sum(row["action_text"] == action for row in rows) for action in ACTIONS},
        "action_class": {
            action: {
                label: sum(row["action_text"] == action and row["label"] == label for row in rows)
                for label in LABEL_ORDER
            }
            for action in ACTIONS
        },
    }


def _select_test_pairs(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    strata = defaultdict(list)
    pair_to_group = {}
    for pair_id, members in pairs.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} is not a two-example pair")
        first, second = members
        if first["canonical_group_id"] != second["canonical_group_id"]:
            raise ValueError(f"{pair_id} crosses canonical groups")
        group_id = first["canonical_group_id"]
        if group_id in pair_to_group.values():
            raise ValueError("A canonical group contains multiple pair IDs")
        pair_to_group[pair_id] = group_id
        labels = set((first["label"], second["label"]))
        comparison = next(tuple(sorted(candidate, key=LABEL_ORDER.index))
                          for candidate in PAIR_COMPARISONS if set(candidate) == labels)
        strata[(first["action_text"], comparison)].append(pair_id)

    rng = random.Random(SEED)
    selected = set()
    for action in ACTIONS:
        for comparison in PAIR_COMPARISONS:
            key = (action, comparison)
            candidates = sorted(strata[key])
            if len(candidates) != 10:
                raise ValueError(f"Expected 10 pairs in {key}; found {len(candidates)}")
            rng.shuffle(candidates)
            selected.update(candidates[:TEST_PAIRS_PER_STRATUM])
    return selected, pairs, pair_to_group


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_split():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen challenge split artifacts: {OUTPUT}")
    raw = DATASET.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    selected_pairs, pairs, pair_to_group = _select_test_pairs(rows)
    test_groups = {pair_to_group[pair_id] for pair_id in selected_pairs}
    train_rows = [row for row in rows if row["canonical_group_id"] not in test_groups]
    test_rows = [row for row in rows if row["canonical_group_id"] in test_groups]
    if (len(train_rows), len(test_rows)) != (192, 48):
        raise ValueError("Challenge split did not produce the expected 192/48 rows")
    split_dir = OUTPUT / f"split_seed_{SEED}"
    train_groups = sorted({row["canonical_group_id"] for row in train_rows})
    test_group_ids = sorted({row["canonical_group_id"] for row in test_rows})
    digest = hashlib.sha256(raw).hexdigest()
    config = {
        "algorithm": "stratified seeded shuffle of pair groups, then canonical-group assignment",
        "selection_unit": "canonical_group_id; each group is verified to contain one matched pair",
        "seed": SEED,
        "test_pairs_per_action_label_comparison": TEST_PAIRS_PER_STRATUM,
        "expected_test_examples_per_action_class": 4,
        "expected_train_examples_per_action_class": 16,
    }
    _write_json(split_dir / "training_example_ids.json", [row["id"] for row in train_rows])
    _write_json(split_dir / "testing_example_ids.json", [row["id"] for row in test_rows])
    _write_json(split_dir / "training_canonical_group_ids.json", train_groups)
    _write_json(split_dir / "testing_canonical_group_ids.json", test_group_ids)
    _write_json(split_dir / "metadata.json", {
        "split_name": split_dir.name,
        "seed": SEED,
        "dataset_version": "irreversibility_challenge_v1",
        "dataset_sha256": digest,
        "generation_configuration": config,
        "training_counts": _counts(train_rows),
        "testing_counts": _counts(test_rows),
        "training_group_count": len(train_groups),
        "testing_group_count": len(test_group_ids),
    })
    _write_json(OUTPUT / "manifest.json", {
        "manifest_version": 1,
        "dataset_path": "dataset/irreversibility_challenge_v1.jsonl",
        "dataset_version": "irreversibility_challenge_v1",
        "dataset_sha256": digest,
        "seed": SEED,
        "split_directory": split_dir.name,
        "split_role": "challenge_train_test_evaluation; not a replacement for v3 splits",
    })
    return {
        "seed": SEED,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "train_groups": len(train_groups),
        "test_groups": len(test_group_ids),
        "train_action_class": _counts(train_rows)["action_class"],
        "test_action_class": _counts(test_rows)["action_class"],
        "pairs": len(pairs),
        "test_pairs": len(selected_pairs),
        "dataset_sha256": digest,
    }


if __name__ == "__main__":
    print(json.dumps(create_split(), indent=2))
