"""Independently check saved RQ1 IDs, representations, and layer metrics."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_rq1 import (
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_LAYERS,
    LABELS,
    PROJECT_OUTPUT,
    SPLIT_SEED,
    SPLIT_SEEDS,
    load_frozen_split,
    read_dataset,
    read_json,
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_partition_shards(folder, expected_rows):
    expected = {row["id"]: row["label"] for row in expected_rows}
    saved = {}
    shard_count = 0
    for shard_path in sorted(folder.glob("*.npz")):
        shard_count += 1
        with np.load(shard_path, allow_pickle=False) as shard:
            ids = shard["example_ids"].astype(str).tolist()
            labels = shard["labels"].astype(str).tolist()
            states = shard["hidden_states"]
        if states.shape != (len(ids), EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE):
            raise AssertionError(f"Bad representation shape in {shard_path.name}: {states.shape}")
        if states.dtype != np.float16:
            raise AssertionError(f"Representations are not FP16 in {shard_path.name}")
        for example_id, label in zip(ids, labels):
            if example_id not in expected or label != expected[example_id]:
                raise AssertionError(f"Unexpected ID/label in {shard_path.name}: {example_id}")
            if example_id in saved:
                raise AssertionError(f"Duplicate representation ID: {example_id}")
            saved[example_id] = label
    if saved != expected:
        raise AssertionError(f"Representation ID coverage mismatch in {folder}")
    return shard_count


def calculate_metrics(true_labels, predicted_labels):
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion[LABELS.index(true_label)][LABELS.index(predicted_label)] += 1
    total = len(true_labels)
    row_totals = [sum(row) for row in confusion]
    column_totals = [sum(confusion[row][column] for row in range(len(LABELS)))
                     for column in range(len(LABELS))]
    recalls = [confusion[index][index] / row_totals[index] for index in range(len(LABELS))]
    precisions = [confusion[index][index] / column_totals[index]
                  if column_totals[index] else 0.0 for index in range(len(LABELS))]
    f1s = [2 * precision * recall / (precision + recall)
           if precision + recall else 0.0
           for precision, recall in zip(precisions, recalls)]
    return {
        "count": total,
        "accuracy": sum(confusion[index][index] for index in range(len(LABELS))) / total,
        "balanced_accuracy": sum(recalls) / len(LABELS),
        "macro_precision": sum(precisions) / len(LABELS),
        "macro_recall": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1s) / len(LABELS),
        "confusion_matrix": confusion,
    }


def main(split_seed):
    run = PROJECT_OUTPUT / f"rq1_seed_{split_seed}"
    rows = read_dataset()
    train_rows, test_rows, dataset_hash, metadata = load_frozen_split(rows, split_seed)
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise AssertionError("RQ1 manifest is not complete")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise AssertionError("RQ1 manifest dataset hash mismatch")

    train_shards = verify_partition_shards(run / "representations" / "train", train_rows)
    test_shards = verify_partition_shards(run / "representations" / "test", test_rows)
    expected_test = {row["id"]: row["label"] for row in test_rows}
    metric_records = read_json(run / "layer_metrics.json")
    if len(metric_records) != EXPECTED_LAYERS:
        raise AssertionError("Expected 29 layer metric records")

    layer_checks = []
    for layer in range(EXPECTED_LAYERS):
        prediction_path = run / "layer_results" / f"layer_{layer:02d}_predictions.jsonl"
        metric_path = run / "layer_results" / f"layer_{layer:02d}_metrics.json"
        predictions = load_jsonl(prediction_path)
        if len(predictions) != len(test_rows):
            raise AssertionError(f"Wrong prediction count for layer {layer}")
        ids = [item.get("id") for item in predictions]
        if len(set(ids)) != len(ids) or set(ids) != set(expected_test):
            raise AssertionError(f"Wrong or duplicate prediction IDs for layer {layer}")
        for item in predictions:
            if item.get("true_label") != expected_test[item["id"]]:
                raise AssertionError(f"Wrong ground truth saved for {item['id']}, layer {layer}")
            if item.get("layer") != layer or item.get("predicted_label") not in LABELS:
                raise AssertionError(f"Invalid prediction record for {item['id']}, layer {layer}")
            if set(item) != {"id", "true_label", "predicted_label", "layer"}:
                raise AssertionError(f"Unexpected fields in model prediction for {item['id']}")

        true_labels = [item["true_label"] for item in predictions]
        predicted_labels = [item["predicted_label"] for item in predictions]
        calculated = calculate_metrics(true_labels, predicted_labels)
        saved = read_json(metric_path)
        for key, value in calculated.items():
            saved_value = saved.get(key)
            if isinstance(value, float):
                if not isinstance(saved_value, (float, int)) or not math.isclose(value, saved_value, abs_tol=1e-10):
                    raise AssertionError(f"Metric mismatch for layer {layer}: {key}")
            elif saved_value != value:
                raise AssertionError(f"Metric mismatch for layer {layer}: {key}")
        if metric_records[layer] != saved:
            raise AssertionError(f"Summary metric disagrees with layer {layer} file")
        layer_checks.append({
            "layer": layer,
            "accuracy": calculated["accuracy"],
            "balanced_accuracy": calculated["balanced_accuracy"],
            "macro_f1": calculated["macro_f1"],
            "prediction_ids_verified": len(ids),
        })

    report = {
        "status": "passed",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "split_seed": split_seed,
        "training_examples": len(train_rows),
        "testing_examples": len(test_rows),
        "train_representation_shards": train_shards,
        "test_representation_shards": test_shards,
        "representations_per_example": [EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE],
        "representation_dtype": "float16",
        "layers_independently_recalculated": layer_checks,
        "split_group_counts": {
            "train": metadata["training_group_count"],
            "test": metadata["testing_group_count"],
        },
        "checks": [
            "all train and test representation IDs and labels match frozen split",
            "representations are FP16 with 29 outputs x 1536 hidden values",
            "all 29 prediction files have every test ID exactly once",
            "saved true labels match v3 labels and prediction records contain no evaluation-only fields",
            "accuracy, balanced accuracy, macro precision/recall/F1 and confusion matrices recalculated directly from predictions",
        ],
    }
    report_path = run / "independent_validation.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SPLIT_SEEDS, default=SPLIT_SEED)
    main(parser.parse_args().seed)
