"""Recheck all frozen-split probe outputs and save a three-seed summary."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "final_sprint"
DATASET = ROOT / "dataset" / "irreversibility_dataset_v3.jsonl"
SPLIT_ROOT = ROOT / "dataset" / "experimental_splits_v1"
LABELS = ("reversible", "partially_reversible", "irreversible")
SEEDS = (20261120, 20261121, 20261122)
LAYERS = 29
HIDDEN_SIZE = 1536
RESULT_FILES = (
    "rq1_three_seed_metrics.csv",
    "rq1_three_seed_summary.csv",
    "rq1_three_seed_summary.json",
    "rq1_three_seed_independent_validation.json",
    "rq1_three_seed_layer_comparison.svg",
    "rq2_seed_20261120_identity_check.json",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_metrics(truth, predicted):
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for actual, estimate in zip(truth, predicted):
        matrix[LABELS.index(actual)][LABELS.index(estimate)] += 1
    recalls = [matrix[i][i] / sum(matrix[i]) for i in range(len(LABELS))]
    precisions = [
        matrix[i][i] / sum(matrix[row][i] for row in range(len(LABELS)))
        if sum(matrix[row][i] for row in range(len(LABELS))) else 0.0
        for i in range(len(LABELS))
    ]
    f1_scores = [2 * p * r / (p + r) if p + r else 0.0
                 for p, r in zip(precisions, recalls)]
    return {
        "count": len(truth),
        "accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / len(truth),
        "balanced_accuracy": sum(recalls) / len(LABELS),
        "macro_precision": sum(precisions) / len(LABELS),
        "macro_recall": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1_scores) / len(LABELS),
        "confusion_matrix": matrix,
    }


def compare_saved_metrics(calculated, saved, context):
    for key, value in calculated.items():
        other = saved.get(key)
        if isinstance(value, float):
            if not isinstance(other, (int, float)) or not math.isclose(value, other, abs_tol=1e-10):
                raise AssertionError(f"{context}: {key} mismatch: recalculated {value}, saved {other}")
        elif other != value:
            raise AssertionError(f"{context}: {key} mismatch: recalculated {value}, saved {other}")


def check_shards(folder, expected_rows):
    expected = {row["id"]: row["label"] for row in expected_rows}
    observed = {}
    shard_count = 0
    for shard_path in sorted(folder.glob("*.npz")):
        shard_count += 1
        with np.load(shard_path, allow_pickle=False) as shard:
            ids = shard["example_ids"].astype(str).tolist()
            labels = shard["labels"].astype(str).tolist()
            hidden = shard["hidden_states"]
        if hidden.shape != (len(ids), LAYERS, HIDDEN_SIZE):
            raise AssertionError(f"{shard_path}: invalid tensor shape {hidden.shape}")
        if hidden.dtype != np.float16 or len(labels) != len(ids):
            raise AssertionError(f"{shard_path}: invalid dtype or ID/label lengths")
        for example_id, label in zip(ids, labels):
            if example_id not in expected or expected[example_id] != label:
                raise AssertionError(f"{shard_path}: unknown ID or mismatched label: {example_id}")
            if example_id in observed:
                raise AssertionError(f"Duplicate representation ID: {example_id}")
            observed[example_id] = label
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise AssertionError(f"Representation coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    return shard_count


def check_one_seed(seed, by_id, dataset_hash, split_manifest):
    split_dir = SPLIT_ROOT / f"split_seed_{seed}"
    metadata = read_json(split_dir / "metadata.json")
    train_ids = read_json(split_dir / "training_example_ids.json")
    test_ids = read_json(split_dir / "testing_example_ids.json")
    if seed not in split_manifest.get("seeds", []):
        raise AssertionError(f"Seed {seed} is not listed in the original split manifest")
    if (metadata.get("seed") != seed or metadata.get("dataset_sha256") != dataset_hash
            or split_manifest.get("dataset_sha256") != dataset_hash):
        raise AssertionError(f"Seed {seed}: frozen metadata or dataset hash mismatch")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise AssertionError(f"Seed {seed}: duplicate IDs in frozen partition")
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(by_id):
        raise AssertionError(f"Seed {seed}: frozen IDs do not partition v3 exactly")
    if any(example_id not in by_id for example_id in train_ids + test_ids):
        raise AssertionError(f"Seed {seed}: frozen split references unknown v3 ID")

    train_rows = [by_id[item] for item in train_ids]
    test_rows = [by_id[item] for item in test_ids]
    train_labels = Counter(row["label"] for row in train_rows)
    test_labels = Counter(row["label"] for row in test_rows)
    if len(train_rows) != 1920 or len(test_rows) != 480:
        raise AssertionError(f"Seed {seed}: expected 1920 train / 480 test")
    if train_labels != Counter({label: 640 for label in LABELS}):
        raise AssertionError(f"Seed {seed}: training class counts changed: {train_labels}")
    if test_labels != Counter({label: 160 for label in LABELS}):
        raise AssertionError(f"Seed {seed}: test class counts changed: {test_labels}")

    train_groups = {row["canonical_group_id"] for row in train_rows}
    test_groups = {row["canonical_group_id"] for row in test_rows}
    if train_groups & test_groups:
        raise AssertionError(f"Seed {seed}: canonical group crosses split")

    run_dir = OUTPUT / f"rq1_seed_{seed}"
    config = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete" or config.get("split_seed") != seed:
        raise AssertionError(f"Seed {seed}: experiment is not marked complete for this seed")
    if manifest.get("dataset_sha256") != dataset_hash or config.get("dataset_sha256") != dataset_hash:
        raise AssertionError(f"Seed {seed}: experiment dataset hash mismatch")
    if manifest.get("train_examples_extracted_and_probed") != len(train_rows):
        raise AssertionError(f"Seed {seed}: train completion count mismatch")
    if manifest.get("test_examples_extracted_and_evaluated") != len(test_rows):
        raise AssertionError(f"Seed {seed}: test completion count mismatch")

    train_shards = check_shards(run_dir / "representations" / "train", train_rows)
    test_shards = check_shards(run_dir / "representations" / "test", test_rows)
    aggregate_metrics = read_json(run_dir / "layer_metrics.json")
    if len(aggregate_metrics) != LAYERS:
        raise AssertionError(f"Seed {seed}: expected 29 aggregate metric rows")

    layer_records = []
    for layer in range(LAYERS):
        prediction_path = run_dir / "layer_results" / f"layer_{layer:02d}_predictions.jsonl"
        metric_path = run_dir / "layer_results" / f"layer_{layer:02d}_metrics.json"
        if not prediction_path.is_file() or not metric_path.is_file():
            raise AssertionError(f"Seed {seed}: missing prediction/metric file for layer {layer}")
        predictions = read_jsonl(prediction_path)
        ids = [item.get("id") for item in predictions]
        if len(ids) != 480 or len(set(ids)) != 480 or set(ids) != set(test_ids):
            raise AssertionError(f"Seed {seed}, layer {layer}: IDs missing, duplicated or unexpected")
        truth = []
        predicted = []
        for item in predictions:
            if set(item) != {"id", "true_label", "predicted_label", "layer"}:
                raise AssertionError(f"Seed {seed}, layer {layer}: unexpected prediction fields")
            if item["true_label"] != by_id[item["id"]]["label"]:
                raise AssertionError(f"Seed {seed}, layer {layer}: wrong true label for {item['id']}")
            if item["layer"] != layer or item["predicted_label"] not in LABELS:
                raise AssertionError(f"Seed {seed}, layer {layer}: invalid prediction record")
            truth.append(item["true_label"])
            predicted.append(item["predicted_label"])

        metrics = calculate_metrics(truth, predicted)
        saved_layer = read_json(metric_path)
        compare_saved_metrics(metrics, saved_layer, f"seed {seed} layer {layer} metrics")
        compare_saved_metrics(metrics, aggregate_metrics[layer], f"seed {seed} layer {layer} summary")
        layer_records.append({"seed": seed, "layer": layer, **metrics})

    return {
        "seed": seed,
        "training_count": len(train_ids),
        "testing_count": len(test_ids),
        "training_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "testing_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "train_representation_shards": train_shards,
        "test_representation_shards": test_shards,
        "prediction_files": LAYERS,
        "prediction_records_verified": len(test_ids) * LAYERS,
        "layers": layer_records,
    }


def verify_rq2_identity(by_id):
    run = OUTPUT / "rq2_direct_answers"
    config = read_json(run / "config.json")
    records = read_jsonl(run / "predictions.jsonl")
    if config.get("split_seed") != 20261120 or config.get("subset_seed") != 20262025:
        raise AssertionError("RQ2 is not bound to its documented seed and subset seed")
    if len(records) != 120 or len({item["example_id"] for item in records}) != 120:
        raise AssertionError("RQ2 does not contain 120 unique IDs")

    split_dir = SPLIT_ROOT / "split_seed_20261120"
    test_ids = read_json(split_dir / "testing_example_ids.json")
    test_rows = [by_id[item] for item in test_ids]
    rng = np.random.default_rng(config["subset_seed"])
    expected = []
    actions = sorted({row["action_text"] for row in test_rows})
    for action in actions:
        for label in LABELS:
            cell = [row for row in test_rows
                    if row["action_text"] == action and row["label"] == label]
            chosen = rng.choice(len(cell), size=10, replace=False)
            expected.extend(cell[int(index)]["id"] for index in chosen)
    actual = [item["example_id"] for item in records]
    if actual != expected:
        raise AssertionError("RQ2 IDs differ from the exact seeded 120-example selection")
    if any(item["example_id"] not in set(test_ids) for item in records):
        raise AssertionError("RQ2 contains a non-test example")
    for item in records:
        row = by_id[item["example_id"]]
        if item["true_label"] != row["label"] or item["action_type"] != row["action_text"]:
            raise AssertionError(f"RQ2 truth/action mismatch for {item['example_id']}")
    cell_counts = Counter((item["action_type"], item["true_label"]) for item in records)
    if len(cell_counts) != 12 or any(value != 10 for value in cell_counts.values()):
        raise AssertionError("RQ2 subset is not balanced at 10 per action/class cell")
    return {
        "status": "passed",
        "split_seed": 20261120,
        "subset_seed": config["subset_seed"],
        "count": len(actual),
        "exact_ordered_example_ids": actual,
        "action_class_counts": {f"{action}/{label}": cell_counts[(action, label)]
                                for action in actions for label in LABELS},
        "repeated_across_other_splits": False,
    }


def write_comparison_svg(summary):
    width, height = 1200, 760
    left, plot_width = 95, 1035
    x_step = plot_width / 28
    panels = [
        ("Balanced accuracy by layer (mean +/- population SD)", "mean_balanced_accuracy", "std_balanced_accuracy", 90, 260),
        ("Macro-F1 by layer (mean +/- population SD)", "mean_macro_f1", "std_macro_f1", 420, 260),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#18212b}.axis{stroke:#59636e;stroke-width:1}.grid{stroke:#d8dde3;stroke-width:1}.score{fill:none;stroke:#1769aa;stroke-width:3}.err{stroke:#1769aa;stroke-width:1.5}</style>',
        '<text x="600" y="32" text-anchor="middle" font-size="22" font-weight="bold">Qwen2.5-1.5B-Instruct linear probes across three frozen splits</text>',
    ]
    for title, mean_key, std_key, top, panel_height in panels:
        bottom = top + panel_height
        parts.append(f'<text x="600" y="{top - 24}" text-anchor="middle" font-size="17">{title}</text>')
        for tick in range(0, 6):
            value = tick / 5
            y = bottom - value * panel_height
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}"/>')
        points = []
        for record in summary:
            x = left + record["layer"] * x_step
            mean = record[mean_key]
            deviation = record[std_key]
            y = bottom - mean * panel_height
            y_high = bottom - min(1.0, mean + deviation) * panel_height
            y_low = bottom - max(0.0, mean - deviation) * panel_height
            parts.append(f'<line class="err" x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}"/>')
            parts.append(f'<line class="err" x1="{x - 3:.1f}" y1="{y_high:.1f}" x2="{x + 3:.1f}" y2="{y_high:.1f}"/>')
            parts.append(f'<line class="err" x1="{x - 3:.1f}" y1="{y_low:.1f}" x2="{x + 3:.1f}" y2="{y_low:.1f}"/>')
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#1769aa"/>')
            if record["layer"] % 2 == 0 or record["layer"] == 28:
                parts.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="10">{record["layer"]}</text>')
        parts.append(f'<polyline class="score" points="{" ".join(points)}"/>')
        parts.append(f'<text x="600" y="{bottom + 38}" text-anchor="middle" font-size="13">Hidden-state output (0 = embedding, 1–28 = transformer blocks)</text>')
    parts.append('<text x="18" y="380" transform="rotate(-90 18 380)" text-anchor="middle" font-size="13">Score</text>')
    parts.append('<text x="600" y="748" text-anchor="middle" font-size="11">Error bars show descriptive population SD over three overlapping frozen splits; they are not independent replications.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main():
    if any((OUTPUT / name).exists() for name in RESULT_FILES):
        existing = [name for name in RESULT_FILES if (OUTPUT / name).exists()]
        raise FileExistsError(f"Refusing to overwrite summary outputs: {existing}")

    dataset_hash = file_sha256(DATASET)
    rows = read_jsonl(DATASET)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != 2400:
        raise AssertionError("v3 dataset must have 2400 unique IDs")
    split_manifest = read_json(SPLIT_ROOT / "manifest.json")
    if split_manifest.get("dataset_sha256") != dataset_hash:
        raise AssertionError("Current v3 hash disagrees with the original split manifest")
    if tuple(split_manifest.get("seeds", [])) != SEEDS:
        raise AssertionError("Original split manifest does not list the three expected seeds")

    seed_results = [check_one_seed(seed, by_id, dataset_hash, split_manifest) for seed in SEEDS]
    per_seed_rows = [metric for result in seed_results for metric in result["layers"]]
    aggregate = []
    for layer in range(LAYERS):
        layer_rows = [item for item in per_seed_rows if item["layer"] == layer]
        ba_values = [item["balanced_accuracy"] for item in layer_rows]
        f1_values = [item["macro_f1"] for item in layer_rows]
        aggregate.append({
            "layer": layer,
            "mean_balanced_accuracy": statistics.mean(ba_values),
            "std_balanced_accuracy": statistics.pstdev(ba_values),
            "mean_macro_f1": statistics.mean(f1_values),
            "std_macro_f1": statistics.pstdev(f1_values),
            "seed_count": len(layer_rows),
        })
    best_layer = max(aggregate, key=lambda item: (item["mean_balanced_accuracy"], -item["layer"]))
    rq2_check = verify_rq2_identity(by_id)

    independent_report = {
        "status": "passed",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "frozen_split_manifest": str((SPLIT_ROOT / "manifest.json").relative_to(ROOT)),
        "seeds": seed_results,
        "rq2_identity_check": rq2_check,
        "checks": [
            "frozen train/test IDs match the original per-seed ID files and partition all 2400 v3 IDs once",
            "all train/test representation shards contain expected IDs and labels exactly once in FP16 29x1536 arrays",
            "all 29 prediction files per seed cover each of 480 test IDs exactly once",
            "saved ground-truth labels match the v3 source rows",
            "balanced accuracy, macro-F1 and confusion matrices were recalculated manually from saved predictions",
            "recalculated metrics match both per-layer metrics files and aggregate layer_metrics.json",
            "RQ2 contains only its exact seeded 120 IDs from split 20261120",
        ],
        "limitations": [
            "The three frozen splits overlap and must not be presented as independent dataset replications.",
            "The summary describes performance on existing v3 test wording; it is not a wording-held-out test.",
        ],
    }
    full_summary = {
        "status": "passed",
        "dataset_sha256": dataset_hash,
        "split_seeds": list(SEEDS),
        "primary_best_layer": best_layer["layer"],
        "best_layer_mean_balanced_accuracy": best_layer["mean_balanced_accuracy"],
        "best_layer_mean_macro_f1": best_layer["mean_macro_f1"],
        "standard_deviation_definition": "population standard deviation across the three specified overlapping frozen splits",
        "per_seed_layer_metrics": per_seed_rows,
        "across_seed_layer_summary": aggregate,
        "rq2_seed_20261120_only": rq2_check,
        "no_rq2_repetition_claimed": True,
    }

    with (OUTPUT / "rq1_three_seed_metrics.csv").open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "seed", "layer", "balanced_accuracy", "macro_f1", "confusion_matrix_json"
        ])
        writer.writeheader()
        for item in per_seed_rows:
            writer.writerow({
                "seed": item["seed"],
                "layer": item["layer"],
                "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"],
                "confusion_matrix_json": json.dumps(item["confusion_matrix"], separators=(",", ":")),
            })
    with (OUTPUT / "rq1_three_seed_summary.csv").open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (OUTPUT / "rq1_three_seed_summary.json").write_text(
        json.dumps(full_summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "rq1_three_seed_independent_validation.json").write_text(
        json.dumps(independent_report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "rq1_three_seed_layer_comparison.svg").write_text(
        write_comparison_svg(aggregate), encoding="utf-8"
    )
    (OUTPUT / "rq2_seed_20261120_identity_check.json").write_text(
        json.dumps(rq2_check, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "validation_status": "passed",
        "seed_summaries": [{
            "seed": result["seed"],
            "prediction_records_verified": result["prediction_records_verified"],
            "layer0_balanced_accuracy": result["layers"][0]["balanced_accuracy"],
            "layer28_balanced_accuracy": result["layers"][28]["balanced_accuracy"],
            "layer28_macro_f1": result["layers"][28]["macro_f1"],
        } for result in seed_results],
        "best_layer_by_mean_balanced_accuracy": best_layer,
        "rq2_seed": rq2_check["split_seed"],
        "rq2_examples": rq2_check["count"],
        "outputs": list(RESULT_FILES),
    }, indent=2))


if __name__ == "__main__":
    main()
