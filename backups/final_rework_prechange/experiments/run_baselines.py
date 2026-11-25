"""Run simple CPU baselines on the frozen v3 splits and challenge set."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.model_input import build_model_input  # noqa: E402


LABELS = ["reversible", "partially_reversible", "irreversible"]
BASELINES = ["majority", "action_name", "tfidf", "shuffled_label"]
SPLIT_SEEDS = [20261120, 20261121, 20261122]
SHUFFLE_SEED_OFFSET = 900000
LOGISTIC_REGRESSION_SETTINGS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 0,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    return records


def validate_dataset(records, dataset_hash, expected_hash, name):
    if dataset_hash != expected_hash:
        raise ValueError(f"{name} hash does not match its frozen manifest")
    ids = [record.get("id") for record in records]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError(f"{name} has a missing or duplicate example ID")
    for record in records:
        if record.get("label") not in LABELS:
            raise ValueError(f"{name} contains an unknown label for {record.get('id')}")
        # The approved projection is the only route from a row to model inputs.
        projected = build_model_input(record)
        if set(projected) != {"environment", "state_text", "action_text"}:
            raise ValueError("Unexpected model input projection fields")
        if not isinstance(projected["state_text"], str) or not isinstance(projected["action_text"], str):
            raise ValueError(f"Text model input is invalid for {record['id']}")


def load_split(dataset_by_id, split_root, seed):
    split_directory = split_root / f"split_seed_{seed}"
    training_ids = read_json(split_directory / "training_example_ids.json")
    testing_ids = read_json(split_directory / "testing_example_ids.json")
    metadata = read_json(split_directory / "metadata.json")

    if metadata.get("seed") != seed:
        raise ValueError(f"Split seed mismatch in {split_directory}")
    if len(training_ids) != len(set(training_ids)) or len(testing_ids) != len(set(testing_ids)):
        raise ValueError(f"Duplicate IDs in split {seed}")
    if set(training_ids) & set(testing_ids):
        raise ValueError(f"Train/test IDs overlap in split {seed}")
    if set(training_ids) | set(testing_ids) != set(dataset_by_id):
        raise ValueError(f"Split {seed} does not cover the v3 dataset exactly")

    training_records = [dataset_by_id[example_id] for example_id in training_ids]
    testing_records = [dataset_by_id[example_id] for example_id in testing_ids]
    train_groups = {record["canonical_group_id"] for record in training_records}
    test_groups = {record["canonical_group_id"] for record in testing_records}
    if train_groups & test_groups:
        raise ValueError(f"Canonical group crosses split {seed}")
    for field in ("scenario_group_id", "input_group_id", "pair_id"):
        train_values = {
            record[field] for record in training_records if record.get(field)
        }
        test_values = {
            record[field] for record in testing_records if record.get(field)
        }
        if train_values & test_values:
            raise ValueError(f"{field} crosses split {seed}")
    training_inputs = {
        tuple(build_model_input(record)[field] for field in ("environment", "state_text", "action_text"))
        for record in training_records
    }
    testing_inputs = {
        tuple(build_model_input(record)[field] for field in ("environment", "state_text", "action_text"))
        for record in testing_records
    }
    if training_inputs & testing_inputs:
        raise ValueError(f"Identical model input crosses split {seed}")
    return training_records, testing_records, metadata


def model_text(record):
    projected = build_model_input(record)
    return projected["state_text"] + " " + projected["action_text"]


def deterministic_mode(labels):
    counts = Counter(labels)
    # Ties are resolved by the fixed LABELS order above.
    return max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))


def fit_baseline(baseline, training_records, seed):
    train_labels = [record["label"] for record in training_records]

    if baseline == "majority":
        return {"kind": "constant", "label": deterministic_mode(train_labels)}

    if baseline == "action_name":
        labels_by_action = defaultdict(list)
        for record in training_records:
            action_name = build_model_input(record)["action_text"]
            labels_by_action[action_name].append(record["label"])
        action_labels = {
            action_name: deterministic_mode(labels)
            for action_name, labels in labels_by_action.items()
        }
        return {
            "kind": "action_name",
            "action_labels": action_labels,
            "fallback_label": deterministic_mode(train_labels),
        }

    labels_for_fit = list(train_labels)
    if baseline == "shuffled_label":
        # Only training labels are shuffled. Test and challenge labels stay intact.
        shuffle_seed = seed + SHUFFLE_SEED_OFFSET
        random.Random(shuffle_seed).shuffle(labels_for_fit)
    elif baseline != "tfidf":
        raise ValueError(f"Unknown baseline: {baseline}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, min_df=1)
    training_text = [model_text(record) for record in training_records]
    training_vectors = vectorizer.fit_transform(training_text)
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_SETTINGS)
    classifier.fit(training_vectors, labels_for_fit)
    return {"kind": "tfidf", "vectorizer": vectorizer, "classifier": classifier}


def predict(fitted_model, records):
    if fitted_model["kind"] == "constant":
        return [fitted_model["label"] for _ in records]
    if fitted_model["kind"] == "action_name":
        return [
            fitted_model["action_labels"].get(
                build_model_input(record)["action_text"], fitted_model["fallback_label"]
            )
            for record in records
        ]
    vectors = fitted_model["vectorizer"].transform([model_text(record) for record in records])
    return fitted_model["classifier"].predict(vectors).tolist()


def evaluate_records(records, predictions):
    true_labels = [record["label"] for record in records]
    metrics = {
        "count": len(records),
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "macro_precision": precision_score(true_labels, predictions, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(true_labels, predictions, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(true_labels, predictions, labels=LABELS, average="macro", zero_division=0),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=LABELS).tolist(),
        "per_class": {},
        "per_action": {},
    }

    for label in LABELS:
        metrics["per_class"][label] = {
            "precision": precision_score(true_labels, predictions, labels=[label], average="micro", zero_division=0),
            "recall": recall_score(true_labels, predictions, labels=[label], average="micro", zero_division=0),
            "f1": f1_score(true_labels, predictions, labels=[label], average="micro", zero_division=0),
        }

    for action_name in sorted({build_model_input(record)["action_text"] for record in records}):
        selected = [
            (record, prediction)
            for record, prediction in zip(records, predictions)
            if build_model_input(record)["action_text"] == action_name
        ]
        action_true = [record["label"] for record, _ in selected]
        action_predicted = [prediction for _, prediction in selected]
        metrics["per_action"][action_name] = {
            "count": len(selected),
            "accuracy": accuracy_score(action_true, action_predicted),
            "balanced_accuracy": balanced_accuracy_score(action_true, action_predicted),
            "macro_precision": precision_score(action_true, action_predicted, labels=LABELS, average="macro", zero_division=0),
            "macro_recall": recall_score(action_true, action_predicted, labels=LABELS, average="macro", zero_division=0),
            "macro_f1": f1_score(action_true, action_predicted, labels=LABELS, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(action_true, action_predicted, labels=LABELS).tolist(),
        }
    return metrics


def save_predictions(path, records, predictions, split_name, baseline):
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record, prediction in zip(records, predictions):
            output = {
                "id": record["id"],
                "true_label": record["label"],
                "predicted_label": prediction,
                "action": build_model_input(record)["action_text"],
                "split": split_name,
                "baseline": baseline,
            }
            output_file.write(json.dumps(output, sort_keys=True) + "\n")


def summarize_metrics(metrics_by_run):
    metric_names = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    grouped = defaultdict(list)
    for run in metrics_by_run:
        grouped[run["baseline"]].append(run["metrics"])
    summary = {}
    for baseline, run_metrics in grouped.items():
        summary[baseline] = {}
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in run_metrics]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[baseline][metric_name] = {
                "mean": mean,
                "standard_deviation": variance ** 0.5,
            }
    return summary


def run_experiments(args):
    dataset_path = Path(args.dataset)
    split_root = Path(args.split_root)
    challenge_path = Path(args.challenge_dataset)
    output_root = Path(args.output_dir)
    split_manifest = read_json(split_root / "manifest.json")
    challenge_manifest = read_json(split_root.parent / "experimental_challenge_v1" / "manifest.json")

    actual_dataset_hash = sha256_file(dataset_path)
    expected_dataset_hash = split_manifest["dataset_sha256"]
    v3_records = read_jsonl(dataset_path)
    validate_dataset(v3_records, actual_dataset_hash, expected_dataset_hash, "v3 dataset")
    dataset_by_id = {record["id"]: record for record in v3_records}

    actual_challenge_hash = sha256_file(challenge_path)
    challenge_records = read_jsonl(challenge_path)
    validate_dataset(
        challenge_records,
        actual_challenge_hash,
        challenge_manifest["dataset_sha256"],
        "challenge dataset",
    )
    if len(challenge_records) != 240:
        raise ValueError("Expected exactly 240 challenge examples")
    if args.seeds != SPLIT_SEEDS:
        raise ValueError(f"The approved v3 split seeds are {SPLIT_SEEDS}")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_root = output_root / "predictions"
    predictions_root.mkdir()

    all_metrics = []
    split_counts = {}
    for seed in args.seeds:
        training_records, testing_records, metadata = load_split(dataset_by_id, split_root, seed)
        if metadata.get("dataset_sha256") != actual_dataset_hash:
            raise ValueError(f"Split {seed} metadata uses a different dataset hash")
        split_counts[str(seed)] = {
            "training": len(training_records),
            "testing": len(testing_records),
        }
        for baseline in args.baselines:
            model = fit_baseline(baseline, training_records, seed)
            test_predictions = predict(model, testing_records)
            test_metrics = evaluate_records(testing_records, test_predictions)
            test_name = f"{baseline}_seed{seed}_v3_test"
            save_predictions(predictions_root / f"{test_name}.jsonl", testing_records, test_predictions, test_name, baseline)
            all_metrics.append({"baseline": baseline, "seed": seed, "evaluation": "v3_test", "metrics": test_metrics})

            # Reuse the model fitted only on v3 training rows for all 240 challenge rows.
            challenge_predictions = predict(model, challenge_records)
            challenge_metrics = evaluate_records(challenge_records, challenge_predictions)
            challenge_name = f"{baseline}_seed{seed}_challenge_all"
            save_predictions(
                predictions_root / f"{challenge_name}.jsonl",
                challenge_records,
                challenge_predictions,
                challenge_name,
                baseline,
            )
            all_metrics.append({
                "baseline": baseline,
                "seed": seed,
                "evaluation": "challenge_all_external",
                "metrics": challenge_metrics,
            })

    manifest = {
        "experiment": "phase4_cpu_baselines_v1",
        "v3_dataset_path": str(dataset_path),
        "v3_dataset_sha256": actual_dataset_hash,
        "challenge_dataset_path": str(challenge_path),
        "challenge_dataset_sha256": actual_challenge_hash,
        "split_manifest_path": str(split_root / "manifest.json"),
        "seeds": args.seeds,
        "baselines": args.baselines,
        "split_counts": split_counts,
        "external_challenge_training_policy": "v3 training IDs only; all 240 challenge examples are evaluation-only",
        "tfidf": {"ngram_range": [1, 1], "lowercase": True, "min_df": 1},
        "logistic_regression": LOGISTIC_REGRESSION_SETTINGS,
        "shuffled_label_seed_offset": SHUFFLE_SEED_OFFSET,
        "majority_tie_break": LABELS,
        "model_input_fields": ["environment", "state_text", "action_text"],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)
    with (output_root / "metrics_by_split.json").open("w", encoding="utf-8") as output_file:
        json.dump(all_metrics, output_file, indent=2)
    summary = {
        "v3_test": summarize_metrics([run for run in all_metrics if run["evaluation"] == "v3_test"]),
        "challenge_all_external": summarize_metrics([
            run for run in all_metrics if run["evaluation"] == "challenge_all_external"
        ]),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(f"Completed {len(args.baselines)} baselines across {len(args.seeds)} frozen splits.")
    print(f"Results: {output_root.resolve()}")
    for run in all_metrics:
        print(
            f"{run['baseline']} seed={run['seed']} {run['evaluation']}: "
            f"accuracy={run['metrics']['accuracy']:.4f} macro_f1={run['metrics']['macro_f1']:.4f}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fixed CPU baselines on frozen v3 splits")
    parser.add_argument("--baseline", dest="baselines", nargs="+", choices=BASELINES, default=BASELINES)
    parser.add_argument("--dataset", default="dataset/irreversibility_dataset_v3.jsonl")
    parser.add_argument("--split-root", default="dataset/experimental_splits_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=SPLIT_SEEDS)
    parser.add_argument("--challenge-dataset", default="dataset/irreversibility_challenge_v1.jsonl")
    parser.add_argument("--output-dir", default="results/baselines_v1")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_arguments())
