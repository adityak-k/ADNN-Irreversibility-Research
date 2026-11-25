"""Run fixed CPU baselines on the frozen v3 splits and challenge set."""

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
BASELINES = ["majority", "action_name", "action_text_tfidf", "tfidf", "shuffled_label"]
SPLIT_SEEDS = [20261120, 20261121, 20261122]
SHUFFLE_REPETITIONS = 100
SHUFFLE_SEED_START = 20270000
SHUFFLE_SEED_STRIDE = 1000
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
    """Return only the approved state and candidate-action text fields."""
    projected = build_model_input(record)
    return projected["state_text"] + " " + projected["action_text"]


def action_text(record):
    """Return the action_text field; in this dataset it is a canonical action name."""
    return build_model_input(record)["action_text"]


def deterministic_mode(labels):
    counts = Counter(labels)
    # Ties are resolved by the fixed LABELS order above.
    return max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))


def shuffle_training_labels(labels, shuffle_seed):
    """Return a deterministic permutation with exactly the same class counts."""
    shuffled = list(labels)
    random.Random(shuffle_seed).shuffle(shuffled)
    if Counter(shuffled) != Counter(labels):
        raise AssertionError("A label permutation changed the training class counts")
    return shuffled


def fit_baseline(baseline, training_records, seed, shuffle_seed=None):
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
        # A permutation keeps the training class counts exactly unchanged.
        if shuffle_seed is None:
            shuffle_seed = seed + 900000
        labels_for_fit = shuffle_training_labels(train_labels, shuffle_seed)
    elif baseline not in {"tfidf", "action_text_tfidf"}:
        raise ValueError(f"Unknown baseline: {baseline}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, min_df=1)
    if baseline == "action_text_tfidf":
        training_text = [action_text(record) for record in training_records]
    else:
        training_text = [model_text(record) for record in training_records]
    training_vectors = vectorizer.fit_transform(training_text)
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_SETTINGS)
    classifier.fit(training_vectors, labels_for_fit)
    return {
        "kind": "tfidf",
        "input_type": "action_text" if baseline == "action_text_tfidf" else "state_plus_action",
        "vectorizer": vectorizer,
        "classifier": classifier,
    }


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
    if fitted_model["input_type"] == "action_text":
        texts = [action_text(record) for record in records]
    else:
        texts = [model_text(record) for record in records]
    vectors = fitted_model["vectorizer"].transform(texts)
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


def build_shuffle_seeds():
    """Return the predetermined 100-seed schedule for each frozen split."""
    return {
        split_seed: [
            SHUFFLE_SEED_START + split_index * SHUFFLE_SEED_STRIDE + repetition
            for repetition in range(1, SHUFFLE_REPETITIONS + 1)
        ]
        for split_index, split_seed in enumerate(SPLIT_SEEDS)
    }


def summarize_shuffle_control(all_metrics, real_metrics):
    """Compare each split's real TF-IDF score with its own shuffle distribution."""
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1")
    output = {}
    for seed in SPLIT_SEEDS:
        output[str(seed)] = {}
        for evaluation in ("v3_test", "challenge_all_external"):
            shuffle_runs = [
                run for run in all_metrics
                if run["baseline"] == "shuffled_label"
                and run["seed"] == seed and run["evaluation"] == evaluation
            ]
            real_run = next(
                run for run in real_metrics
                if run["baseline"] == "tfidf"
                and run["seed"] == seed and run["evaluation"] == evaluation
            )
            scores = [run["metrics"]["accuracy"] for run in shuffle_runs]
            output[str(seed)][evaluation] = {
                "repetitions": len(shuffle_runs),
                "real_tfidf": {},
                "shuffled_accuracy": {
                    "mean": sum(scores) / len(scores),
                    "standard_deviation": (
                        sum((value - sum(scores) / len(scores)) ** 2 for value in scores)
                        / len(scores)
                    ) ** 0.5,
                    "minimum": min(scores),
                    "maximum": max(scores),
                    "values": scores,
                },
                "empirical_upper_tail_probability_accuracy": (
                    1 + sum(value >= real_run["metrics"]["accuracy"] for value in scores)
                ) / (len(scores) + 1),
            }
            for metric_name in metric_names:
                shuffle_values = [run["metrics"][metric_name] for run in shuffle_runs]
                mean_value = sum(shuffle_values) / len(shuffle_values)
                real_value = real_run["metrics"][metric_name]
                output[str(seed)][evaluation]["real_tfidf"][metric_name] = real_value
                if metric_name != "accuracy":
                    output[str(seed)][evaluation][f"shuffled_{metric_name}"] = {
                        "mean": mean_value,
                        "standard_deviation": (
                            sum((value - mean_value) ** 2 for value in shuffle_values)
                            / len(shuffle_values)
                        ) ** 0.5,
                        "minimum": min(shuffle_values),
                        "maximum": max(shuffle_values),
                        "values": shuffle_values,
                    }
    return output


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
    shuffle_seed_schedule = build_shuffle_seeds()
    baseline_runs = [name for name in args.baselines if name != "shuffled_label"]
    for seed in args.seeds:
        training_records, testing_records, metadata = load_split(dataset_by_id, split_root, seed)
        if metadata.get("dataset_sha256") != actual_dataset_hash:
            raise ValueError(f"Split {seed} metadata uses a different dataset hash")
        split_counts[str(seed)] = {
            "training": len(training_records),
            "testing": len(testing_records),
        }
        # Fit each ordinary model on this split's training IDs only.
        split_real_metrics = []
        for baseline in baseline_runs:
            model = fit_baseline(baseline, training_records, seed)
            test_predictions = predict(model, testing_records)
            test_metrics = evaluate_records(testing_records, test_predictions)
            test_name = f"{baseline}_seed{seed}_v3_test"
            save_predictions(predictions_root / f"{test_name}.jsonl", testing_records, test_predictions, test_name, baseline)
            test_run = {
                "baseline": baseline,
                "seed": seed,
                "evaluation": "v3_test",
                "prediction_file": f"{test_name}.jsonl",
                "metrics": test_metrics,
            }
            all_metrics.append(test_run)
            split_real_metrics.append(test_run)

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
            challenge_run = {
                "baseline": baseline,
                "seed": seed,
                "evaluation": "challenge_all_external",
                "prediction_file": f"{challenge_name}.jsonl",
                "metrics": challenge_metrics,
            }
            all_metrics.append(challenge_run)
            split_real_metrics.append(challenge_run)

        if "shuffled_label" in args.baselines:
            original_training_labels = [record["label"] for record in training_records]
            original_class_counts = dict(Counter(original_training_labels))
            for repetition, shuffle_seed in enumerate(shuffle_seed_schedule[seed], start=1):
                model = fit_baseline(
                    "shuffled_label", training_records, seed, shuffle_seed=shuffle_seed
                )
                # Test labels and challenge labels are never passed to fit_baseline.
                for evaluation, evaluation_records in (
                    ("v3_test", testing_records),
                    ("challenge_all_external", challenge_records),
                ):
                    predictions = predict(model, evaluation_records)
                    metrics = evaluate_records(evaluation_records, predictions)
                    run_name = (
                        f"shuffled_label_seed{seed}_repeat{repetition:03d}_"
                        f"{evaluation}"
                    )
                    save_predictions(
                        predictions_root / f"{run_name}.jsonl",
                        evaluation_records,
                        predictions,
                        run_name,
                        "shuffled_label",
                    )
                    all_metrics.append({
                        "baseline": "shuffled_label",
                        "seed": seed,
                        "shuffle_repetition": repetition,
                        "shuffle_seed": shuffle_seed,
                        "training_class_counts_before_shuffle": original_class_counts,
                        "training_class_counts_after_shuffle": original_class_counts,
                        "evaluation": evaluation,
                        "prediction_file": f"{run_name}.jsonl",
                        "metrics": metrics,
                    })

    manifest = {
        "experiment": "phase4_cpu_baselines_v2",
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
        "shuffled_label_repetitions_per_split": SHUFFLE_REPETITIONS,
        "shuffled_label_seed_schedule": shuffle_seed_schedule,
        "shuffled_label_policy": (
            "Permute training labels only; this preserves training class counts. "
            "The same split-specific model evaluates against unchanged v3 test labels "
            "and all unchanged challenge labels."
        ),
        "majority_tie_break": LABELS,
        "model_input_fields": ["environment", "state_text", "action_text"],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)
    with (output_root / "metrics_by_split.json").open("w", encoding="utf-8") as output_file:
        json.dump(all_metrics, output_file, indent=2)
    real_runs = [run for run in all_metrics if run["baseline"] != "shuffled_label"]
    summary = {
        "v3_test": summarize_metrics([
            run for run in real_runs if run["evaluation"] == "v3_test"
        ]),
        "challenge_all_external": summarize_metrics([
            run for run in real_runs if run["evaluation"] == "challenge_all_external"
        ]),
        "shuffled_label_by_split": summarize_shuffle_control(
            all_metrics,
            real_runs,
        ) if "shuffled_label" in args.baselines and "tfidf" in args.baselines else {},
        "interpretation_note": (
            "The three frozen splits overlap and are not independent dataset replications. "
            "Shuffle distributions are reported separately by split."
        ),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(
        f"Completed {len(args.baselines)} baselines across {len(args.seeds)} frozen splits; "
        f"{SHUFFLE_REPETITIONS if 'shuffled_label' in args.baselines else 0} shuffle repetitions per split."
    )
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
    parser.add_argument("--output-dir", default="results/baselines_v2")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_arguments())
