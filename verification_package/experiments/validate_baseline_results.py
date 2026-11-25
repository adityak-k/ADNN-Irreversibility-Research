"""Independently check baseline predictions, IDs, metrics, and shuffle runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("reversible", "partially_reversible", "irreversible")
ACTIONS = ("edit_file", "delete_file", "send_email", "transfer_money")
SPLIT_SEEDS = (20261120, 20261121, 20261122)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _class_metrics(true_values, predicted_values):
    results = {}
    recalls = []
    precisions = []
    f1_values = []
    for label in LABELS:
        true_positive = sum(actual == label and predicted == label
                            for actual, predicted in zip(true_values, predicted_values))
        false_positive = sum(actual != label and predicted == label
                             for actual, predicted in zip(true_values, predicted_values))
        false_negative = sum(actual == label and predicted != label
                             for actual, predicted in zip(true_values, predicted_values))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[label] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    return results, precisions, recalls, f1_values


def _calculate_metrics(records, predictions):
    true_values = [row["label"] for row in records]
    correct = sum(actual == predicted for actual, predicted in zip(true_values, predictions))
    per_class, precisions, recalls, f1_values = _class_metrics(true_values, predictions)
    confusion = [[
        sum(actual == true_label and predicted == predicted_label
            for actual, predicted in zip(true_values, predictions))
        for predicted_label in LABELS
    ] for true_label in LABELS]
    per_action = {}
    for action in sorted({row["action_text"] for row in records}):
        selected = [(row, prediction) for row, prediction in zip(records, predictions)
                    if row["action_text"] == action]
        action_true = [row["label"] for row, _ in selected]
        action_predicted = [prediction for _, prediction in selected]
        action_class, action_p, action_r, action_f = _class_metrics(action_true, action_predicted)
        action_confusion = [[
            sum(actual == true_label and predicted == predicted_label
                for actual, predicted in zip(action_true, action_predicted))
            for predicted_label in LABELS
        ] for true_label in LABELS]
        per_action[action] = {
            "count": len(selected),
            "accuracy": sum(actual == predicted for actual, predicted in zip(action_true, action_predicted)) / len(selected),
            "balanced_accuracy": sum(action_r) / len(action_r),
            "macro_precision": sum(action_p) / len(action_p),
            "macro_recall": sum(action_r) / len(action_r),
            "macro_f1": sum(action_f) / len(action_f),
            "confusion_matrix": action_confusion,
        }
    return {
        "count": len(records),
        "accuracy": correct / len(records),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "labels": list(LABELS),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "per_action": per_action,
    }


def _compare(saved, calculated, location="metrics"):
    errors = []
    if isinstance(calculated, dict):
        if not isinstance(saved, dict) or set(saved) != set(calculated):
            return [f"{location}: saved metric keys differ from recomputed keys"]
        for key in calculated:
            errors.extend(_compare(saved[key], calculated[key], f"{location}.{key}"))
    elif isinstance(calculated, list):
        if saved != calculated:
            errors.append(f"{location}: saved list differs from recomputed values")
    elif isinstance(calculated, (int, float)) and not isinstance(calculated, bool):
        if not isinstance(saved, (int, float)) or abs(saved - calculated) > 1e-12:
            errors.append(f"{location}: saved value {saved} differs from {calculated}")
    elif saved != calculated:
        errors.append(f"{location}: saved value differs from recomputed value")
    return errors


def validate_results(output_dir, dataset_path=None, split_root=None, challenge_path=None):
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path or ROOT / "dataset" / "irreversibility_dataset_v3.jsonl")
    split_root = Path(split_root or ROOT / "dataset" / "experimental_splits_v1")
    challenge_path = Path(challenge_path or ROOT / "dataset" / "irreversibility_challenge_v1.jsonl")
    manifest = _read_json(output_dir / "manifest.json")
    metric_runs = _read_json(output_dir / "metrics_by_split.json")
    v3_rows = _read_jsonl(dataset_path)
    challenge_rows = _read_jsonl(challenge_path)
    errors = []

    if manifest.get("v3_dataset_sha256") != _sha256(dataset_path):
        errors.append("The v3 dataset hash does not match the baseline manifest")
    if manifest.get("challenge_dataset_sha256") != _sha256(challenge_path):
        errors.append("The challenge dataset hash does not match the baseline manifest")
    if manifest.get("external_challenge_training_policy") != (
            "v3 training IDs only; all 240 challenge examples are evaluation-only"):
        errors.append("The external challenge training policy is not the approved one")
    if set(row["id"] for row in v3_rows) & set(row["id"] for row in challenge_rows):
        errors.append("v3 and challenge example IDs overlap")

    v3_by_id = {row["id"]: row for row in v3_rows}
    challenge_by_id = {row["id"]: row for row in challenge_rows}
    expected_prediction_names = set()
    split_training_counts = {}
    split_evaluations = {}
    for seed in SPLIT_SEEDS:
        folder = split_root / f"split_seed_{seed}"
        train_ids = _read_json(folder / "training_example_ids.json")
        test_ids = _read_json(folder / "testing_example_ids.json")
        split_metadata = _read_json(folder / "metadata.json")
        if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(v3_by_id):
            errors.append(f"Split {seed} does not partition v3 IDs exactly")
        if split_metadata.get("dataset_sha256") != _sha256(dataset_path):
            errors.append(f"Split {seed} metadata uses a different dataset hash")
        train_rows = [v3_by_id[item] for item in train_ids]
        test_rows = [v3_by_id[item] for item in test_ids]
        for field in ("canonical_group_id", "scenario_group_id", "input_group_id", "pair_id"):
            train_groups = {row.get(field) for row in train_rows if row.get(field)}
            test_groups = {row.get(field) for row in test_rows if row.get(field)}
            if train_groups & test_groups:
                errors.append(f"{field} crosses the frozen split {seed}")
        split_training_counts[seed] = Counter(v3_by_id[item]["label"] for item in train_ids)
        split_evaluations[seed] = {
            "v3_test": [v3_by_id[item] for item in test_ids],
            "challenge_all_external": challenge_rows,
        }

    shuffle_seeds = {
        (seed, evaluation): set()
        for seed in SPLIT_SEEDS
        for evaluation in ("v3_test", "challenge_all_external")
    }
    repetitions = Counter()
    repetition_numbers = {key: set() for key in shuffle_seeds}
    ordinary_run_counts = Counter()
    for run in metric_runs:
        filename = run.get("prediction_file")
        if not filename or Path(filename).name != filename:
            errors.append("A run has an invalid prediction filename")
            continue
        expected_prediction_names.add(filename)
        prediction_path = output_dir / "predictions" / filename
        if not prediction_path.is_file():
            errors.append(f"Missing prediction file: {filename}")
            continue
        predictions = _read_jsonl(prediction_path)
        seed = run.get("seed")
        evaluation = run.get("evaluation")
        if seed not in split_evaluations or evaluation not in split_evaluations.get(seed, {}):
            errors.append(f"Unexpected split/evaluation in {filename}")
            continue
        expected_rows = split_evaluations[seed][evaluation]
        if run.get("baseline") != "shuffled_label":
            ordinary_run_counts[(run.get("baseline"), seed, evaluation)] += 1
        expected_by_id = {row["id"]: row for row in expected_rows}
        observed_ids = [row.get("id") for row in predictions]
        if len(observed_ids) != len(set(observed_ids)):
            errors.append(f"Duplicate prediction IDs in {filename}")
        if set(observed_ids) != set(expected_by_id):
            errors.append(f"Example ID coverage is wrong in {filename}")
            continue
        aligned_rows = []
        predicted_labels = []
        for prediction in predictions:
            source = expected_by_id[prediction["id"]]
            if prediction.get("true_label") != source["label"]:
                errors.append(f"True label mismatch for {prediction['id']} in {filename}")
            if prediction.get("action") != source["action_text"]:
                errors.append(f"Action mismatch for {prediction['id']} in {filename}")
            if prediction.get("predicted_label") not in LABELS:
                errors.append(f"Invalid prediction label for {prediction['id']} in {filename}")
            aligned_rows.append(source)
            predicted_labels.append(prediction.get("predicted_label"))
        recomputed = _calculate_metrics(aligned_rows, predicted_labels)
        errors.extend(_compare(run.get("metrics"), recomputed, filename))

        if run.get("baseline") == "shuffled_label":
            repetition = run.get("shuffle_repetition")
            shuffle_seed = run.get("shuffle_seed")
            repetitions[(seed, evaluation)] += 1
            seed_key = (seed, evaluation)
            repetition_numbers[seed_key].add(repetition)
            if shuffle_seed in shuffle_seeds[seed_key]:
                errors.append(f"Repeated shuffle seed {shuffle_seed} in split {seed}")
            shuffle_seeds[seed_key].add(shuffle_seed)
            if run.get("training_class_counts_before_shuffle") != dict(split_training_counts[seed]):
                errors.append(f"Incorrect original training label counts in {filename}")
            if run.get("training_class_counts_after_shuffle") != dict(split_training_counts[seed]):
                errors.append(f"Shuffle did not preserve training label counts in {filename}")
            if not isinstance(repetition, int) or not 1 <= repetition <= 100:
                errors.append(f"Invalid shuffle repetition number in {filename}")

    expected_ordinary_baselines = {
        "majority", "action_name", "action_text_tfidf", "tfidf"
    }
    if set(manifest.get("baselines", [])) != expected_ordinary_baselines | {"shuffled_label"}:
        errors.append("Baseline manifest does not identify the approved five-baseline run")
    for baseline in expected_ordinary_baselines:
        for seed in SPLIT_SEEDS:
            for evaluation in ("v3_test", "challenge_all_external"):
                if ordinary_run_counts[(baseline, seed, evaluation)] != 1:
                    errors.append(
                        f"Expected one {baseline} run for split {seed} {evaluation}"
                    )

    prediction_dir = output_dir / "predictions"
    actual_prediction_names = {path.name for path in prediction_dir.glob("*.jsonl")}
    if actual_prediction_names != expected_prediction_names:
        errors.append("Prediction directory contains missing or unexpected JSONL files")
    for seed in SPLIT_SEEDS:
        for evaluation in ("v3_test", "challenge_all_external"):
            if len(shuffle_seeds[(seed, evaluation)]) != 100:
                errors.append(
                    f"Split {seed} {evaluation} has "
                    f"{len(shuffle_seeds[(seed, evaluation)])} unique shuffle seeds, expected 100"
                )
            if repetitions[(seed, evaluation)] != 100:
                errors.append(
                    f"Split {seed} {evaluation} has {repetitions[(seed, evaluation)]} shuffle runs, expected 100"
                )
            if repetition_numbers[(seed, evaluation)] != set(range(1, 101)):
                errors.append(
                    f"Split {seed} {evaluation} does not contain repetitions 1 through 100 exactly once"
                )

    report = {
        "output_directory": str(output_dir),
        "v3_dataset_sha256": _sha256(dataset_path),
        "challenge_dataset_sha256": _sha256(challenge_path),
        "run_count": len(metric_runs),
        "prediction_file_count": len(actual_prediction_names),
        "shuffle_runs_per_split_and_evaluation": {
            str(seed): {
                evaluation: repetitions[(seed, evaluation)]
                for evaluation in ("v3_test", "challenge_all_external")
            }
            for seed in SPLIT_SEEDS
        },
        "errors": errors,
        "passed": not errors,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Recalculate and validate saved baseline results")
    parser.add_argument("--results-dir", default="results/baselines_v2")
    parser.add_argument("--report", default="reports/validation/baselines_v2_validation.json")
    arguments = parser.parse_args()
    report = validate_results(arguments.results_dir)
    report_path = Path(arguments.report)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
