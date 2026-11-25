"""Extract real Qwen hidden states and run layer-wise linear probes.

All output files are new and are written below experiments/final_sprint.
The frozen v3 dataset and its saved split files are read-only inputs.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from dataset.model_input import MODEL_INPUT_FIELDS, build_model_input
from probes.evaluate import evaluate_predictions


DATASET_PATH = ROOT / "dataset" / "irreversibility_dataset_v3.jsonl"
SPLIT_ROOT = ROOT / "dataset" / "experimental_splits_v1"
MODEL_PATH = Path(r"C:\ADNN_MODELS\Qwen2.5-1.5B-Instruct")
PROJECT_OUTPUT = ROOT / "experiments" / "final_sprint"
LABELS = ("reversible", "partially_reversible", "irreversible")
SPLIT_SEEDS = (20261120, 20261121, 20261122)
SPLIT_SEED = SPLIT_SEEDS[0]
EXPECTED_LAYERS = 29  # embedding output and 28 transformer blocks
EXPECTED_HIDDEN_SIZE = 1536
SHARD_SIZE = 8
MAX_INPUT_TOKENS = 768


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_dataset():
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("V3 dataset contains duplicate IDs")
    for row in rows:
        if row.get("model_input") != build_model_input(row):
            raise ValueError(f"Approved model input mismatch for {row['id']}")
    return rows


def load_frozen_split(rows, split_seed=SPLIT_SEED):
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(f"Seed is not an approved frozen split: {split_seed}")
    folder = SPLIT_ROOT / f"split_seed_{split_seed}"
    metadata = read_json(folder / "metadata.json")
    manifest = read_json(SPLIT_ROOT / "manifest.json")
    dataset_hash = sha256_file(DATASET_PATH)
    if metadata.get("dataset_sha256") != dataset_hash or manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("The saved split is not bound to the current v3 dataset hash")
    if metadata.get("seed") != split_seed:
        raise ValueError("Frozen split seed does not match the requested seed")

    by_id = {row["id"]: row for row in rows}
    train_ids = read_json(folder / "training_example_ids.json")
    test_ids = read_json(folder / "testing_example_ids.json")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("Frozen split contains duplicate IDs")
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(by_id):
        raise ValueError("Frozen split does not partition all v3 examples exactly")

    train_rows = [by_id[item] for item in train_ids]
    test_rows = [by_id[item] for item in test_ids]
    for field in ("canonical_group_id", "scenario_group_id", "input_group_id", "pair_id"):
        train_groups = {row.get(field) for row in train_rows if row.get(field)}
        test_groups = {row.get(field) for row in test_rows if row.get(field)}
        if train_groups & test_groups:
            raise ValueError(f"{field} crosses the frozen split")
    if len(train_rows) != 1920 or len(test_rows) != 480:
        raise ValueError("Unexpected frozen split size")
    return train_rows, test_rows, dataset_hash, metadata


def format_permitted_input(row):
    """Format only fields returned by the approved model-input projection."""
    projected = build_model_input(row)
    if set(projected) != set(MODEL_INPUT_FIELDS):
        raise ValueError("Model input contains non-approved fields")
    return (
        f"Environment: {projected['environment']}\n"
        f"State: {projected['state_text']}\n"
        f"Candidate action: {projected['action_text']}"
    )


def open_log(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if log_path.exists() else "x"
    return log_path.open(mode, encoding="utf-8", newline="\n")


def log_event(log_handle, message):
    stamp = datetime.now(timezone.utc).isoformat()
    log_handle.write(f"{stamp} {message}\n")
    log_handle.flush()
    os.fsync(log_handle.fileno())
    print(message, flush=True)


def load_local_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sprint run")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda:0")
    model.eval()
    if model.config.num_hidden_layers != 28 or model.config.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            f"Unexpected model architecture: {model.config.num_hidden_layers} layers, "
            f"hidden size {model.config.hidden_size}"
        )
    return tokenizer, model


def extract_one(model, tokenizer, row):
    text = format_permitted_input(row)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    tokens = {name: tensor.to("cuda:0") for name, tensor in tokens.items()}
    attention = tokens.get("attention_mask")
    if attention is None or int(attention[0].sum().item()) == 0:
        raise ValueError(f"No input tokens for {row['id']}")
    last_token = int(torch.nonzero(attention[0], as_tuple=False)[-1].item())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = output.hidden_states
    if hidden is None or len(hidden) != EXPECTED_LAYERS:
        raise ValueError(f"Expected {EXPECTED_LAYERS} hidden states for {row['id']}")
    vectors = torch.stack([
        layer[0, last_token].detach().to(device="cpu", dtype=torch.float16)
        for layer in hidden
    ]).numpy()
    if vectors.shape != (EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE):
        raise ValueError(f"Wrong feature shape for {row['id']}: {vectors.shape}")
    return vectors


def load_valid_shards(folder, expected_rows, log_handle):
    expected = {row["id"]: row for row in expected_rows}
    found = {}
    for shard_path in sorted(folder.glob("*.npz")):
        try:
            with np.load(shard_path, allow_pickle=False) as shard:
                ids = shard["example_ids"].astype(str).tolist()
                labels = shard["labels"].astype(str).tolist()
                states = shard["hidden_states"]
            if states.shape != (len(ids), EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE):
                raise ValueError("wrong saved tensor shape")
            if len(labels) != len(ids):
                raise ValueError("ID/label length mismatch")
            for index, example_id in enumerate(ids):
                if example_id not in expected:
                    raise ValueError(f"unexpected example ID {example_id}")
                if labels[index] != expected[example_id]["label"]:
                    raise ValueError(f"saved label mismatch for {example_id}")
                if example_id in found:
                    raise ValueError(f"duplicate saved example ID {example_id}")
                found[example_id] = states[index]
        except Exception as error:
            log_event(log_handle, f"IGNORING incomplete/corrupt shard {shard_path.name}: {error}")
    return found


def next_shard_path(folder, partition):
    number = 0
    while True:
        candidate = folder / f"{partition}_{number:05d}.npz"
        if not candidate.exists():
            return candidate
        number += 1


def write_shard(path, rows, features):
    ids = [row["id"] for row in rows]
    labels = [row["label"] for row in rows]
    array = np.asarray(features, dtype=np.float16)
    if array.shape != (len(rows), EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE):
        raise ValueError(f"Refusing malformed shard {path.name}: {array.shape}")
    with path.open("xb") as output_file:
        np.savez_compressed(
            output_file,
            example_ids=np.asarray(ids, dtype=str),
            labels=np.asarray(labels, dtype=str),
            hidden_states=array,
        )
        output_file.flush()
        os.fsync(output_file.fileno())


def extract_partition(model, tokenizer, rows, partition, folder, log_handle):
    folder.mkdir(parents=True, exist_ok=True)
    found = load_valid_shards(folder, rows, log_handle)
    missing_rows = [row for row in rows if row["id"] not in found]
    log_event(log_handle, f"{partition}: {len(found)}/{len(rows)} valid saved rows; {len(missing_rows)} to extract")

    for start in range(0, len(missing_rows), SHARD_SIZE):
        batch_rows = missing_rows[start:start + SHARD_SIZE]
        features = []
        for row in batch_rows:
            features.append(extract_one(model, tokenizer, row))
            if len(features) % 2 == 0:
                log_event(log_handle, f"{partition}: extracted {min(start + len(features), len(missing_rows))}/{len(missing_rows)} pending rows")
        shard_path = next_shard_path(folder, partition)
        write_shard(shard_path, batch_rows, features)
        for row, feature in zip(batch_rows, features):
            found[row["id"]] = feature
        log_event(log_handle, f"saved {shard_path.relative_to(PROJECT_OUTPUT)} with {len(batch_rows)} rows")
        del features
        torch.cuda.empty_cache()

    if set(found) != {row["id"] for row in rows}:
        raise RuntimeError(f"{partition} extraction is incomplete; no complete marker will be written")
    ordered = np.stack([found[row["id"]] for row in rows]).astype(np.float16, copy=False)
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    ids = [row["id"] for row in rows]
    return ids, labels, ordered


def make_config(dataset_hash, metadata, smoke, split_seed):
    from transformers import __version__ as transformers_version
    return {
        "experiment": "ADNN_RQ1_layerwise_linear_probe",
        "status": "smoke_test" if smoke else "in_progress",
        "model_name": "Qwen2.5-1.5B-Instruct",
        "model_path": str(MODEL_PATH),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "model_layers": 28,
        "hidden_state_outputs": EXPECTED_LAYERS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "input_fields": list(MODEL_INPUT_FIELDS),
        "input_format": "Environment / State / Candidate action; only approved projection values",
        "dtype": "float16",
        "device": torch.cuda.get_device_name(0),
        "dataset_version": "irreversibility_dataset_v3",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "split_seed": split_seed,
        "split_metadata": metadata,
        "probe": {"classifier": "sklearn LogisticRegression", "penalty": "l2", "C": 1.0,
                  "solver": "lbfgs", "max_iter": 1000, "random_state": split_seed},
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "shard_size": SHARD_SIZE,
        "subset_policy": "none" if not smoke else "first train ID and first test ID in frozen ID order",
    }


def ensure_config(folder, config):
    path = folder / "config.json"
    if path.exists():
        existing = read_json(path)
        # Run state changes from in_progress to complete, so compare fixed fields only.
        for key, value in config.items():
            if key == "status":
                continue
            if existing.get(key) != value:
                raise ValueError(f"Resume config mismatch for {key}; refusing to mix runs")
        return
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def train_and_evaluate(folder, train_ids, train_labels, train_features,
                       test_ids, test_labels, test_features, split_seed, log_handle):
    layer_dir = folder / "layer_results"
    layer_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for layer_index in range(EXPECTED_LAYERS):
        metrics_path = layer_dir / f"layer_{layer_index:02d}_metrics.json"
        predictions_path = layer_dir / f"layer_{layer_index:02d}_predictions.jsonl"
        coefficients_path = layer_dir / f"layer_{layer_index:02d}_probe.npz"
        if metrics_path.exists() and predictions_path.exists() and coefficients_path.exists():
            existing_metrics = read_json(metrics_path)
            all_metrics.append(existing_metrics)
            log_event(log_handle, f"resume: layer {layer_index} outputs already exist; keeping them")
            continue
        if any(path.exists() for path in (metrics_path, predictions_path, coefficients_path)):
            raise FileExistsError(f"Incomplete saved outputs for layer {layer_index}; refusing overwrite")

        classifier = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
            random_state=split_seed,
        )
        classifier.fit(train_features[:, layer_index, :].astype(np.float32), train_labels)
        predicted = classifier.predict(test_features[:, layer_index, :].astype(np.float32))
        metrics = evaluate_predictions(test_labels.tolist(), predicted.tolist(), LABELS)
        metrics.update({"layer": layer_index, "control": layer_index == 0,
                        "train_examples": len(train_ids), "test_examples": len(test_ids)})

        with predictions_path.open("x", encoding="utf-8", newline="\n") as output_file:
            for example_id, truth, prediction in zip(test_ids, test_labels, predicted):
                output_file.write(json.dumps({
                    "id": example_id,
                    "true_label": str(truth),
                    "predicted_label": str(prediction),
                    "layer": layer_index,
                }) + "\n")
        with coefficients_path.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                classes=np.asarray(classifier.classes_, dtype=str),
                coefficients=classifier.coef_.astype(np.float32),
                intercept=classifier.intercept_.astype(np.float32),
            )
        with metrics_path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(json.dumps(metrics, indent=2) + "\n")
        all_metrics.append(metrics)
        log_event(log_handle, f"layer {layer_index}: balanced_accuracy={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
        del classifier
    if len(all_metrics) != EXPECTED_LAYERS:
        raise RuntimeError("Not all 29 layer metrics are present")
    return all_metrics


def run(smoke=False, split_seed=SPLIT_SEED):
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(f"Seed is not an approved frozen split: {split_seed}")
    run_root = PROJECT_OUTPUT / ("smoke_rq1" if smoke else f"rq1_seed_{split_seed}")
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "execution.log"
    with open_log(log_path) as log_handle:
        rows = read_dataset()
        train_rows, test_rows, dataset_hash, metadata = load_frozen_split(rows, split_seed)
        if smoke:
            train_rows, test_rows = train_rows[:1], test_rows[:1]
        config = make_config(dataset_hash, metadata, smoke, split_seed)
        ensure_config(run_root, config)
        log_event(log_handle, f"starting {'smoke' if smoke else 'full'} run; dataset rows={len(rows)} train={len(train_rows)} test={len(test_rows)}")
        log_event(log_handle, f"v3 dataset sha256={dataset_hash}; split seed={split_seed}")

        tokenizer, model = load_local_model()
        log_event(log_handle, f"loaded local model on {torch.cuda.get_device_name(0)} with fp16 weights")
        train_ids, train_labels, train_features = extract_partition(
            model, tokenizer, train_rows, "train", run_root / "representations" / "train", log_handle
        )
        test_ids, test_labels, test_features = extract_partition(
            model, tokenizer, test_rows, "test", run_root / "representations" / "test", log_handle
        )
        log_event(log_handle, f"all requested representations ready: train={len(train_ids)} test={len(test_ids)} layers={train_features.shape[1]}")
        if smoke:
            smoke_path = run_root / "smoke_result.json"
            if not smoke_path.exists():
                smoke_result = {
                    "status": "smoke_pass_not_experiment",
                    "model": "Qwen2.5-1.5B-Instruct",
                    "dataset_sha256": dataset_hash,
                    "split_seed": split_seed,
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "hidden_state_shape_per_example": list(train_features.shape[1:]),
                    "hidden_state_dtype": str(train_features.dtype),
                    "input_fields": list(MODEL_INPUT_FIELDS),
                }
                with smoke_path.open("x", encoding="utf-8", newline="\n") as output_file:
                    output_file.write(json.dumps(smoke_result, indent=2) + "\n")
            log_event(log_handle, "SMOKE PASS: real 29-output fp16 hidden states saved; no probe metrics claimed")
            return []

        del model, tokenizer
        torch.cuda.empty_cache()

        metrics = train_and_evaluate(
            run_root, train_ids, train_labels, train_features,
            test_ids, test_labels, test_features, split_seed, log_handle,
        )
        summary_path = run_root / "layer_metrics.json"
        if not summary_path.exists():
            with summary_path.open("x", encoding="utf-8", newline="\n") as output_file:
                output_file.write(json.dumps(metrics, indent=2) + "\n")
        elif len(read_json(summary_path)) != EXPECTED_LAYERS:
            raise ValueError("Existing layer_metrics.json is incomplete; refusing overwrite")

        expected_train = {row["id"] for row in train_rows}
        expected_test = {row["id"] for row in test_rows}
        if set(train_ids) != expected_train or set(test_ids) != expected_test:
            raise RuntimeError("Final saved representations do not match requested ID sets")

        if not smoke:
            manifest_path = run_root / "manifest.json"
            if not manifest_path.exists():
                manifest = {
                    **config,
                    "status": "complete",
                    "train_examples_extracted_and_probed": len(train_ids),
                    "test_examples_extracted_and_evaluated": len(test_ids),
                    "layers_evaluated": EXPECTED_LAYERS,
                    "prediction_files": EXPECTED_LAYERS,
                    "conclusion_scope": "Frozen v3 scenario split only; wording-template leakage remains a limitation.",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                with manifest_path.open("x", encoding="utf-8", newline="\n") as output_file:
                    output_file.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            log_event(log_handle, "COMPLETE: manifest.json written after all extraction, probe, and evaluation checks passed")
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run one train and one test example in isolated smoke output")
    parser.add_argument("--seed", type=int, choices=SPLIT_SEEDS, default=SPLIT_SEED,
                        help="Choose one of the three already-frozen v3 split seeds")
    arguments = parser.parse_args()
    run(smoke=arguments.smoke, split_seed=arguments.seed)


if __name__ == "__main__":
    main()
