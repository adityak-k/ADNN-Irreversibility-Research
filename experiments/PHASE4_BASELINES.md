# Phase 4: CPU Baselines

This experiment runs four simple baselines on the three frozen Phase 3.4 v3 splits. For each seed, a baseline is fitted only on that split's v3 training IDs and evaluated on its paired v3 test IDs. The fitted model is then evaluated separately on all 240 challenge examples. Challenge rows are never used for fitting, tuning, or model selection. The challenge dataset's existing 192/48 split remains available for a later separate experiment.

## Inputs and leakage controls

The runner reads `dataset/irreversibility_dataset_v3.jsonl`, `dataset/irreversibility_challenge_v1.jsonl`, and the saved IDs under `dataset/experimental_splits_v1`. It checks the dataset hashes against the frozen manifests and checks exact split coverage. Rows reach model code only through `dataset/model_input.py`; TF-IDF reads `state_text` and `action_text`, while the action-only baseline reads `action_text`. Labels are used only as training targets or evaluation truth. Recovery traces, outcome records, action parameters, and post-action states are not model inputs.

The majority baseline predicts the most frequent training label. Ties use the fixed order `reversible`, `partially_reversible`, `irreversible`. The action-only baseline predicts the most frequent training label for each action name, using the same tie rule. TF-IDF uses lowercase unigram word features with `min_df=1` and multinomial logistic regression with `C=1.0`, `max_iter=1000`, `solver=lbfgs`, and `random_state=0`. These settings are fixed in advance. The shuffled-label control shuffles only the training labels with Python's seeded random generator (split seed plus 900000); test and challenge labels remain unchanged.

## Commands

Run from the project root in PowerShell. To run one baseline at a time:

```powershell
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --baseline majority --dataset dataset\irreversibility_dataset_v3.jsonl --split-root dataset\experimental_splits_v1 --seeds 20261120 20261121 20261122 --challenge-dataset dataset\irreversibility_challenge_v1.jsonl --output-dir results\baselines_v1\majority
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --baseline action_name --dataset dataset\irreversibility_dataset_v3.jsonl --split-root dataset\experimental_splits_v1 --seeds 20261120 20261121 20261122 --challenge-dataset dataset\irreversibility_challenge_v1.jsonl --output-dir results\baselines_v1\action_name
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --baseline tfidf --dataset dataset\irreversibility_dataset_v3.jsonl --split-root dataset\experimental_splits_v1 --seeds 20261120 20261121 20261122 --challenge-dataset dataset\irreversibility_challenge_v1.jsonl --output-dir results\baselines_v1\tfidf
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --baseline shuffled_label --dataset dataset\irreversibility_dataset_v3.jsonl --split-root dataset\experimental_splits_v1 --seeds 20261120 20261121 20261122 --challenge-dataset dataset\irreversibility_challenge_v1.jsonl --output-dir results\baselines_v1\shuffled_label
```

Alternatively, one invocation runs all four baselines:

```powershell
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --baseline majority action_name tfidf shuffled_label --dataset dataset\irreversibility_dataset_v3.jsonl --split-root dataset\experimental_splits_v1 --seeds 20261120 20261121 20261122 --challenge-dataset dataset\irreversibility_challenge_v1.jsonl --output-dir results\baselines_v1
```

The runner refuses to overwrite a non-empty output directory. Move or rename prior results yourself before a new run.

## Metrics and outputs

For each baseline, split, and evaluation set, the runner saves accuracy, balanced accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, confusion matrix, and per-action metrics. It also saves the mean and population standard deviation across three seeds. Per-example prediction files contain the example ID, true label, predicted label, action, evaluation name, and baseline.

Outputs are written to `results/baselines_v1/`:

- `manifest.json`: data hashes, seeds, model settings, and input policy.
- `metrics_by_split.json`: metrics for each v3 test and all-challenge evaluation.
- `summary.json`: mean and standard deviation across seeds by evaluation type.
- `predictions/`: one JSONL prediction file per baseline, seed, and evaluation.

## Tests

Run the focused test module with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_phase4_baselines -v
```

These CPU baselines do not load neural models and do not require a GPU.
