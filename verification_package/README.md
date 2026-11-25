# Do AI Agents Know Which Actions Cannot Be Undone?

This repository studies action reversibility in a small, controlled sandbox.
The implemented sandbox covers file edits and deletion, email sending, and
money transfers. Recovery is limited to three tool calls. Operational state is
used to decide reversibility; audit history is retained separately.

## Current status

- The sandbox, action outcome rules, execution-backed v3 dataset, challenge
  dataset, and frozen group-level splits are implemented.
- CPU baselines are implemented in `experiments/run_baselines.py`. They include
  majority, action-name, action-text-only TF-IDF, state-plus-action TF-IDF, and
  a 100-repetition shuffled-training-label control per frozen split.
- The historical datasets, frozen split assignments, and Phase 4 v1 results
  remain preserved. New baseline output uses `results/baselines_v2/`.
- Optional utilities now define interfaces for loading open-weight models,
  extracting real hidden states, training linear probes, collecting direct
  model answers, and saving agent traces. No neural model has been loaded and
  no neural experiment has been run.
- ScienceWorld integration and the complete RQ1–RQ3 experiments remain to be
  implemented. See [the project guide](docs/PROJECT_GUIDE.md).

## Setup and commands

Create a virtual environment, install `requirements.txt`, then run from this
folder in Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
python -B -m unittest discover -v
python -B dataset/validate_verified_dataset_v3.py
python -B dataset/validate_experimental_splits_v1.py --output-dir reports/validation/run_01_splits
python -B dataset/validate_challenge_dataset_v1.py --output-dir reports/validation/run_01_challenge
python -B experiments/run_baselines.py --output-dir results/baselines_v2
```

Validation and experiment commands refuse to overwrite their saved output.
Choose a new report or result directory for a new run.

For model experiments, first install a suitable PyTorch build for your
computer, then install `requirements-models.txt`. A 7B parameter model generally
needs substantial GPU memory or system RAM. Model loading is opt-in and has
not been attempted as part of this project foundation.

See [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for file roles, data flow,
reversibility rules, experiment details, and remaining assignment work.
