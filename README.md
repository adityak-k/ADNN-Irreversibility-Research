# Do AI Agents Know Which Actions Cannot Be Undone?

This project studies whether language models can identify when actions in a controlled tool-using environment can be undone. The sandbox covers file edits and deletion, email sending, and money transfers. Recovery is evaluated against operational state within a three-tool-call horizon; audit history is retained separately.

## Project status

The project includes the sandbox and reversibility rules, execution-backed datasets, frozen group-level v3 splits, challenge data, CPU baselines, and a final Qwen experiment sprint. The Phase 4 baseline outputs are in `results/baselines_v2/`; hidden-state probe outputs, direct-answer results, pilot traces, and independent RQ1 verification are in `experiments/final_sprint/`.

The final sprint ran Qwen2.5-1.5B-Instruct on the three frozen v3 splits. It extracted last-token hidden states from the embedding output and all 28 transformer layers, then evaluated L2 logistic-regression probes. The best mean balanced accuracy was 1.000 at layer 4; layer 0 scored 0.333333 and layer 28 scored 0.998611. The direct-answer run used 120 examples from seed 20261120 and achieved 0.416667 accuracy and 0.333735 macro-F1. The RQ3 behavior work is an eight-task pilot, not the complete behavior evaluation. See `experiments/final_sprint/FINAL_SPRINT_STATUS.md` for scope, metrics, limitations, and exact commands.

The v3 dataset has documented wording-template leakage. High probe scores therefore show linear separability under the frozen split protocol; they do not establish generalization to unseen wording or causal mechanisms. ScienceWorld and a second model family have not been completed.

## Project layout

- `sandbox/`: action environment and recovery tools.
- `dataset/`: datasets, generation and validation code, grouping, model-input projection, and frozen split artifacts.
- `tests/`: sandbox, dataset, oracle, and baseline regression tests.
- `experiments/`: CPU baselines and final sprint experiment scripts/results.
- `results/`: baseline predictions, metrics, validation outputs, and logs.
- `reports/`, `docs/`: research audits, project guide, and validation reports.
- `backups/`: preserved historical source, data, and rules snapshots.
- `verification_package/`: source and verification evidence packaged for review.

## Environment and dependencies

### CPU baselines and tests

Use Python 3.14 (the project test and baseline environment) and install the pinned baseline dependency:

```powershell
python -m pip install -r requirements.txt
python -B -m unittest discover -v
```

`requirements.txt` pins scikit-learn 1.9.1. Sandbox execution, dataset generation/validation, and most tests use the Python standard library.

### Final sprint model environment

The final sprint used a separate Windows environment and a local model that are not included in this repository:

- Model: Qwen2.5-1.5B-Instruct at `C:\ADNN_MODELS\Qwen2.5-1.5B-Instruct`.
- PyTorch: `2.14.1+cu130`.
- Transformers: `5.24.1`.
- CUDA runtime: `13.0`.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU; inference used FP16.

`requirements-models.txt` records the optional model-side Python packages (Transformers 5.23.0 and Accelerate 1.16.0) used by the earlier model utility setup. It does not pin PyTorch because the appropriate CPU/CUDA build depends on the machine. The final sprint’s observed runtime versions above are recorded separately in its configs and status report; use the sprint environment when reproducing those exact results. Model weights are not distributed here.

## Reproduction commands

Run commands from the project root in PowerShell. For CPU baselines:

```powershell
python -B -m unittest discover -v
python -B dataset/validate_verified_dataset_v3.py
python -B dataset/validate_experimental_splits_v1.py --output-dir reports/validation/run_01_splits
python -B dataset/validate_challenge_dataset_v1.py --output-dir reports/validation/run_01_challenge
python -B experiments/run_baselines.py --output-dir results/baselines_v2
```

For the final sprint, activate the matching local model environment and use the commands recorded in `experiments/final_sprint/FINAL_SPRINT_STATUS.md`. The scripts refuse to overwrite completed run outputs; reproduce in a fresh project copy with the frozen dataset and split artifacts.

## Research rules and data

`REVERSIBILITY_RULES.md` defines reversible, partially reversible, and irreversible outcomes using operational state and the three-action recovery horizon. Audit records are preserved independently. Dataset records are JSONL; the v3 validation report and split manifests record dataset/split hashes and group-level assignments. Do not regenerate frozen datasets or split assignments when reproducing reported experiments.

## Known limitations

- V3 has wording-template leakage across train/test groups.
- The near-perfect probe scores should not be interpreted as evidence of general causal reasoning.
- Direct answers were evaluated on one 120-example subset and were not repeated across all three splits.
- The sandbox behavior evaluation is an unsuccessful eight-task pilot, not the required full study.
- ScienceWorld and the second model family remain future work.