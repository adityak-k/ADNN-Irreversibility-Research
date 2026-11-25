# ADNN Final Sprint Status

Run date: 2026-11-25. This report records only experiments that ran and outputs that were checked.

## RQ1: hidden-state probes

Executed all three existing frozen v3 splits: seeds `20261120`, `20261121`, and `20261122`. Each run used 1,920 training and 480 test examples; no split assignment changed. In total, 7,200 example-partition representations were extracted across the three overlapping splits (2,400 rows per split), and 29 L2 logistic-regression probes were evaluated per split. Layer 0 is the embedding-output control.

Model: local `Qwen2.5-1.5B-Instruct`, `C:\ADNN_MODELS\Qwen2.5-1.5B-Instruct`; 28 transformer layers, hidden size 1,536; FP16 inference on `NVIDIA GeForce RTX 5050 Laptop GPU`. Model config SHA256: `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670`. Runtime observed: PyTorch `2.14.1+cu130`, Transformers `5.24.1`, CUDA `13.0`.

Dataset: unchanged `dataset/irreversibility_dataset_v3.jsonl`, 2,400 examples; SHA256 `a7094909662c3158372fd66b8f04abd67311af0c6466978024c475f38848c026`. Model-visible input was restricted to `environment`, `state_text`, and `action_text`. Probe training used labels only from the corresponding training IDs. Test labels and outcomes were not sent to Qwen during extraction.

Headline results (all metrics are recalculated from saved predictions):

| Layer | Mean balanced accuracy | Population SD | Mean macro-F1 | Population SD |
| ---: | ---: | ---: | ---: | ---: |
| 0 (embedding control) | 0.333333 | 0.000000 | 0.166667 | 0.000000 |
| 4 (best mean balanced accuracy) | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| 28 | 0.998611 | 0.000982 | 0.998611 | 0.000982 |

The per-seed, per-layer table, including confusion matrices, is `rq1_three_seed_metrics.csv`; mean and standard deviation for every layer are in `rq1_three_seed_summary.csv`. The three split assignments overlap, so these standard deviations are descriptive across the three prescribed splits, not independent-replication uncertainty.

These results show that v3 labels are highly linearly separable in many Qwen hidden states under the existing split protocol. They do not establish generalization to held-out wording or unseen causal mechanisms. The project’s prior audit found substantial wording-template overlap across v3 train/test. Treat near-perfect scores as a warning about shortcut learning as well as a positive probe result.

## RQ2: direct answers

Executed once on exactly 120 examples from the test partition for seed `20261120`: 10 examples from each of the 12 action/class cells. The selection seed was `20262025`. Ground-truth labels were excluded from all prompts. The exact ordered IDs are saved in `rq2_seed_20261120_identity_check.json` and the original `rq2_direct_answers/predictions.jsonl`; this experiment was not repeated on the other two splits.

Accuracy: `0.416667`; macro-F1: `0.333735`; macro precision: `0.279461`; macro recall: `0.416667`. All 120 answers were parsed. The model predicted no `partially_reversible` examples (0/40 partial-class recall). On those exact IDs, layer 28 scored accuracy `1.000000`, macro-F1 `1.000000`; direct-answer/probe agreement was `0.416667`.

## RQ3: sandbox behavior pilot

An eight-example pilot ran using four reversible/irreversible matched pairs (one pair per action) and the existing `sandbox.Sandbox`. Scenario content used only the approved three input fields, alongside generic system and task instructions; the trace contains outputs, tool calls, operational states, and audit history. The saved pilot is in `rq3_sandbox_pilot_v2/`.

The pilot did not produce a usable structured behavior evaluation: all eight interaction records ended with `unparseable_output`; six candidate actions executed successfully, including three ground-truth irreversible actions. The model made zero pre-action state checks, requested zero confirmations, refused zero actions, and selected zero safer alternatives. This is an eight-task exploratory pilot only, not the compulsory 60-task evaluation. The first pilot attempt is preserved in `rq3_sandbox_pilot/`; it failed while serializing a `frozenset` in a trace. The rerun normalized the snapshot and saved all eight traces.

## Controls and incomplete requirements

- Ran the layer 0 embedding-output control and same-example RQ2 comparison with layer 28.
- Did not rerun majority, action-only, TF-IDF, or shuffled-label controls in this sprint. Existing Phase 4 baseline artifacts were left untouched.
- Did not evaluate Qwen on the separate 240-example challenge set or run a wording-held-out Qwen evaluation.
- Did not complete the assignment’s full 60-task agent behavior study.
- No model download, package installation, neural probe, activation steering, or hidden-state intervention was performed.
- The output contained expected deprecation warnings for Transformers `torch_dtype` and scikit-learn's `penalty="l2"`; these did not stop execution. No scores or artifacts were altered to address warnings.
- Each completed experiment's `config.json` retains its initial `in_progress` status; the separate `manifest.json` is the completion marker. No existing configuration was overwritten to change that status.

## Independent verification and outputs

Independent validation passed for all three seeds. For each seed it checked the original frozen ID files against the v3 examples and split metadata, all train/test representation IDs and labels, 29 prediction files, exact test-ID coverage, true labels, and manual recalculation of balanced accuracy, macro-F1, and confusion matrices. The three-seed report is `rq1_three_seed_independent_validation.json`. Summary generation also checked the RQ2 IDs against the exact seeded subset. The layer comparison graph is `rq1_three_seed_layer_comparison.svg` (SVG used because matplotlib is not installed in the supplied environment; no package was installed).

All new code and outputs are inside `experiments/final_sprint/`. Existing datasets, frozen splits, sandbox code, reversibility rules, historical results, logs, and backups were not written by these runs. The current v3 hash matches the frozen split manifest.

## Commands used

Run these from the project root in PowerShell with the supplied environment:

```powershell
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq1.py --smoke
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq1.py --seed 20261120
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq1.py --seed 20261121
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq1.py --seed 20261122
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\validate_rq1_outputs.py --seed 20261120
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\validate_rq1_outputs.py --seed 20261121
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\validate_rq1_outputs.py --seed 20261122
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq2.py
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\run_rq3_sandbox_pilot.py
& 'C:\ADNN-ML-ENV\Scripts\python.exe' -B experiments\final_sprint\verify_and_summarize_three_seeds.py
```

Experiment runners and summary validator refuse to overwrite their outputs. To repeat a completed run, use a fresh project copy with an empty `experiments/final_sprint/` output tree; the frozen dataset and split inputs remain unchanged.
