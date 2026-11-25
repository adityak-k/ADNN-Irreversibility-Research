# Final foundation rework status

Date: 2026-11-25

## Result

The existing experimental foundation is preserved and now has safer dataset
generation, separated validation outputs, expanded CPU baselines, an independent
saved-results checker, and beginner-readable entry points for future model and
agent experiments. No historical dataset, frozen split assignment, v1 baseline
result, execution log, or historical backup changed.

The pre-change hash comparison found **101/101 protected files unchanged**.
Backups exist for every modified source/configuration file. See
`prechange_protected_hashes.json` and `preservation_check_final.json` in this folder.

## Final architecture

```text
Sandbox and reversibility rules
        ↓
Execution-backed datasets + outcome checks
        ↓
Approved model-input projection
        ↓
Frozen scenario-grouped splits
        ↓
CPU baselines + independently checked predictions
        ↓
Optional model / hidden-state / probe / answer / agent utilities
```

The neural and ScienceWorld stages are prepared but incomplete. No model
weights were downloaded or loaded, and no neural results were produced.

## What changed

- The legacy v1 data generator no longer writes on import. V1, v2, and the
  historical v3 generator paths now refuse to overwrite existing datasets.
- The v2 and future v3/v4 email generators now use claim-linked correction
  messages, and their wording describes static undo
  availability; it does not claim a real countdown. Existing v3 examples remain
  unchanged.
- The frozen v3 split generator refuses an existing split directory. Split and
  challenge validators now save reports to separate, caller-selected
  directories and refuse to replace prior reports. Frozen split directories
  remain read-only inputs.
- The baseline runner now includes majority, action-name majority,
  action-text-only TF-IDF, state-plus-action TF-IDF, and 100 fixed shuffled-label
  repetitions for each frozen split. Shuffles permute training labels only and
  preserve their class counts. The same fitted shuffled model is evaluated
  against that split's test rows and all 240 challenge rows.
- `experiments/validate_baseline_results.py` checks every prediction ID and true
  label, recalculates metrics independently, checks group crossings and hashes,
  and verifies all 100 shuffle repetitions per split/evaluation.
- Optional modules now cover family-neutral Hugging Face model loading,
  last-token hidden-state extraction and saving, layer-wise L2 logistic probes,
  three-class/binary metrics, direct-answer recording, and observable agent
  caution logging/summary functions.
- Core requirements no longer pin a CUDA-only PyTorch wheel or list every
  transitive package. Neural packages are optional and PyTorch is installed
  separately for the target computer.

## Dataset, split, and test results

- V3: **2,400** examples; **800** per label; **200** in every action/class cell;
  **720** valid matched pairs; zero validation errors. SHA-256:
  `a7094909662c3158372fd66b8f04abd67311af0c6466978024c475f38848c026`.
- Frozen v3 splits: all three passed independent checks, with 1,920 training
  and 480 test examples, exact 160/40 action/class cell counts, same-seed replay,
  and zero canonical, scenario, input, or pair crossings. Group counts are
  recorded in the split validation report.
- Challenge v1: **240** examples, 20 per action/class cell, 120 matched pairs;
  its frozen split is 192/48 with 96/24 groups and zero group crossings. All
  240 outcomes passed the independent reference oracle.
- Full test suite: **53 tests passed**.
- `pip check`: **No broken requirements found**.
- New baseline results: 624 predictions across the five baselines, three
  frozen splits, v3 test evaluations, and all-challenge external evaluations.
  The independent result checker recomputed every saved metric and found zero
  errors. Every expected example ID and true label matched.

| Baseline | Mean v3 test accuracy | Mean external challenge accuracy |
|---|---:|---:|
| Majority class | 0.3333 | 0.3333 |
| Action-name majority | 0.3333 | 0.3333 |
| Action-text-only TF-IDF | 0.3333 | 0.3333 |
| State-plus-action TF-IDF | 1.0000 | 0.3764 |

The action field contains canonical tokens such as `edit_file`, so action-text
TF-IDF is another action-name lexical control, not a natural-language action
description experiment.

The 100-run shuffled-label accuracy summaries are intentionally kept separate
by split because the three frozen splits overlap:

| Seed | Shuffled v3 test mean ± SD | Shuffled challenge mean ± SD | Real TF-IDF challenge accuracy | Empirical upper-tail accuracy |
|---:|---:|---:|---:|---:|
| 20261120 | 0.3291 ± 0.0422 | 0.3288 ± 0.0418 | 0.3792 | 0.1089 |
| 20261121 | 0.3254 ± 0.0341 | 0.3397 ± 0.0414 | 0.3708 | 0.2475 |
| 20261122 | 0.3353 ± 0.0421 | 0.3337 ± 0.0394 | 0.3792 | 0.1485 |

The challenge TF-IDF scores do not clearly exceed the per-split shuffled-label
distributions. The perfect v3 score alongside weak challenge performance
supports a wording/generalization concern; it does not establish semantic or
causal reasoning. The empirical upper-tail figures use the finite-sample
plus-one calculation and are descriptive controls, not a substitute for
independent dataset replications.

## Commands actually run

Run these from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v
.\.venv\Scripts\python.exe -B dataset\validate_verified_dataset_v3.py
.\.venv\Scripts\python.exe -B dataset\validate_experimental_splits_v1.py --output-dir reports\final_rework\validation\v3_splits
.\.venv\Scripts\python.exe -B dataset\validate_challenge_dataset_v1.py --output-dir reports\final_rework\validation\challenge_v1
.\.venv\Scripts\python.exe -B experiments\run_baselines.py --output-dir results\baselines_v2
.\.venv\Scripts\python.exe -B experiments\validate_baseline_results.py --results-dir results\baselines_v2 --report reports\final_rework\validation\baselines_v2_rerun.json
.\.venv\Scripts\python.exe -B -m pip check
```

Those report/result paths now contain outputs and intentionally refuse
overwriting. Use new names such as `reports\validation\run_02_splits`,
`reports\validation\run_02_challenge`, `reports\validation\baselines_v3.json`,
and `results\baselines_v3` for another run.

## Logs and reports

- Test logs: `results/logs/final_rework_full_tests.txt` and
  `results/logs/final_rework_final_tests.txt`, and the final
  `results/logs/final_rework_last_full_tests.txt`.
- Dataset/split validation logs: `results/logs/final_rework_v3_dataset_validation.txt`,
  `results/logs/final_rework_v3_splits_validation.txt`, and
  `results/logs/final_rework_challenge_validation.txt`.
- Baseline logs: `results/logs/final_rework_baselines_v2.txt`,
  `results/logs/final_rework_baselines_validation.txt`, and
  `results/logs/final_rework_baselines_validation_rerun.txt`.
- Independent validation reports: `reports/final_rework/validation/`.

The first baseline result-checker run failed because its checker treated each
shuffle's test and challenge evaluation as duplicate seeds. The runner outputs
were intact. The checker was fixed to track seeds separately by split and
evaluation; its preserved rerun passed with 624/624 predictions and no errors.

## Exact file lists

`created_files.txt` lists every newly created file, including all 624 prediction
JSONL files and the eight pre-change backups. Modified source/config files are:

- `dataset/generate_dataset.py`
- `dataset/generate_verified_dataset.py`
- `dataset/generate_verified_dataset_v3.py`
- `dataset/validate_experimental_splits_v1.py`
- `dataset/validate_challenge_dataset_v1.py`
- `dataset/generate_experimental_splits_v1.py`
- `experiments/run_baselines.py`
- `requirements.txt`
- `sandbox/environment.py`

The nine backups preserve the prior contents at the same relative paths under
`backups/final_rework_prechange/`.

## Remaining limitations and assignment work

- **RQ1 is not run:** no model hidden states or probe scores exist. The new
  functions do not replace a split-aware extraction and evaluation script.
- **RQ2 is not run:** direct-answer generation is implemented as a utility, but
  no model answer records or probe-agreement analysis exist.
- **RQ3 is not run:** the logger and caution summaries exist, but there is no
  tool-enabled agent loop or executed behavior study.
- **ScienceWorld is not integrated.** It remains a compulsory second testbed.
- Two open-weight instruction-tuned model families have not been selected,
  loaded, or evaluated. Current CUDA availability was checked, but no model was
  loaded and no hardware capacity claim was made.
- V3 retains known wording-template leakage and legacy email countdown phrases;
  those are frozen and documented limitations. No wording-held-out v3 split was
  added in this rework.
- The preserved v2 dataset contains 200 `send_email` examples labelled partial
  whose recorded correction messages have no claim metadata/link. They do not
  meet the current claim-linked recovery rule and must not be used as valid
  email ground truth. The v2 dataset was not edited; the v2 generator was
  repaired for any newly named output.
- The challenge set is synthetic and small; it does not prove generalization to
  unseen causal mechanisms or real-world services.
- The reference oracle is separately implemented and does not call `Sandbox`,
  but the dataset replay checker does share `Sandbox`; this is reduced shared
  logic, not complete implementation independence.
- The four `.pyc` files created by a syntax compilation check are listed in
  `created_files.txt` and were left in place because the instruction prohibited
  deleting files.

### Preservation incident disclosed

While preparing a source patch, a patch operation briefly removed
`dataset/generate_dataset.py`. I immediately restored it from the byte-identical
pre-change backup and verified matching hashes at that point. The file was
later intentionally edited under the approved rework with that backup
preserved; the historical dataset is unchanged. This was a transient violation
of the no-deletion instruction and is reported here rather than hidden.
