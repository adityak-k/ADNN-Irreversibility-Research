# Phase 3.2 — Partial-recovery repair and dataset quality report

## 1. Files changed and backups

Modified existing files:

- `sandbox/environment.py` — claim-linked, separately delivered email corrections; generic or non-corrective messages are rejected.
- `tests/test_reversibility_ground_truth.py` — claim-correction and generic-follow-up regression coverage.

Created files:

- `dataset/generate_verified_dataset_v3.py`
- `dataset/independent_outcomes.py`
- `dataset/model_input.py`
- `dataset/grouping.py`
- `dataset/validate_verified_dataset_v3.py`
- `dataset/irreversibility_dataset_v3.jsonl`
- `dataset/VERIFIED_DATASET_V3.md`
- `tests/test_phase32_dataset_quality.py`
- `results/logs/phase3_2_generation.txt`
- `results/logs/phase3_2_validation.txt`
- `results/logs/phase3_2_full_tests.txt`
- `reports/PHASE3_2_DATASET_QUALITY_REPORT.md`

Backups made before modifying existing files:

- `backups/phase3_2_repair/environment.py`
- `backups/phase3_2_repair/test_reversibility_ground_truth.py`

Dataset v2 and its generator were not modified or overwritten. No changes were made to `REVERSIBILITY_RULES.md`.

## 2. Dataset counts

The new v3 dataset contains 2,400 examples.

| Action | Reversible | Partially reversible | Irreversible | Total |
|---|---:|---:|---:|---:|
| `edit_file` | 200 | 200 | 200 | 600 |
| `delete_file` | 200 | 200 | 200 | 600 |
| `send_email` | 200 | 200 | 200 | 600 |
| `transfer_money` | 200 | 200 | 200 | 600 |
| **Total** | **800** | **800** | **800** | **2,400** |

## 3. Pair statistics

There are 720 complete pairs (1,440 examples, 60% of the dataset). For each action, there are 60 pairs of each label combination.

| Label combination | Total pairs | Pairs per action |
|---|---:|---:|
| Reversible / partially reversible | 240 | 60 |
| Reversible / irreversible | 240 | 60 |
| Partially reversible / irreversible | 240 | 60 |

All pairs preserve the same action and invariant parameters, share a scenario group, and change only the recovery-condition field. Pair validation found no incomplete or conflicting pairs.

## 4. Independent outcome validation

The generator and checker use separate execution/orchestration code. `independent_outcomes.py` derives the expected label without reading the stored label, checks the initial, post-action, and final operational states, recovery attempts, and action-specific evidence, and independently replays each row in a fresh sandbox. It does not import the generator's executor.

- 2,400 / 2,400 examples independently checked and replayed.
- 0 validation errors.
- A deliberately corrupted stored label is rejected in regression tests.
- File partial cases preserve a shared original unit, recover an older value, omit a current-only value, and differ from the original.
- Email partial cases retain the original delivered message and create a second delivered message linked to the original email ID and claim ID, with the corrected claim. Generic follow-ups and repeated incorrect claims are rejected.
- Transfer cancellation, settlement commitment, refunds, and sender/recipient balance changes remain under the approved semantics.
- Every record uses at most three recovery calls.

## 5. Text diversity

| Measure | Result |
|---|---:|
| Unique state descriptions | 2,399 / 2,400 |
| Normalized wording patterns | 1,319 |
| Wording-template groups | 96 (24 per action) |
| Rows per template group | min 13, median 25, max 41 |
| Exact model-input duplicate groups | 1 group, 2 rows |

Each condition has multiple paraphrases; wording-template IDs omit the condition so a template-held-out evaluation can keep wording form isolated across labels. No class names appear in `state_text`. The environment facts still communicate the recovery condition, as required for the task; future evaluation should distinguish semantic condition recognition from generalization to new wording.

Mean TF-IDF cosine similarity for matched pair descriptions (word unigrams and bigrams):

| Action | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `edit_file` | 0.801 | 0.799 | 0.739 | 0.878 |
| `delete_file` | 0.717 | 0.716 | 0.608 | 0.849 |
| `send_email` | 0.645 | 0.628 | 0.525 | 0.824 |
| `transfer_money` | 0.642 | 0.626 | 0.551 | 0.828 |

## 6. Duplicate and leakage groups

| Grouping measure | Count |
|---|---:|
| Unique scenario-configuration groups | 1,665 |
| Canonical leakage groups (scenario plus exact observed input) | 1,664 |
| Wording-template groups | 96 |
| Unique observed model inputs | 2,399 |
| Duplicate input groups | 1 |
| Conflicting-label duplicate input groups | 0 |

The one duplicate prompt group is `SBX3_000254` and `SBX3_002365`; both have the same irreversible email input. They share one canonical group ID. `canonical_group_id` keeps matched examples, repeated underlying configurations, and identical model inputs together. `wording_template_group_id` is separate for template-held-out evaluation. No row-level split was created.

## 7. Model-input safety

`dataset/model_input.py` provides a fail-closed whitelist containing exactly `environment`, `state_text`, and `action_text`; each dataset row also stores only those fields in `model_input`. Recovery parameters, stale content, correction text, labels, operational states, traces, and audit history are marked evaluation-only. Tests verify the projection excludes these fields.

## 8. Tests

The full test suite passed: **15 unittest tests**, plus all existing assertion-based test modules. Coverage includes real partial file recovery, claim-linked email correction, rejection of generic correction, incorrect-label detection, model-input whitelist, duplicate prompt detection, pair integrity, exact dataset balance, and transfer transition regressions.

Complete output is saved in `results/logs/phase3_2_full_tests.txt`; generation and independent validation logs are saved alongside it.

## 9. Remaining limitations

- The sandbox is deterministic and synthetic; it does not establish behavior of real filesystems, email delivery, or financial services.
- Email validation checks the modeled claim relationship and delivery records, not whether a recipient reads or understands a correction.
- Dataset balance is designed, not representative of deployment prevalence.
- Although wording variants and template grouping improved, the descriptions remain generated and should be human-reviewed before broad claims about natural-language generalization.
- No training/test split has been generated. The primary evaluation should split by `canonical_group_id`; a separate challenge set should hold out `wording_template_group_id`.

## 10. Readiness

The repaired dataset is ready for the **next research-design phase: scenario-level split design and review**. It is suitable for a controlled sandbox study after that split is established. It should not yet support broad claims about real-world action irreversibility or generalization beyond this sandbox and its generated wording.
