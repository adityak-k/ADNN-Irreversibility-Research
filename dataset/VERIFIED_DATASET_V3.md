# Verified dataset v3

`irreversibility_dataset_v3.jsonl` is a new execution-backed version. Dataset v2 remains preserved as `irreversibility_dataset_v2.jsonl`; this generator never writes to v2. The approved rules in `../REVERSIBILITY_RULES.md` remain unchanged.

## Ground-truth construction

Each row is generated in a fresh custom `Sandbox`. The generator records initial, post-action, and final operational states, executes the candidate operation, and makes no more than three recovery-tool calls. Audit history is recorded separately. The generator's label is then independently checked by `independent_outcomes.py`, which does not import the generator executor and examines observed state, actual recovered content, correction relationships, balances, settlement commitments, and recovery attempts. The v3 validator independently replays each record using its own setup and recovery sequence and checks that replay against the stored evidence.

For partial file recovery, the earlier version is constructed from the same file-specific shared information as the original and an older value for the same field. The independent checker verifies that the recovered copy retains the shared unit and older value while omitting the current-only unit. For partial email recovery, the original contains a claim ID and incorrect claim; the separate correction message must cite the original email and claim ID and include a different corrected claim. The delivered original remains intact. The sandbox does not model whether a recipient reads or accepts that correction.

Pending transfer cancellation and settled transfer refund semantics are unchanged. A pending request reserves sender funds; cancellation returns them and leaves no settlement commitment. Refund after settlement can improve balances but leaves the settlement commitment.

## Pairing, grouping, and input safety

The dataset has 60 pairs for each of the three label comparisons under each action: 720 pairs and 1,440 paired examples (60%). Each pair shares the same invariant action parameters and differs only in its recovery-condition field. Dataset records include canonical scenario groups, exact model-input groups, and wording-template groups. Scenario and exact-input groups combine into `canonical_group_id`; wording templates are tracked separately for a template-held-out evaluation.

Use `dataset.model_input.build_model_input(record)` to obtain the only approved model input: `environment`, `state_text`, and `action_text`. The nested `model_input` field stores this whitelist. Labels, action parameters, file-version evidence, correction text, post-action state, recovery attempts, audit history, and lifecycle evidence are evaluation-only. Do not construct prompts from whole JSONL records.

## Email timing scope

Email undo is a static sandbox setting. No clock advances, and an enabled undo option does not expire. The legacy v3 `recall_window_seconds` values appear in state descriptions but are not consumed by `Sandbox`; they are not evidence of a simulated countdown. Interpret these examples as testing static undo availability only. The v3 JSONL is preserved unchanged; future challenge examples use wording that does not imply a timed window.

## Diversity and scope

Descriptions vary openings, sentence forms, condition paraphrases, file versions, claim content, recipients, and transfer conditions. The set is intentionally balanced at 200 examples per action/class cell. It is still a controlled synthetic sandbox dataset; held-out wording evaluation should accompany the primary canonical-group splits. Three seeded group-level 80/20 splits are frozen under `experimental_splits_v1` for seeds 20261120, 20261121, and 20261122. Do not regenerate or overwrite those artifacts for a v3 experiment.

## Validation

Run `python dataset/validate_verified_dataset_v3.py`. It checks schema, exact action/class balance, pair completeness and invariants, group identifiers, duplicate/conflicting inputs, the model-input whitelist, and independent outcome evidence. It replays every row with a fresh sandbox through the independent checker. Generation, validation, and full test logs are saved in `../results/logs/phase3_2_*.txt`.
