# Reversibility Challenge Dataset v1

This is a separate, execution-backed challenge dataset. It does not replace or alter the original, v2, or v3 JSONL datasets or their frozen splits.

## Design

- 240 examples: 20 examples for each of four actions by three outcome labels.
- 120 matched pairs: for every action, ten pairs for each of the three label comparisons.
- Pair members share the action, base parameters, and wording structure; only the recovery condition changes.
- Three wording structures are used for every action and each structure occurs with every label.
- State text describes only the pre-action state and candidate action. The model input is restricted to `environment`, `state_text`, and `action_text`.
- The generator executes each scenario in a fresh `Sandbox`. Labels are derived from the observed operational states and recovery attempts, then checked by `dataset/reference_oracle.py`, which does not call `Sandbox`.
- Recovery plans use at most three calls. The generator tries each relevant tool at most once and stops after a successful recovery.

The challenge uses observable content and environment facts rather than explicit class words. Those facts still carry causal information; this dataset does not make the three labels independent of the environment or prove reasoning beyond the represented sandbox mechanisms.

## Email scope

Email undo availability is static. There is no clock or expiry event. Challenge descriptions contain no countdown. V3 remains byte-for-byte unchanged; its legacy `recall_window_seconds` wording is documented as non-operative metadata in `VERIFIED_DATASET_V3.md` and `REVERSIBILITY_RULES.md`.

## Evaluation split

`experimental_challenge_v1/` contains a canonical-grouped 80/20 split using seed `20261125`: 192 training examples in 96 groups and 48 test examples in 24 groups. Every action/class cell has 16 training and 4 test examples. The manifest and metadata bind the split to the challenge dataset SHA-256. This small split is a challenge evaluation, not a replacement for the larger frozen v3 splits.

## Validation

Run `python dataset/validate_challenge_dataset_v1.py` to check execution evidence against the independent reference oracle, balance, pairs, wording-template coverage, grouping, split integrity, and same-seed split replay. The validator writes its reports only under `dataset/experimental_challenge_v1/`.

The generator and split creator refuse to overwrite existing versioned artifacts. Create a new version/path for any later design change.
