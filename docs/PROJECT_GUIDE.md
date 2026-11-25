# Project guide

## Purpose and research questions

The project asks whether AI systems can identify actions whose effects cannot
be fully undone. Its compulsory questions are:

1. **RQ1:** Can reversibility be decoded from model hidden states with linear
   probes, and at which layers is it strongest?
2. **RQ2:** Does a model state the right reversibility class, and does that
   answer agree with the probe prediction?
3. **RQ3:** Does an agent behave more carefully before actions that are hard to
   undo?

The current code provides the sandbox, datasets, frozen splits, CPU baselines,
and starter utilities for model experiments. It does not yet answer RQ1–RQ3.
It also does not yet integrate ScienceWorld, the required second testbed.

## Folder guide

| Path | Purpose |
|---|---|
| `sandbox/` | Toy stateful environment and a three-call recovery helper. |
| `dataset/` | Historical and verified JSONL datasets, model-input projection, outcome checkers, grouping, generators, and split validators. |
| `dataset/experimental_splits_v1/` | Frozen v3 split assignments and their original manifests. Treat as read-only. |
| `dataset/experimental_challenge_v1/` | Frozen challenge-set split. Treat as read-only. |
| `experiments/` | CPU baseline runner and direct-answer utility. |
| `hidden_states/` | Functions for extracting and saving actual model representations. |
| `probes/` | Per-layer logistic-regression probe and evaluation functions. |
| `agent/` | Pre-action task projection and behavior trace writer. This is not a complete agent loop. |
| `tests/` | Standard-library `unittest` coverage for the sandbox, data checks, and experiment utilities. |
| `results/baselines_v1/` | Historical Phase 4 results. Do not overwrite. |
| `results/baselines_v2/` | New results from the expanded baseline run. |
| `backups/` | Historical backups and the source/config copies made before this rework. |
| `reports/` | Audits and validation reports. |

Important files:

- `sandbox/environment.py` implements the environment's file, email, and
  transfer transitions. `sandbox/recovery.py` limits a supplied recovery plan
  to three calls.
- `REVERSIBILITY_RULES.md` defines operational-state comparison, partial
  recovery, email's static undo setting, and transfer lifecycle semantics.
- `dataset/model_input.py` is the only approved projection to model-visible
  fields: `environment`, `state_text`, and `action_text`.
- `dataset/generate_verified_dataset_v3.py` is retained under its historical
  filename, but its safe default now creates a new v4 dataset. It no longer
  writes the frozen v3 path and no longer describes static email undo as a
  countdown.
- `dataset/independent_outcomes.py` checks evidence and replays scenarios in a
  sandbox. `dataset/reference_oracle.py` separately computes expected outcomes
  without importing the sandbox. These are independent in different ways; the
  replay checker still shares the sandbox implementation.
- `dataset/grouping.py` joins matched scenarios and identical observed inputs
  into canonical leakage groups.
- `dataset/validate_verified_dataset_v3.py` validates v3 structure, balance,
  model inputs, pairs, and outcome evidence.
- `dataset/generate_experimental_splits_v1.py` creates frozen v3 splits;
  `dataset/validate_experimental_splits_v1.py` independently checks those
  assignments and saves its report to a separate report directory.
- `dataset/generate_challenge_dataset_v1.py` and
  `dataset/validate_challenge_dataset_v1.py` build/check the smaller external
  challenge data and its frozen split.
- `experiments/run_baselines.py` runs all CPU baselines and saves metrics and
  ID-linked predictions. It reads labels only as training targets/evaluation
  truth; text features come from the permitted projection.
- `models/load_model.py`, `hidden_states/`, `probes/`,
  `experiments/direct_answers.py`, and `agent/behavior_log.py` are reusable
  utilities, not evidence that neural experiments have run.

`dataset/generate_dataset.py` is the legacy text-only generator. It is not
execution-backed ground truth. Importing it now has no file-writing side
effect, and its output path refuses to overwrite the historical dataset.
Likewise, the v2 and v3 generators refuse existing paths.
The preserved v2 JSONL itself has 200 email examples labelled partial whose
correction records are not claim-linked under the current rules. Keep v2 for
historical comparison, but use v3 as the primary baseline dataset and do not
treat those v2 email labels as verified ground truth.

## Data flow

1. A generator sets up a fresh `Sandbox`, records initial operational state,
   executes one initiating action, tries recovery tools, and records final
   operational state and audit history separately.
2. The ground-truth checker compares observed evidence with the operational
   state rules. The audit log alone cannot make a complete operational reversal
   partial.
3. Dataset validators check labels, evidence, pairs, input projection, and
   grouping. The reference oracle offers a separate expected-outcome check.
4. Frozen split IDs are loaded by the baseline runner. The canonical group and
   related scenario/input identifiers must not cross each split.
5. A model receives only `environment`, `state_text`, and `action_text` via
   `dataset.model_input.build_model_input`. Labels, action parameters, audit
   logs, final states, and recovery traces are evaluation-only.
6. Text baselines fit their vectorizers and classifiers on that split's v3
   training IDs only. They evaluate on that split's test IDs and, separately,
   on all challenge rows. Challenge rows never fit a model.

## Operational reversibility definitions

The initiating action is outside the horizon. Each recovery-tool call counts
once, including a failed attempt; observation and comparison do not count.

- **Reversible:** all relevant operational state is exactly restored within
  three recovery calls.
- **Partially reversible:** recovery makes observable progress, but the exact
  original operational state is not restored.
- **Irreversible:** no meaningful recovery is achieved within the horizon.

For files, existence and exact contents matter. A stale backup can be partial
only when it restores some original information while remaining different from
the immediate original. For email, undo availability is static; there is no
clock. A correction can mitigate a delivered message but cannot retract it.
For money, cancellation of a pending reservation restores balances and is
reversible. Refunding a settled transfer may restore balances, but the
settlement commitment remains, so it is partial. Audit history is preserved
for analysis but is not itself an operational effect.

Frozen v3 descriptions include legacy second-count phrases for email. Those
numbers do not drive the sandbox and must not be interpreted as a simulated
expiry clock. The frozen dataset was not changed. New generation wording states
static availability directly.

## Baselines and interpretation

Run all fixed baselines with:

```powershell
python -B experiments/run_baselines.py --output-dir results/baselines_v2
```

This uses three existing frozen v3 splits and these baselines:

1. Majority class, with the fixed tie order in the runner.
2. Action-name majority, choosing the most common training label for that
   action.
3. Action-text-only unigram TF-IDF plus logistic regression.
4. State-plus-action unigram TF-IDF plus logistic regression.
5. Shuffled-label control, with 100 predetermined permutations of training
   labels per split. Permutation preserves the training label counts. Test and
   challenge labels are never shuffled.

`action_text` currently contains canonical names such as `edit_file`; it is not
a natural-language description of the proposed action. Therefore the
action-text TF-IDF baseline is an action-name lexical control, not a test of
generalization across natural-language action descriptions.

TF-IDF settings are fixed in code: lowercase word unigrams, `min_df=1`, and
logistic regression with `C=1`, `lbfgs`, and 1,000 maximum iterations. No
hyperparameter tuning is run. Outputs include a manifest, every per-run metric,
summary statistics, and prediction JSONL records with example IDs and true
labels. The shuffle distribution is summarized separately for each split; the
three overlapping splits are not independent dataset replications.

## Future model experiments

The model utilities use Hugging Face `AutoTokenizer` and
`AutoModelForCausalLM`, so the same interface can support Qwen and Llama
families. No weights are downloaded or loaded by tests. Suggested candidates
are Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, subject to model access,
licensing, memory, and assignment approval.

The intended flow is:

1. Load the model explicitly with `models.load_model.load_instruction_model`.
2. Build each prompt from the approved model-input projection.
3. Extract the last non-padding token representation at each layer with
   `hidden_states.extract.extract_last_token_hidden_states` and save it with
   example IDs using `hidden_states.save.save_representations`.
4. Align IDs to each frozen split and fit one L2 logistic-regression probe per
   layer with `probes.train.train_layer_probes`.
5. Compare three-class and reversible-vs-not-reversible metrics with
   `probes.evaluate`.
6. Record direct answers using `experiments.direct_answers`; compare answer
   labels with probe predictions for RQ2.
7. Build a real tool-enabled agent and use `agent.behavior_log` to record the
   pre-action context, model output, checks, confirmation behavior, tool calls,
   and outcome for RQ3. `agent.behavior_metrics` summarizes these observable
   caution proxies without claiming they prove risk understanding.

The project currently lacks model-run scripts that align hidden states across
all models and splits, a complete direct-answer comparison report, a live agent
loop, and a ScienceWorld adapter. Those are still implementation tasks; the
utilities here do not satisfy those experiments by themselves. A 7B/8B model
may not be practical without a suitable GPU or substantial RAM. Tests do not
require those weights.

## Dependencies and reproducibility

`requirements.txt` contains only the fixed scikit-learn dependency used by the
baselines and probe classifier. `requirements-models.txt` contains optional
Transformers and Accelerate dependencies. Install a CPU or CUDA PyTorch build
separately using the official selector appropriate to the computer. This avoids
the old frozen CUDA-only `torch` package pin in the core requirements.

All scripts should be run from the project root. Seeds and model settings are
saved in result manifests. New dataset generators, validators, hidden-state
savers, and experiment runners refuse to overwrite existing artifacts; use a
new versioned output path for repeated runs.

## Limitations and remaining assignment requirements

- The sandbox is deterministic and synthetic. It does not establish how real
  file, email, or payment services behave.
- V3 contains strongly label-associated condition wording. Its excellent
  in-split TF-IDF scores do not establish semantic or causal reasoning.
- The challenge set is small, and its text is generated from known sandbox
  mechanisms. It is not evidence of unseen-mechanism generalization.
- The action field is a canonical operation name rather than a natural-language
  instruction.
- The three v3 splits overlap in scenario information and should be reported
  separately, not as three independent replications.
- No open-weight model, hidden-state extraction, neural probe, RQ2 model answer,
  agent-behavior experiment, or ScienceWorld experiment has been run.
- The assignment still requires two open-weight instruction-tuned models,
  layer-wise hidden-state analysis, direct-answer/probe agreement, a real agent
  behavior experiment, ScienceWorld as a second testbed, and appropriate
  statistical evaluation. No results for those items should be claimed until
  the runs are executed and saved.
