# Experimental Split Validation Report

Dataset: `dataset/irreversibility_dataset_v3.jsonl` (irreversibility_dataset_v3)
SHA-256: `a7094909662c3158372fd66b8f04abd67311af0c6466978024c475f38848c026`
Validation: **PASSED**

| Seed | Train rows | Test rows | Train groups | Test groups | Cell counts | Leakage | Replay |
|---:|---:|---:|---:|---:|---|---|---|
| 20261120 | 1920 | 480 | 1361 | 303 | PASS | PASS | PASSED |
| 20261121 | 1920 | 480 | 1370 | 294 | PASS | PASS | PASSED |
| 20261122 | 1920 | 480 | 1370 | 294 | PASS | PASS | PASSED |

## Action/class counts

### Seed 20261120

Training (target 160 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 160 | 160 | 160 |
| delete_file | 160 | 160 | 160 |
| send_email | 160 | 160 | 160 |
| transfer_money | 160 | 160 | 160 |

Testing (target 40 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 40 | 40 | 40 |
| delete_file | 40 | 40 | 40 |
| send_email | 40 | 40 | 40 |
| transfer_money | 40 | 40 | 40 |

### Seed 20261121

Training (target 160 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 160 | 160 | 160 |
| delete_file | 160 | 160 | 160 |
| send_email | 160 | 160 | 160 |
| transfer_money | 160 | 160 | 160 |

Testing (target 40 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 40 | 40 | 40 |
| delete_file | 40 | 40 | 40 |
| send_email | 40 | 40 | 40 |
| transfer_money | 40 | 40 | 40 |

### Seed 20261122

Training (target 160 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 160 | 160 | 160 |
| delete_file | 160 | 160 | 160 |
| send_email | 160 | 160 | 160 |
| transfer_money | 160 | 160 | 160 |

Testing (target 40 per cell):

| Action | Reversible | Partial | Irreversible |
|---|---:|---:|---:|
| edit_file | 40 | 40 | 40 |
| delete_file | 40 | 40 | 40 |
| send_email | 40 | 40 | 40 |
| transfer_money | 40 | 40 | 40 |

## Validation scope

Independent checks covered exact ID partitioning, duplicates/missing IDs, canonical/scenario/input group crossings, matched-pair crossings, action/class targets, dataset hashes, metadata, and deterministic same-seed MILP replay. No models or embeddings were used.
