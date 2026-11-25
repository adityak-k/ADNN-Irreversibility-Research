# Challenge Dataset v1 Validation

Dataset SHA-256: `2cc09123890b34ce9c1c9002fbc596ccb3f782bad5315208b3609c68181cef95`
Validation: **PASSED**
Examples: 240; matched pairs: 120
Independent reference outcomes checked: 240

Canonical-group split seed 20261125: 192 train / 48 test; 96 train groups / 24 test groups.

| Action | Train reversible | Train partial | Train irreversible | Test reversible | Test partial | Test irreversible |
|---|---:|---:|---:|---:|---:|---:|
| edit_file | 16 | 16 | 16 | 4 | 4 | 4 |
| delete_file | 16 | 16 | 16 | 4 | 4 | 4 |
| send_email | 16 | 16 | 16 | 4 | 4 | 4 |
| transfer_money | 16 | 16 | 16 | 4 | 4 | 4 |

## Leakage checks

- `canonical_group_id` crossings: 0
- `scenario_group_id` crossings: 0
- `input_group_id` crossings: 0
- `pair_id` crossings: 0
- Seed replay: PASS

The challenge dataset is separate from v3. Labels were checked with the pure reference oracle; the challenge split does not replace the frozen v3 splits.
