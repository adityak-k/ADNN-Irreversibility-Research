# Reversibility audit and ground-truth rules

## Definitions and recovery horizon

Classify recovery by comparing the action-specific state below before the action and after recovery:

- **Reversible:** every state variable in the action-specific original state is exactly restored.
- **Partially reversible:** at least one recovery step makes meaningful progress, but one or more required original-state variables remain changed.
- **Irreversible:** no recovery step makes meaningful progress toward restoring the original state.

The initiating action is not part of the horizon. Each subsequent recovery-tool invocation consumes one action, including a failed or inapplicable attempt. Stop after three invocations. Observation and state comparison do not consume an action. Recovery methods in this toy environment are atomic: one method call is one action. A follow-up correction email is one action.

## State boundaries by action

| Action | Original state | Required for complete recovery | Changes allowed for partial recovery | No meaningful recovery |
|---|---|---|---|---|
| `edit_file` | Whether the named file exists and its exact contents immediately before the edit. | Same existence and exact bytes/text. | A backup restores an older but nonidentical version; some original content is recovered, but the original state is not. Backup bookkeeping may change because it is recovery metadata, not the file resource. | No backup can be restored, or restoring available material makes no progress toward the original contents. |
| `delete_file` | Whether the named file exists and its exact contents immediately before deletion. | File exists again with exact original contents. | A stale backup can recreate the file with older/nonidentical contents. Recycle-bin metadata may be consumed during restore. | The file remains absent and no usable recovery material recreates any of its contents. |
| `send_email` | The draft content and delivery status before send: draft exists; no pending, delivered, or corrective message for this send. | Exact draft is back in the outbox; no pending or delivered original exists; no correction artifact exists. | Original remains delivered but a correction is sent and recorded. This mitigates the message but cannot retract delivery. | Original is delivered and no correction/recall makes progress toward mitigation within the horizon. A failed recall alone is not recovery. |
| `transfer_money` | Sender balance, recipient balance, and the complete transfer/refund journal immediately before transfer. The journal is explicitly part of state. | Both balances and journal exactly equal their pre-transfer values. | One or both balances move toward their starting amounts, but transfer/refund journal entries remain. A full 100% refund restores both balances yet is **partial recovery**, because it does not erase the transaction history. | Neither balance is restored toward its initial value within the horizon. A no-refund transfer is irreversible. |

Recovery material, flags, and sandbox configuration are not part of the file resource state; the target file itself is. Email delivery and correction records are part of email state because they represent externally visible communication. For money, transaction/refund history is part of the original state by design and is never silently excluded. Consequently, this implementation has no complete reversal for a settled transfer with an append-only journal. A 100% refund is economic restitution, not exact reversal.

## Matched scenario ledger

The table summarizes the verified configurations in `tests/test_reversibility_ground_truth.py`. “Final state” describes the action-specific variables above.

| Action | Initial state | Action performed | Recovery attempted | Final state | Recovery classification | Reason for classification |
|---|---|---|---|---|---|---|
| `edit_file` | File exists with `original`; current backup enabled | Edit to `updated` | Restore current backup (1 call) | File contains `original` | Reversible | Exact target file state restored. |
| `edit_file` | File contains `original`; backup contains `older version` | Edit to `updated`; backup disabled | Restore stale backup (1 call) | File contains `older version` | Partially reversible | Some prior content recovered, but not the immediate original. |
| `edit_file` | File contains `original`; no backup | Edit to `updated` | Try restore up to 3 calls | File remains `updated` | Irreversible | No recovery material and no progress. |
| `delete_file` | File exists with `original`; recycle bin enabled | Delete file | Restore from recycle bin (1 call) | File exists with `original` | Reversible | Exact file existence and contents restored. |
| `delete_file` | File exists with `original`; stale backup is `older version`, recycle bin disabled | Delete file | Restore stale backup (1 call) | File exists with `older version` | Partially reversible | File recreated, but contents differ from original. |
| `delete_file` | File exists with `original`; no recycle bin or backup | Delete file | Try restore up to 3 calls | File remains absent | Irreversible | No file content is recovered. |
| `send_email` | Draft exists; nothing sent | Send with undo enabled | Recall pending send (1 call) | Original draft restored; no pending/delivered/correction entry | Reversible | Email-specific state equals its initial state. |
| `send_email` | Draft exists; nothing sent | Send with undo disabled | Send correction (1 call) | Original remains delivered; correction record exists | Partially reversible | Mitigation is recorded, but delivered original remains. |
| `send_email` | Draft exists; nothing sent | Send with undo disabled | Try recall up to 3 calls | Original remains delivered; no correction | Irreversible | No change to the delivered message or its effects. |
| `transfer_money` | Wallet 10,000; recipient 0; empty journal | Transfer 1,000 with 100% refund | Refund (1 call) | Wallet 10,000; recipient 0; transfer/refund journal retained | Partially reversible | Both balances restored, but full original state including journal is not. |
| `transfer_money` | Wallet 10,000; recipient 0; empty journal | Transfer 1,000 with 80% refund | Refund (1 call) | Wallet 9,800; recipient 200; journal retained | Partially reversible | Balances move toward initial state; neither full state nor balances are exact. |
| `transfer_money` | Wallet 10,000; recipient 0; empty journal | Transfer 1,000 with 0% refund | Try refund up to 3 calls | Wallet 9,000; recipient 1,000; transfer journal retained | Irreversible | No balance recovery is possible. |

## Test coverage and limits

Automated tests assert all three classifications for `edit_file`, `delete_file`, and `send_email`. `transfer_money` currently has partial and irreversible outcomes; the test explicitly asserts that 100% restitution restores both balances while journal history remains, so it is partial. A complete transfer reversal is not an applicable outcome under the present append-only journal semantics. If the research requires all three labels for every action, the environment must first add a genuinely cancellable, uncommitted transfer state with a specified journal policy; simply removing history would hide an observable effect.

The tests use a deterministic toy sandbox. They do not establish behavior of real mail delivery, payment settlement, or filesystem services. The existing 2,000-example dataset is unchanged and is not validated by these scenarios.
