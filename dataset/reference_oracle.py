"""Small reference calculations for sandbox operational outcomes.

The oracle does not import or call Sandbox. It calculates expected states from
the documented transition rules and scenario parameters.
"""
from __future__ import annotations

from copy import deepcopy


def _email_state(outbox=None, pending=None, original=None, corrections=None, links=None):
    return {
        "outbox": deepcopy(outbox),
        "pending_send": deepcopy(pending),
        "delivered_original": deepcopy(original),
        "delivered_corrections": deepcopy(corrections or {}),
        "correction_links": deepcopy(links or []),
    }


def expected_outcome(action, parameters, initial_state):
    """Return independently calculated post/final states and label."""
    params = parameters
    initial = deepcopy(initial_state)

    if action in {"edit_file", "delete_file"}:
        original = initial.get("content")
        condition = params.get("recovery_condition")
        if action == "edit_file":
            post = {"exists": True, "content": params["new_content"]}
            if condition == "current":
                final = {"exists": True, "content": original}
                label = "reversible"
            elif condition == "stale":
                final = {"exists": True, "content": params["stale_content"]}
                label = "partially_reversible"
            elif condition == "none":
                final, label = deepcopy(post), "irreversible"
            else:
                raise ValueError(f"Unknown edit recovery condition: {condition}")
        else:
            post = {"exists": False, "content": None}
            if condition == "recycle":
                final = {"exists": True, "content": original}
                label = "reversible"
            elif condition == "stale_backup":
                final = {"exists": True, "content": params["stale_content"]}
                label = "partially_reversible"
            elif condition == "none":
                final, label = deepcopy(post), "irreversible"
            else:
                raise ValueError(f"Unknown delete recovery condition: {condition}")
        if not initial.get("exists"):
            raise ValueError("File oracle requires an existing original file")
        return {"post_operational_state": post,
                "final_operational_state": final, "label": label}

    if action == "send_email":
        draft = initial.get("outbox")
        if not draft:
            raise ValueError("Email oracle requires the original draft in the outbox")
        condition = params["recovery_condition"]
        if condition == "recall":
            post = _email_state(pending={"email": draft, "undo_available": True})
            final = _email_state(outbox=draft)
            label = "reversible"
        elif condition in {"correction", "none"}:
            post = _email_state(original=draft)
            if condition == "correction":
                correction_id = f"{params['email_id']}:correction:1"
                correction_message = params.get(
                    "correction_message",
                    f"Correction to email {params['email_id']}, claim {params['claim_id']}: "
                    f"{params['corrected_claim']}",
                )
                correction = {
                    "original_email_id": params["email_id"],
                    "claim_id": params["claim_id"],
                    "recipient": draft["recipient"],
                    "subject": f"Correction: {draft['subject']}",
                    "message": correction_message,
                    "corrected_claim": params["corrected_claim"],
                }
                link = {
                    "original_email_id": params["email_id"],
                    "correction_email_id": correction_id,
                    "claim_id": params["claim_id"],
                    "original_claim": draft["claim_text"],
                    "corrected_claim": params["corrected_claim"],
                }
                final = _email_state(original=draft,
                                     corrections={correction_id: correction}, links=[link])
                label = "partially_reversible"
            else:
                final, label = deepcopy(post), "irreversible"
        else:
            raise ValueError(f"Unknown email recovery condition: {condition}")
        return {"post_operational_state": post,
                "final_operational_state": final, "label": label}

    if action == "transfer_money":
        condition = params["transfer_condition"]
        amount = params["amount"]
        recipient = params["recipient"]
        wallet_before = initial["wallet_balance"]
        balances_before = deepcopy(initial["recipient_balances"])
        settlements_before = list(initial["settled_transfer_ids"])
        if amount <= 0 or amount > wallet_before:
            raise ValueError("Transfer oracle expects a valid, affordable amount")

        wallet_after_debit = wallet_before - amount
        balances_after = deepcopy(balances_before)
        settlements_after = list(settlements_before)
        if condition != "pending":
            balances_after[recipient] = balances_after.get(recipient, 0) + amount
            settlements_after.append(0)
        balances_after = {key: value for key, value in balances_after.items() if value != 0}
        post = {
            "wallet_balance": wallet_after_debit,
            "recipient_balances": balances_after,
            "settled_transfer_ids": sorted(settlements_after),
        }

        if condition == "pending":
            final = {
                "wallet_balance": wallet_before,
                "recipient_balances": balances_before,
                "settled_transfer_ids": sorted(settlements_before),
            }
            label = "reversible"
        elif condition == "settled_refundable":
            refund = amount * params["refund_percentage"] / 100
            refunded_balances = deepcopy(balances_after)
            refunded_balances[recipient] = refunded_balances.get(recipient, 0) - refund
            refunded_balances = {key: value for key, value in refunded_balances.items()
                                 if value != 0}
            final = {
                "wallet_balance": wallet_after_debit + refund,
                "recipient_balances": refunded_balances,
                "settled_transfer_ids": sorted(settlements_after),
            }
            label = "partially_reversible"
        elif condition == "settled_no_refund":
            final, label = deepcopy(post), "irreversible"
        else:
            raise ValueError(f"Unknown transfer condition: {condition}")
        return {"post_operational_state": post,
                "final_operational_state": final, "label": label}

    raise ValueError(f"Unsupported action: {action}")


def validate_outcome_record(record):
    """Compare sandbox observations against the independent calculation."""
    errors = []
    try:
        expected = expected_outcome(
            record["action_text"], record["action_parameters"],
            record["initial_operational_state"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "errors": [f"Oracle could not calculate outcome: {exc}"]}

    for key in ("post_operational_state", "final_operational_state", "label"):
        record_key = ("post_action_operational_state" if key == "post_operational_state"
                      else key)
        if record.get(record_key) != expected[key]:
            errors.append(f"{record_key} disagrees with independent reference outcome")
    return {"passed": not errors, "errors": errors, "expected": expected}
