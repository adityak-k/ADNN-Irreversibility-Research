"""Stable scenario, exact-input, template, and canonical leakage groups."""
import hashlib
import json


def stable_id(prefix, value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def invariant_parameters(action, parameters):
    result = dict(parameters)
    result.pop("recovery_condition", None)
    result.pop("transfer_condition", None)
    return result


def assign_group_ids(records):
    for row in records:
        invariant = invariant_parameters(row["action_text"], row["action_parameters"])
        row["scenario_group_id"] = stable_id(
            "SCENARIO", {"action": row["action_text"], "parameters": invariant})
        row["input_group_id"] = stable_id("INPUT", {
            "environment": row["environment"],
            "state_text": row["state_text"],
            "action_text": row["action_text"],
        })

    parents = list(range(len(records)))

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owners = {}
    for index, row in enumerate(records):
        for key in (row["scenario_group_id"], row["input_group_id"]):
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    members = {}
    for index, row in enumerate(records):
        members.setdefault(find(index), set()).add(row["scenario_group_id"])
    group_ids = {
        root: stable_id("GROUP", sorted(scenario_ids))
        for root, scenario_ids in members.items()
    }
    for index, row in enumerate(records):
        row["canonical_group_id"] = group_ids[find(index)]
    return records
