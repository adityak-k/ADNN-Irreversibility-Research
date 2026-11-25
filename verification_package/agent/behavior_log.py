"""Save agent behavior for later RQ3 analysis without exposing labels to the agent."""

import json
from pathlib import Path

from dataset.model_input import build_model_input


def build_agent_task(record):
    """Return only the pre-action fields permitted as agent-visible context."""
    projected = build_model_input(record)
    return {
        "environment": projected["environment"],
        "state_text": projected["state_text"],
        "action_text": projected["action_text"],
    }


def save_agent_traces(output_path, traces):
    """Save recorded traces once; records include outcomes only for evaluation."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite agent traces: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        for trace in traces:
            required = {
                "id", "agent_input", "agent_output", "tool_calls", "label",
                "pre_action_checks", "confirmation_requested", "action_executed",
            }
            missing = required - set(trace)
            if missing:
                raise ValueError(f"Agent trace is missing fields: {sorted(missing)}")
            if set(trace["agent_input"]) != {"environment", "state_text", "action_text"}:
                raise ValueError("Agent input must contain only the approved projection")
            output_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
