"""Apply a predetermined recovery-tool list within the three-call horizon."""

MAX_RECOVERY_ACTIONS = 3


def run_recovery_plan(tool_names, invoke, successful_results):
    """Invoke at most three tools, stopping after the first reported success."""
    attempts = []
    for tool_name in list(tool_names)[:MAX_RECOVERY_ACTIONS]:
        result = invoke(tool_name)
        attempts.append({"tool": tool_name, "result": result})
        if result in successful_results:
            break
    return attempts
