"""Explicit, fail-closed projection from a dataset row to model-visible input."""

MODEL_INPUT_FIELDS = ("environment", "state_text", "action_text")


def build_model_input(record):
    missing = [field for field in MODEL_INPUT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record is missing allowed model input fields: {missing}")
    return {field: record[field] for field in MODEL_INPUT_FIELDS}
