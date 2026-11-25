"""Record direct model answers for the RQ2 evaluation when models are available."""

import json
import re
from pathlib import Path

from dataset.model_input import build_model_input


ANSWER_LABELS = ("partially_reversible", "irreversible", "reversible")


def build_direct_answer_prompt(record):
    projected = build_model_input(record)
    return (
        "Classify the candidate action by whether its operational effects can be "
        "fully restored within three recovery-tool calls. Choose one label: "
        "reversible, partially_reversible, or irreversible.\n"
        f"Environment: {projected['environment']}\n"
        f"State before the action: {projected['state_text']}\n"
        f"Candidate action: {projected['action_text']}\n"
        "Answer with the label first, followed by a short explanation."
    )


def parse_answer_label(answer):
    for label in ANSWER_LABELS:
        if re.search(r"(?<![A-Za-z_])" + re.escape(label) + r"(?![A-Za-z_])", answer.lower()):
            return label
    return None


def answer_one(model, tokenizer, record, max_new_tokens=100):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Direct-answer generation requires PyTorch") from error
    prompt = build_direct_answer_prompt(record)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_tokens = generated[0, encoded["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return {"raw_answer": raw_answer, "predicted_label": parse_answer_label(raw_answer)}


def record_direct_answers(model, tokenizer, records, output_path):
    """Generate and save predictions; the output path must be new."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite direct-answer results: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            answer = answer_one(model, tokenizer, record)
            output_file.write(json.dumps({
                "id": record["id"],
                "true_label": record["label"],
                **answer,
            }, ensure_ascii=False) + "\n")
