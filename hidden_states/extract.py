"""Extract the last non-padding token representation from every model layer."""

from dataset.model_input import build_model_input


def format_model_input(record):
    """Turn the approved projection into readable text for the tokenizer."""
    projected = build_model_input(record)
    return (
        f"Environment: {projected['environment']}\n"
        f"State: {projected['state_text']}\n"
        f"Candidate action: {projected['action_text']}"
    )


def extract_last_token_hidden_states(model, tokenizer, text, max_length=1024):
    """Return one real hidden-state vector per embedding/layer output.

    The returned NumPy array has shape ``(layers + 1, hidden_size)``. Index 0
    is the embedding output; the final index is the model's final layer output.
    """
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)

    if not output.hidden_states:
        raise ValueError("The model returned no hidden states")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        last_token_index = encoded["input_ids"].shape[1] - 1
    else:
        token_positions = torch.nonzero(attention_mask[0], as_tuple=False)
        if token_positions.numel() == 0:
            raise ValueError("The tokenizer returned an empty input")
        last_token_index = int(token_positions[-1].item())

    layer_vectors = [
        layer_output[0, last_token_index].detach().float().cpu()
        for layer_output in output.hidden_states
    ]
    return torch.stack(layer_vectors).numpy()


def extract_dataset_hidden_states(model, tokenizer, records, max_length=1024):
    """Extract representations and IDs in dataset order using allowed fields."""
    representations = []
    example_ids = []
    for record in records:
        representations.append(
            extract_last_token_hidden_states(
                model, tokenizer, format_model_input(record), max_length=max_length
            )
        )
        example_ids.append(record["id"])
    return example_ids, representations
