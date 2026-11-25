"""Load a causal instruction model through Hugging Face's AutoModel API.

AutoModelForCausalLM supports different model families, including Qwen and
Llama, without model-family-specific project code.
"""


def load_instruction_model(model_name, device="auto"):
    """Load a model and tokenizer when the optional dependencies are installed.

    This function does not download anything until it is explicitly called.
    Use ``device="cpu"`` only when sufficient system RAM is available.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Model loading requires the optional packages in requirements-models.txt "
            "and a separately installed PyTorch build."
        ) from error

    if device == "auto":
        if torch.cuda.is_available():
            load_options = {"device_map": "auto", "torch_dtype": "auto"}
        else:
            raise RuntimeError(
                "No CUDA device is available. For small models, explicitly pass "
                "device='cpu'. A 7B model may require substantial system RAM."
            )
    elif device == "cpu":
        load_options = {"torch_dtype": torch.float32}
    else:
        raise ValueError("device must be 'auto' or 'cpu'")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        **load_options,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model
