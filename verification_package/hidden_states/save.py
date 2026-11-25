"""Save extracted representations without replacing earlier experiment files."""

from pathlib import Path

import numpy as np


def save_representations(output_path, example_ids, representations):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite hidden states: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(representations, dtype=np.float32)
    if array.ndim != 3 or len(example_ids) != array.shape[0]:
        raise ValueError("Expected one (layers, hidden_size) array per example ID")
    with output_path.open("xb") as output_file:
        np.savez_compressed(
            output_file,
            example_ids=np.asarray(example_ids, dtype=str),
            hidden_states=array,
        )
