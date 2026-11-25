"""Fit one fixed L2 logistic-regression probe per hidden-state layer."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_layer_probes(hidden_states, labels, seed=0, c_value=1.0):
    """Train probes from an array shaped (examples, layers, hidden_size)."""
    hidden_states = np.asarray(hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (examples, layers, hidden_size)")
    if len(labels) != hidden_states.shape[0]:
        raise ValueError("The number of labels must match the number of examples")

    probes = []
    for layer_index in range(hidden_states.shape[1]):
        probe = LogisticRegression(
            penalty="l2",
            C=c_value,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        probe.fit(hidden_states[:, layer_index, :], labels)
        probes.append(probe)
    return probes
