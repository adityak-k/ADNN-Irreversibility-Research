"""Evaluate binary and three-class probe predictions with fixed label order."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


LABELS = ("reversible", "partially_reversible", "irreversible")
BINARY_LABELS = ("reversible", "not_reversible")


def evaluate_predictions(true_labels, predicted_labels, labels=LABELS):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted label counts do not match")
    return {
        "count": len(true_labels),
        "labels": list(labels),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "macro_precision": precision_score(
            true_labels, predicted_labels, labels=list(labels), average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            true_labels, predicted_labels, labels=list(labels), average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            true_labels, predicted_labels, labels=list(labels), average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=list(labels)
        ).tolist(),
    }


def evaluate_binary_reversibility(true_labels, predicted_labels):
    """Score reversible vs. not-reversible predictions from the 3-class task."""
    binary_true = ["reversible" if value == "reversible" else "not_reversible"
                   for value in true_labels]
    binary_predicted = ["reversible" if value == "reversible" else "not_reversible"
                        for value in predicted_labels]
    return evaluate_predictions(binary_true, binary_predicted, BINARY_LABELS)
