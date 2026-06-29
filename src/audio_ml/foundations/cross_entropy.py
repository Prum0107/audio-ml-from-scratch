import numpy as np

from audio_ml.foundations.softmax import softmax


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute mean muticlass cross-entropy loss."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")

    if targets.ndim != 1:
        raise ValueError("logits and targets must have the same batch size")

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")

    batch_size = logits.shape[0]

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))

    correct_class_logits = shifted_logits[
        np.arange(batch_size),
        targets,
    ]

    losses = log_sum_exp - correct_class_logits

    return float(np.mean(losses))


def cross_entropy_backward(
    logits: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute the gradient of mean cross-entropy with respect to logits."""

    batch_size = logits.shape[0]

    probabilities = softmax(logits, axis=1)
    gradient = probabilities.copy()

    gradient[np.arange(batch_size), targets] -= 1.0
    gradient /= batch_size

    return gradient
