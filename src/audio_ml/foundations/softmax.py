import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute a numerically stable softmax along the given axis.
    """
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
