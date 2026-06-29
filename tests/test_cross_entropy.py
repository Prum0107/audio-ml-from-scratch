import numpy as np
import torch
import torch.nn.functional as F

from audio_ml.foundations.cross_entropy import (
    cross_entropy,
    cross_entropy_backward,
)


def test_cross_entropy_matches_pytorch() -> None:
    logits = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.1, 1.5, 2.2],
        ],
        dtype=np.float64,
    )

    targets = np.array([0, 2])

    actual = cross_entropy(logits, targets)

    expected = F.cross_entropy(
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.long),
    ).item()

    assert np.allclose(actual, expected)


def test_cross_entropy_handles_large_logits() -> None:
    logits = np.array(
        [
            [1000.0, 1001.0, 1002.0],
            [2000.0, 1999.0, 1998.0],
        ]
    )

    targets = np.array([2, 0])

    loss = cross_entropy(logits, targets)

    assert np.isfinite(loss)


def test_cross_entropy_backward_shape() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.2, 0.1],
        ]
    )

    targets = np.array([2, 0])

    gradient = cross_entropy_backward(logits, targets)

    assert gradient.shape == logits.shape


def test_gradient_rows_sum_to_zero() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.2, 0.1],
        ]
    )

    targets = np.array([2, 0])

    gradient = cross_entropy_backward(logits, targets)

    assert np.allclose(gradient.sum(axis=1), 0.0)


def test_cross_entropy_gradient_matches_finite_difference() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    targets = np.array([2, 0])
    epsilon = 1e-5

    analytical_gradient = cross_entropy_backward(logits, targets)
    numerical_gradient = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            logits_plus = logits.copy()
            logits_minus = logits.copy()

            logits_plus[i, j] += epsilon
            logits_minus[i, j] -= epsilon

            loss_plus = cross_entropy(logits_plus, targets)
            loss_minus = cross_entropy(logits_minus, targets)

            numerical_gradient[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    assert np.allclose(
        analytical_gradient,
        numerical_gradient,
        atol=1e-6,
    )
