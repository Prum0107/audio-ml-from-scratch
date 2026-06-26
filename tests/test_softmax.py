import numpy as np
import torch

from audio_ml.foundations.softmax import softmax


def test_softmax_sums_to_one() -> None:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )

    probabilities = softmax(x, axis=-1)

    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_softmax_is_numerically_stable() -> None:
    x = np.array([1000.0, 1001.0, 1002.0])

    probabilities = softmax(x)

    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(), 1.0)


def test_softmax_is_shift_invariant() -> None:
    x = np.array([1.0, 2.0, 3.0])

    original = softmax(x)
    shifted = softmax(x + 1000.0)

    assert np.allclose(original, shifted)


def test_softmax_matches_pytorch() -> None:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected = torch.softmax(
        torch.tensor(x, dtype=torch.float64),
        dim=-1,
    ).numpy()

    actual = softmax(x, axis=-1)

    assert np.allclose(actual, expected)