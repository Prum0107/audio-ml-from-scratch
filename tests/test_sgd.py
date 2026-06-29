import torch

from audio_ml.optim.sgd import SGD


def test_sgd_updates_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.5])

    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()

    expected = torch.tensor([0.95])

    assert torch.allclose(parameter.data, expected)


def test_zero_grad_resets_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.5])

    optimizer = SGD([parameter], lr=0.1)
    optimizer.zero_grad()

    assert torch.allclose(
        parameter.grad,
        torch.zeros_like(parameter.grad),
    )


def test_sgd_with_momentum() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SGD([parameter], lr=0.1, momentum=0.9)

    parameter.grad = torch.tensor([1.0])
    optimizer.step()

    assert torch.allclose(
        parameter.data,
        torch.tensor([0.9]),
    )

    parameter.grad = torch.tensor([1.0])
    optimizer.step()

    expected = torch.tensor([0.71])

    assert torch.allclose(parameter.data, expected)
