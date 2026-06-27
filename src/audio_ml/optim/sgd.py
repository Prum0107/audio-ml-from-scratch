from collections.abc import Iterable

import torch


class SGD:
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")

        self.params = list(params)
        self.lr = lr
        self.momentum = momentum

        self.velocities = [
            torch.zeros_like(param)
            for param in self.params
        ]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for param, velocity in zip(
            self.params,
            self.velocities,
        ):
            if param.grad is None:
                continue

            velocity.mul_(self.momentum)
            velocity.add_(param.grad)

            param -= self.lr * velocity