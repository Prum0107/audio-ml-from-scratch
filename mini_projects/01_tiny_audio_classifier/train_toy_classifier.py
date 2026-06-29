import torch
import torch.nn.functional as F
from torch import nn

from audio_ml.optim.sgd import SGD


def create_toy_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    class_0 = torch.randn(100, 2) + torch.tensor([-2.0, -2.0])
    class_1 = torch.randn(100, 2) + torch.tensor([2.0, -2.0])
    class_2 = torch.randn(100, 2) + torch.tensor([0.0, 2.0])

    inputs = torch.cat([class_0, class_1, class_2], dim=0)

    targets = torch.cat(
        [
            torch.zeros(100, dtype=torch.long),
            torch.ones(100, dtype=torch.long),
            torch.full((100,), 2, dtype=torch.long),
        ]
    )

    return inputs, targets


def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean())


def main() -> None:
    inputs, targets = create_toy_data()

    model = nn.Linear(2, 3)

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
    )

    for epoch in range(100):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            accuracy = calculate_accuracy(logits, targets)

            print(f"epoch={epoch:03d} loss={loss.item():.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
