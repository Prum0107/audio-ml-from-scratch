import torch
from torch import nn


class TinyAudioClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(
            in_features=32,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits
