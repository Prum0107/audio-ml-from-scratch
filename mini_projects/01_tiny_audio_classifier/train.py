from functools import partial
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audio_ml.audio.collate import audio_collate_fn
from audio_ml.audio.dataset import AudioClassificationDataset
from audio_ml.audio.features import LogMelExtractor
from audio_ml.models.tiny_audio_cnn import TinyAudioClassifier
from audio_ml.optim.sgd import SGD


SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.05
MOMENTUM = 0.9
SAMPLE_RATE = 16000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_samples(
    root: str,
) -> list[tuple[str, int]]:
    root_path = Path(root)
    samples: list[tuple[str, int]] = []

    for class_directory in sorted(
        root_path.glob("class_*")
    ):
        label = int(
            class_directory.name.split("_")[-1]
        )

        for audio_path in sorted(
            class_directory.glob("*.wav")
        ):
            samples.append(
                (str(audio_path), label)
            )

    if not samples:
        raise RuntimeError(
            f"No audio files found under {root}"
        )

    return samples


def split_samples(
    samples: list[tuple[str, int]],
    validation_ratio: float = 0.2,
) -> tuple[
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    shuffled_samples = samples.copy()

    generator = random.Random(SEED)
    generator.shuffle(shuffled_samples)

    split_index = int(
        len(shuffled_samples)
        * (1.0 - validation_ratio)
    )

    train_samples = shuffled_samples[:split_index]
    validation_samples = shuffled_samples[split_index:]

    return train_samples, validation_samples


def calculate_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    predictions = logits.argmax(dim=1)

    accuracy = (
        predictions == labels
    ).float().mean()

    return float(accuracy)


def train_one_epoch(
    model: TinyAudioClassifier,
    data_loader: DataLoader,
    optimizer: SGD,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        log_mel = batch["log_mel"].unsqueeze(1)
        labels = batch["labels"]

        log_mel = log_mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(log_mel)
        loss = F.cross_entropy(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


@torch.no_grad()
def evaluate(
    model: TinyAudioClassifier,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        log_mel = batch["log_mel"].unsqueeze(1)
        labels = batch["labels"]

        log_mel = log_mel.to(device)
        labels = labels.to(device)

        logits = model(log_mel)

        loss = F.cross_entropy(
            logits,
            labels,
        )

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def main() -> None:
    set_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    samples = collect_samples(
        "data/toy_audio"
    )

    train_samples, validation_samples = (
        split_samples(samples)
    )

    print(
        f"Train samples: {len(train_samples)}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_samples)}"
    )

    train_dataset = AudioClassificationDataset(
        train_samples,
        target_sample_rate=SAMPLE_RATE,
    )

    validation_dataset = (
        AudioClassificationDataset(
            validation_samples,
            target_sample_rate=SAMPLE_RATE,
        )
    )

    feature_extractor = LogMelExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=64,
    )

    collate_function = partial(
        audio_collate_fn,
        feature_extractor=feature_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_function,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_function,
    )

    model = TinyAudioClassifier(
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    best_validation_accuracy = 0.0

    output_directory = Path("outputs")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
        )

        validation_loss, validation_accuracy = (
            evaluate(
                model=model,
                data_loader=validation_loader,
                device=device,
            )
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            checkpoint_path = (
                output_directory
                / "best_model.pt"
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "validation_accuracy": (
                        validation_accuracy
                    ),
                },
                checkpoint_path,
            )

    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()