from pathlib import Path

import torch
import soundfile as sf


SAMPLE_RATE = 16000
NUM_CLASSES = 4
SAMPLES_PER_CLASS = 20

CLASS_FREQUENCIES = {
    0: 220.0,
    1: 440.0,
    2: 880.0,
    3: 1760.0,
}


def create_waveform(
    frequency: float,
    duration: float,
    sample_rate: int,
    noise_level: float = 0.02,
) -> torch.Tensor:
    num_samples = int(duration * sample_rate)

    time = torch.arange(num_samples) / sample_rate

    waveform = torch.sin(
        2 * torch.pi * frequency * time
    )

    noise = noise_level * torch.randn_like(waveform)
    waveform = waveform + noise

    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    return waveform.unsqueeze(0)


def main() -> None:
    torch.manual_seed(42)

    output_root = Path(
        "data/toy_audio"
    )

    for label in range(NUM_CLASSES):
        class_directory = output_root / f"class_{label}"
        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        frequency = CLASS_FREQUENCIES[label]

        for index in range(SAMPLES_PER_CLASS):
            duration = 0.8 + 0.02 * index

            waveform = create_waveform(
                frequency=frequency,
                duration=duration,
                sample_rate=SAMPLE_RATE,
            )

            output_path = (
                class_directory
                / f"sample_{index:03d}.wav"
            )

            sf.write(
                file=str(output_path),
                data=waveform.squeeze(0).numpy(),
                samplerate=SAMPLE_RATE,
            )

    print(
        f"Created {NUM_CLASSES * SAMPLES_PER_CLASS} "
        f"audio files in {output_root}"
    )


if __name__ == "__main__":
    main()