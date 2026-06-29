from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        target_sample_rate: int = 16000,
    ) -> None:
        self.samples = samples
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, label = self.samples[index]

        audio, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )

        # SoundFile shape: [time, channels]
        waveform = torch.from_numpy(audio).transpose(0, 1)

        # Convert stereo or multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            sample_rate = self.target_sample_rate

        waveform = waveform.squeeze(0)

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "label": label,
            "path": str(Path(path)),
        }
