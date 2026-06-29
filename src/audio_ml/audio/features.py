import torch
import torchaudio


class LogMelExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
    ) -> None:
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel + 1e-6)
        return log_mel
