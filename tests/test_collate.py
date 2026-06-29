import torch

from audio_ml.audio.collate import audio_collate_fn
from audio_ml.audio.features import LogMelExtractor


def test_audio_collate_fn() -> None:
    batch = [
        {
            "waveform": torch.randn(16000),
            "sample_rate": 16000,
            "label": 0,
            "path": "a.wav",
        },
        {
            "waveform": torch.randn(24000),
            "sample_rate": 16000,
            "label": 1,
            "path": "b.wav",
        },
    ]

    extractor = LogMelExtractor(sample_rate=16000)

    output = audio_collate_fn(
        batch=batch,
        feature_extractor=extractor,
    )

    assert output["waveforms"].shape == (2, 24000)
    assert output["labels"].dtype == torch.long
    assert output["lengths"].tolist() == [16000, 24000]
    assert output["padding_mask"].shape == (2, 24000)
    assert output["log_mel"].ndim == 3
    assert torch.isfinite(output["log_mel"]).all()
