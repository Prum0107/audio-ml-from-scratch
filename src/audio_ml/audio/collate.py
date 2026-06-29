import torch
from torch.nn.utils.rnn import pad_sequence

from audio_ml.audio.features import LogMelExtractor


def audio_collate_fn(
    batch: list[dict],
    feature_extractor: LogMelExtractor,
) -> dict:
    waveforms = [item["waveform"] for item in batch]
    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long,
    )
    lengths = torch.tensor(
        [waveform.shape[0] for waveform in waveforms],
        dtype=torch.long,
    )

    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )

    max_length = padded_waveforms.shape[1]

    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)

    log_mel = feature_extractor(padded_waveforms)

    return {
        "waveforms": padded_waveforms,
        "log_mel": log_mel,
        "labels": labels,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "paths": [item["path"] for item in batch],
    }
