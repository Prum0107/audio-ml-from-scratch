import torch

from audio_ml.models.tiny_audio_cnn import TinyAudioClassifier


def test_tiny_audio_classifier_output_shape() -> None:
    model = TinyAudioClassifier(num_classes=4)

    inputs = torch.randn(
        8,
        1,
        64,
        101,
    )

    logits = model(inputs)

    assert logits.shape == (8, 4)


def test_tiny_audio_classifier_backward() -> None:
    model = TinyAudioClassifier(num_classes=4)

    inputs = torch.randn(
        2,
        1,
        64,
        101,
    )

    logits = model(inputs)
    loss = logits.mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
