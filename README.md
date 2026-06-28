<div align="center">

🎧 Audio ML from Scratch

Building audio machine learning systems from first principles

A hands-on repository for implementing core audio machine learning components, training pipelines, and small models from scratch.

</div>

⸻

✨ Overview

This repository documents my journey of learning audio machine learning by implementing important components step by step.

The goal is not only to use existing libraries, but also to understand:

* how classification losses work;
* how gradients update parameters;
* how variable-length audio is processed;
* how log-Mel features are constructed;
* how a complete audio training pipeline is built;
* how to make experiments reproducible.

⸻

🚀 Week 1 Project

Reproducible Tiny Audio Classifier

The first milestone is a complete audio classification pipeline:

Audio Waveform
      ↓
Log-Mel Spectrogram
      ↓
Dataset + collate_fn
      ↓
Tiny CNN
      ↓
Logits
      ↓
Cross Entropy
      ↓
Backpropagation
      ↓
SGD with Momentum
      ↓
Training + Evaluation

The purpose of this project is not to achieve state-of-the-art accuracy.

The purpose is to verify that every stage of the training pipeline works correctly and can be reproduced.

⸻

🧠 What I Implemented

<table>
<tr>
<td width="50%" valign="top">

Mathematical Foundations

* Numerically stable Softmax
* Cross Entropy forward pass
* Cross Entropy backward pass
* Finite-difference gradient checking
* Comparison with PyTorch results

</td>
<td width="50%" valign="top">

Optimization

* Vanilla SGD
* Momentum
* Parameter updates
* Gradient reset
* Optimizer unit tests

</td>
</tr>
<tr>
<td width="50%" valign="top">

Audio Pipeline

* Audio loading with SoundFile
* Mono conversion
* Sample-rate normalization
* Variable-length padding
* Padding masks
* Log-Mel extraction

</td>
<td width="50%" valign="top">

Model and Training

* Tiny CNN classifier
* Global average pooling
* Training loop
* Validation loop
* Accuracy calculation
* Best-checkpoint saving

</td>
</tr>
</table>

⸻

🏗️ Repository Structure

audio-ml-from-scratch/
├── src/
│   └── audio_ml/
│       ├── foundations/
│       │   ├── softmax.py
│       │   └── cross_entropy.py
│       ├── optim/
│       │   └── sgd.py
│       ├── audio/
│       │   ├── features.py
│       │   ├── dataset.py
│       │   └── collate.py
│       └── models/
│           └── tiny_audio_cnn.py
│
├── mini_projects/
│   └── 01_tiny_audio_classifier/
│       ├── generate_toy_audio.py
│       ├── train_toy_classifier.py
│       └── train.py
│
├── tests/
├── pyproject.toml
└── README.md

⸻

📐 Mathematical Foundations

Softmax

For logits (z), Softmax converts raw class scores into probabilities:

[
p_i =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

For numerical stability:

[
p_i =
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
]

Cross Entropy

For the correct class (y):

[
L = -\log p_y
]

The gradient with respect to the logits is:

[
\frac{\partial L}{\partial z_i}

p_i-\mathbb{1}(i=y)
]

SGD with Momentum

[
v_t = \mu v_{t-1} + g_t
]

[
\theta_t = \theta_{t-1} - \eta v_t
]

⸻

🎵 Audio Feature Pipeline

The raw waveform is converted into a log-Mel spectrogram:

Waveform
   ↓
STFT
   ↓
Power Spectrogram
   ↓
Mel Filter Bank
   ↓
Log Compression
   ↓
Log-Mel Spectrogram

Typical input shape:

[B, time]

Log-Mel output shape:

[B, n_mels, frames]

CNN input shape:

[B, 1, n_mels, frames]

⸻

🧩 Model Architecture

Log-Mel Spectrogram
        ↓
Conv2D: 1 → 16
        ↓
ReLU
        ↓
MaxPool2D
        ↓
Conv2D: 16 → 32
        ↓
ReLU
        ↓
Global Average Pooling
        ↓
Linear Layer
        ↓
Class Logits

The model outputs logits directly.

Softmax is not applied before Cross Entropy during training.

⸻

⚙️ Installation

Create a Conda environment:

conda create -n audio-ml python=3.11 -y
conda activate audio-ml

Install the project:

python -m pip install -e .

⸻

🧪 Testing

Run all unit tests:

python -m pytest -q

The test suite covers:

* Softmax numerical stability
* Shift invariance
* Cross Entropy values
* Analytical gradients
* Finite-difference gradient checks
* SGD updates
* Momentum behavior
* Audio padding
* Log-Mel extraction
* CNN output shape
* Backpropagation

⸻

🎛️ Generate the Toy Audio Dataset

python mini_projects/01_tiny_audio_classifier/generate_toy_audio.py

The dataset contains four synthetic frequency classes:

Class	Frequency
Class 0	220 Hz
Class 1	440 Hz
Class 2	880 Hz
Class 3	1760 Hz

The generated dataset is stored under:

data/toy_audio/

The data is not committed because it can be regenerated locally.

⸻

🏋️ Train the Classifier

python mini_projects/01_tiny_audio_classifier/train.py

The training script includes:

* deterministic random seeds;
* train-validation splitting;
* mini-batch loading;
* log-Mel extraction;
* Cross Entropy training;
* SGD with momentum;
* validation evaluation;
* best-checkpoint saving.

⸻

📊 Experimental Results

Model	Feature	Optimizer	Train Accuracy	Validation Accuracy
Tiny CNN	Log-Mel	SGD + Momentum	TODO	TODO

Replace TODO with the actual results from your experiment.

⸻

🔍 Experimental Observations

* The Tiny CNN successfully learned frequency-specific patterns.
* Training loss decreased steadily during optimization.
* Validation accuracy was high because the synthetic classes were intentionally separable.
* The 8-sample overfitting test helped verify that the pipeline could learn correctly.
* The most important engineering issue was audio backend compatibility.
* SoundFile was used for WAV input and output, while Torchaudio was used for feature extraction.

⸻

📝 Week 1 Reflection

<details>
<summary><strong>Week 1: Building a Reproducible Audio Classification Pipeline</strong></summary>
<br>

During the first week, I implemented a complete audio classification pipeline from mathematical foundations to model training and evaluation.

I first implemented a numerically stable Softmax function and multiclass Cross Entropy loss using NumPy. I then derived the Cross Entropy gradient with respect to logits and verified it using finite-difference gradient checking and PyTorch reference implementations.

After that, I implemented SGD with momentum and tested whether model parameters were updated correctly. For the audio pipeline, I added waveform loading, mono conversion, sample-rate normalization, variable-length padding, padding masks, and log-Mel spectrogram extraction.

I used a small convolutional neural network because the purpose of the project was not to maximize classification accuracy, but to validate the entire training system. I also used an 8-sample overfitting test to verify that the model, loss function, gradient flow, and optimizer were working correctly.

One issue I encountered was the compatibility between Torchaudio, TorchCodec, and FFmpeg. I solved this by using SoundFile for WAV loading and saving.

Next week, I will implement Layer Normalization and scaled dot-product attention, with particular attention to tensor shapes, masking behavior, and numerical stability.

</details>

⸻

🗺️ Roadmap

* Numerically stable Softmax
* Cross Entropy forward pass
* Cross Entropy backward pass
* Finite-difference gradient checking
* SGD with momentum
* Audio Dataset
* Variable-length collate function
* Padding mask
* Log-Mel extraction
* Tiny CNN classifier
* Training and validation loop
* Checkpoint saving
* Record final experiment results
* Layer Normalization
* Scaled Dot-Product Attention
* Attention Masking
* Multi-Head Attention
* Tiny Audio Transformer
* Audio-Text Retrieval

⸻

🎯 Long-Term Goal

The long-term goal of this repository is to progress from basic implementations to research-oriented audio systems:

Foundations
    ↓
Audio Classification
    ↓
Attention and Transformers
    ↓
Contrastive Learning
    ↓
Audio-Text Retrieval
    ↓
Long-Audio Understanding

⸻

<div align="center">

Built step by step for Audio AI learning and research preparation

</div>