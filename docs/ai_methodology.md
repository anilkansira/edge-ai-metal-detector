# AI Methodology

This project uses a lightweight Edge AI approach rather than a deep neural network.

## Stages

1. Synthetic PI waveform generation
2. Ground balance compensation
3. Feature extraction
4. Gaussian probabilistic classification
5. Verified waveform memory fusion
6. Confidence estimation

## Why Waveform Memory?

Verified waveform memory allows the model to adapt using confirmed field or lab samples. A new waveform is compared to verified examples using weighted kernel similarity, with the 8 PI waveform samples receiving the highest weight.
