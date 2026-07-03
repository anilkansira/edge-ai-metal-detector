# System Architecture

The project is structured as a modular signal-processing and Edge AI package.

```text
materials.py           Physical metallic target parameters
soil.py                Soil and mineralization profiles
pulse_model.py         PI pulse response and waveform generation
ground_balance.py      Ground response compensation
feature_extraction.py  38-dimensional feature vector extraction
classifier.py          Gaussian probabilistic classifier
waveform_memory.py     Verified waveform memory and kernel similarity
dataset_generator.py   Synthetic waveform dataset generation
pipeline.py            End-to-end training and prediction pipeline
```

## Data Flow

```text
Target parameters
    +
Soil profile
    ↓
Pulse response simulation
    ↓
Ground balance compensation
    ↓
Feature extraction
    ↓
Probabilistic classifier
    ↓
Verified waveform memory fusion
    ↓
Final prediction
```

## Design Rationale

The implementation avoids heavy deep-learning dependencies to remain lightweight and edge-oriented. The classifier is intentionally interpretable and can be inspected through feature distributions, memory similarity, and final confidence factors.
