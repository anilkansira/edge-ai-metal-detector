# Pulse Induction Metal Classifier with Edge AI Waveform Memory

A lightweight signal-processing and Edge AI framework for metallic target classification using **pulse induction (PI) decay waveforms**, **ground balance compensation**, **Gaussian probabilistic classification**, and **verified waveform memory**.

This repository focuses on the engineering and algorithmic side of a pulse induction metal detector. It intentionally excludes visualization/game-engine code and presents the system as a reusable signal-processing and classification package.

---

## Why This Project Matters

Pulse induction metal detectors operate under challenging field conditions. Soil mineralization, target depth, coil height, electronics noise, and ground balance error can strongly affect the measured waveform. This project models that problem as a compact Edge AI pipeline:

```text
PI decay waveform
    ↓
ground balance compensation
    ↓
feature extraction
    ↓
probabilistic classification
    ↓
verified waveform memory
    ↓
final target prediction
```

The goal is not to claim hardware-ready performance, but to demonstrate a clean engineering workflow for sensor modelling, waveform analysis, and adaptive classification.

---

## Key Features

- Pulse induction decay waveform generation
- 8-sample PI response modelling
- Soil and ground response modelling
- Ground balance compensation
- Feature extraction from decay waveforms
- Gaussian probabilistic classifier
- Verified waveform memory for adaptive calibration
- Synthetic dataset generation
- CLI interface with JSON output support
- Evaluation and reporting examples
- Unit tests and GitHub Actions workflow
- Mathematical and architectural documentation

---

## Installation

Clone the repository:

```bash
git clone https://anilkansira/pulse-induction-metal-classifier.git
cd pulse-induction-metal-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the project in editable mode:

```bash
pip install -e ".[dev,plot]"
```

---

## CLI Usage

Run a single prediction:

```bash
pi-classifier predict --material Copper --soil "Black Sand / Magnetite Soil" --depth 0.65
```

JSON output:

```bash
pi-classifier predict --material Copper --soil "Black Sand / Magnetite Soil" --depth 0.65 --json
```

Evaluate the classifier:

```bash
pi-classifier evaluate
```

Evaluate with JSON output:

```bash
pi-classifier evaluate --json
```

---

## Python Examples

Run a single prediction:

```bash
python examples/run_single_prediction.py
```

Train and evaluate:

```bash
python examples/train_and_evaluate.py
```

Demonstrate verified waveform memory:

```bash
python examples/confirmed_memory_demo.py
```

Generate reports:

```bash
python examples/evaluation_report.py
```

Generate waveform plot:

```bash
python examples/generate_waveform_plot.py
```

Export sample synthetic dataset:

```bash
python examples/export_sample_dataset.py
```

---

## Example CLI Output

```text
Pulse Induction Metal Classifier
----------------------------------------
Validation accuracy : 92.40%

Input target
  Material          : Copper
  Soil              : Black Sand / Magnetite Soil
  Depth             : 0.65 m
  Horizontal dist.  : 0.25 m

Prediction
  Class             : Copper
  Confidence        : 71.80%
  SNR norm          : 0.842
  Tau estimate      : 0.790
  Memory match      : None (similarity=0.000, strength=0.000)

Class probabilities
  Copper        :  72.14%
  Brass         :  10.81%
  Aluminum      :   8.74%
```

---

## Repository Structure

```text
pulse-induction-metal-classifier/
│
├── src/pi_detector/
│   ├── materials.py
│   ├── soil.py
│   ├── pulse_model.py
│   ├── ground_balance.py
│   ├── feature_extraction.py
│   ├── classifier.py
│   ├── waveform_memory.py
│   ├── dataset_generator.py
│   ├── pipeline.py
│   └── cli.py
│
├── examples/
│   ├── run_single_prediction.py
│   ├── train_and_evaluate.py
│   ├── confirmed_memory_demo.py
│   ├── evaluation_report.py
│   ├── generate_waveform_plot.py
│   └── export_sample_dataset.py
│
├── docs/
│   ├── mathematical_model.md
│   ├── ai_methodology.md
│   ├── signal_processing_pipeline.md
│   ├── system_architecture.md
│   └── cli_reference.md
│
├── tests/
├── reports/
├── figures/
└── data/
```

---

## Technical Overview

The classifier uses a feature vector extracted from a compensated PI decay waveform:

```text
[
  signal amplitude,
  phase proxy,
  estimated decay tau,
  ferrous index,
  conductivity response,
  depth estimate,
  soil parameters,
  8 decay waveform samples,
  decay area,
  decay slope,
  early/late ratio,
  SNR,
  sweep quality,
  ground balance error,
  electronics noise,
  pulse parameters
]
```

The first-stage prediction is produced using a Gaussian probabilistic classifier. The prediction can then be refined with verified waveform memory, which compares the current feature vector against confirmed field samples using a weighted Gaussian kernel distance.

---

## Verified Waveform Memory

A verified sample is a feature vector with a confirmed target label. In a real workflow, this label could come from excavation, manual confirmation, or lab verification.

The memory module compares a new waveform against verified samples using weighted feature-space distance. The 8 PI waveform samples receive the highest weights, so the model primarily learns from waveform shape instead of simply counting previous labels.

---

## Limitations

- The current dataset is synthetically generated.
- The electromagnetic model is simplified and intended for algorithm prototyping.
- The framework does not claim real-world mine or hazard detection capability.
- Real hardware validation would be required before any field deployment.
- Environmental conditions are approximated using parameterized soil and noise models.
- The classifier is intentionally lightweight and interpretable, not a deep neural network.

---

## Intended Use

This project is designed as an engineering-focused prototype for:

- pulse induction signal analysis,
- lightweight Edge AI classification,
- sensor waveform feature extraction,
- adaptive calibration with verified samples,
- ground balance compensation under varying soil conditions.

---

## Technologies

- Python
- Signal Processing
- Probabilistic Machine Learning
- Edge AI
- Synthetic Sensor Data Generation
- Waveform Classification
- CLI Tooling
- Unit Testing

---

## License

MIT License.
