import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_detector.dataset_generator import generate_dataset
from pi_detector.feature_extraction import FEATURE_NAMES
from pi_detector.pulse_model import PulseConfig


def main():
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(exist_ok=True)

    samples, labels = generate_dataset(
        config=PulseConfig(),
        samples_per_material=25,
        noise_samples=25,
        seed=2026,
    )

    output_path = output_dir / "sample_waveforms.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *FEATURE_NAMES])

        for x, y in zip(samples, labels):
            writer.writerow([y, *[f"{value:.6f}" for value in x]])

    print(f"Saved sample dataset: {output_path}")


if __name__ == "__main__":
    main()
