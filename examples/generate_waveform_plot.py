import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from pi_detector.materials import MATERIALS
from pi_detector.soil import SOILS
from pi_detector.pulse_model import PulseConfig, simulate_target_response


def main():
    output_dir = Path(__file__).resolve().parents[1] / "figures"
    output_dir.mkdir(exist_ok=True)

    config = PulseConfig()
    soil = SOILS["Normal Soil"]
    depth_m = 0.55
    distance_m = 0.15

    times = list(config.sample_times_us)

    plt.figure(figsize=(9, 5))

    for name in ["Iron", "Aluminum", "Copper", "Brass", "Steel"]:
        waveform, _ = simulate_target_response(
            material=MATERIALS[name],
            soil=soil,
            config=config,
            depth_m=depth_m,
            horizontal_distance_m=distance_m,
        )
        plt.plot(times, waveform, marker="o", label=name)

    plt.title("Pulse Induction Decay Waveforms by Material")
    plt.xlabel("Sample time after transmit pulse cut-off (µs)")
    plt.ylabel("Normalized target response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "pi_decay_waveforms.png"
    plt.savefig(output_path, dpi=160)

    print(f"Saved waveform plot: {output_path}")


if __name__ == "__main__":
    main()
