from __future__ import annotations

import argparse
import json

from .materials import MATERIALS
from .soil import SOILS
from .pipeline import PulseInductionPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-classifier",
        description=(
            "Pulse induction metal detector waveform classifier with "
            "ground balance compensation and verified waveform memory."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="Run a single synthetic PI waveform prediction.",
    )
    predict.add_argument(
        "--material",
        default="Copper",
        choices=sorted(MATERIALS.keys()),
        help="Target material to simulate.",
    )
    predict.add_argument(
        "--soil",
        default="Black Sand / Magnetite Soil",
        choices=sorted(SOILS.keys()),
        help="Soil condition to simulate.",
    )
    predict.add_argument(
        "--depth",
        type=float,
        default=0.65,
        help="Burial depth in meters.",
    )
    predict.add_argument(
        "--distance",
        type=float,
        default=0.25,
        help="Horizontal target-to-coil distance in meters.",
    )
    predict.add_argument(
        "--samples-per-material",
        type=int,
        default=450,
        help="Synthetic training samples per metal class.",
    )
    predict.add_argument(
        "--noise-samples",
        type=int,
        default=450,
        help="Synthetic training samples per noise/unknown class.",
    )
    predict.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Train and evaluate the classifier on synthetic validation data.",
    )
    evaluate.add_argument(
        "--samples-per-material",
        type=int,
        default=700,
        help="Synthetic training samples per metal class.",
    )
    evaluate.add_argument(
        "--noise-samples",
        type=int,
        default=700,
        help="Synthetic training samples per noise/unknown class.",
    )
    evaluate.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    return parser


def _prediction_to_dict(material: str, soil: str, depth: float, distance: float, result) -> dict:
    return {
        "input": {
            "material": material,
            "soil": soil,
            "depth_m": depth,
            "horizontal_distance_m": distance,
        },
        "prediction": {
            "class": result.predicted_class,
            "confidence": result.confidence,
            "snr_norm": result.snr_norm,
            "tau_estimate": result.tau_estimate,
        },
        "memory": {
            "best_label": result.memory_match.best_label,
            "best_similarity": result.memory_match.best_similarity,
            "strength": result.memory_match.strength,
        },
        "probabilities": result.probabilities,
    }


def _run_predict(args: argparse.Namespace) -> int:
    pipeline = PulseInductionPipeline()
    accuracy = pipeline.train(
        samples_per_material=args.samples_per_material,
        noise_samples=args.noise_samples,
    )

    result = pipeline.simulate_and_predict(
        material_name=args.material,
        soil_name=args.soil,
        depth_m=args.depth,
        horizontal_distance_m=args.distance,
    )

    payload = _prediction_to_dict(
        material=args.material,
        soil=args.soil,
        depth=args.depth,
        distance=args.distance,
        result=result,
    )
    payload["training"] = {
        "validation_accuracy": accuracy,
        "samples_per_material": args.samples_per_material,
        "noise_samples": args.noise_samples,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("Pulse Induction Metal Classifier")
    print("-" * 40)
    print(f"Validation accuracy : {accuracy * 100:.2f}%")
    print()
    print("Input target")
    print(f"  Material          : {args.material}")
    print(f"  Soil              : {args.soil}")
    print(f"  Depth             : {args.depth:.2f} m")
    print(f"  Horizontal dist.  : {args.distance:.2f} m")
    print()
    print("Prediction")
    print(f"  Class             : {result.predicted_class}")
    print(f"  Confidence        : {result.confidence * 100:.2f}%")
    print(f"  SNR norm          : {result.snr_norm:.3f}")
    print(f"  Tau estimate      : {result.tau_estimate:.3f}")
    print(
        f"  Memory match      : {result.memory_match.best_label} "
        f"(similarity={result.memory_match.best_similarity:.3f}, "
        f"strength={result.memory_match.strength:.3f})"
    )
    print()
    print("Class probabilities")
    for name, prob in sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<14}: {prob * 100:6.2f}%")

    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    pipeline = PulseInductionPipeline()
    accuracy = pipeline.train(
        samples_per_material=args.samples_per_material,
        noise_samples=args.noise_samples,
    )

    payload = {
        "validation_accuracy": accuracy,
        "samples_per_material": args.samples_per_material,
        "noise_samples": args.noise_samples,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("Evaluation completed")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return _run_predict(args)

    if args.command == "evaluate":
        return _run_evaluate(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
