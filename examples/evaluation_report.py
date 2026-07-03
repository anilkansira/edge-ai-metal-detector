import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_detector.classifier import GaussianProbabilisticClassifier
from pi_detector.dataset_generator import CLASS_NAMES, generate_dataset, train_validation_split
from pi_detector.pulse_model import PulseConfig


def compute_confusion_matrix(y_true, y_pred, labels):
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}

    for actual, pred in zip(y_true, y_pred):
        matrix[actual][pred] += 1

    return matrix


def precision_recall_f1(matrix, labels):
    rows = []

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(matrix[label].values())

        rows.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    return rows


def save_confusion_matrix_csv(matrix, labels, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels])

        for actual in labels:
            writer.writerow([actual, *[matrix[actual][pred] for pred in labels]])


def save_metrics_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path(__file__).resolve().parents[1] / "reports"
    output_dir.mkdir(exist_ok=True)

    config = PulseConfig()
    samples, labels = generate_dataset(
        config=config,
        samples_per_material=600,
        noise_samples=600,
        seed=123,
    )

    train_x, train_y, val_x, val_y = train_validation_split(samples, labels, validation_ratio=0.2, seed=99)

    classifier = GaussianProbabilisticClassifier(CLASS_NAMES)
    classifier.fit(train_x, train_y)

    predictions = [classifier.predict(x) for x in val_x]
    accuracy = sum(int(a == b) for a, b in zip(val_y, predictions)) / len(val_y)

    matrix = compute_confusion_matrix(val_y, predictions, CLASS_NAMES)
    rows = precision_recall_f1(matrix, CLASS_NAMES)

    save_confusion_matrix_csv(matrix, CLASS_NAMES, output_dir / "confusion_matrix.csv")
    save_metrics_csv(rows, output_dir / "classification_metrics.csv")

    print("Evaluation report")
    print(f"Validation samples: {len(val_y)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print()
    print(f"Saved: {output_dir / 'confusion_matrix.csv'}")
    print(f"Saved: {output_dir / 'classification_metrics.csv'}")

    print("\nPer-class metrics:")
    for row in rows:
        print(
            f"{row['class']:<14} "
            f"P={row['precision']:.3f} "
            f"R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} "
            f"N={row['support']}"
        )


if __name__ == "__main__":
    main()
