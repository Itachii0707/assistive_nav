"""
metrics.py — Evaluate and compare all trained models.

USAGE:
    python evaluation/metrics.py              # evaluate best model
    python evaluation/metrics.py --compare    # compare all models
    python evaluation/metrics.py --model path/to/best.pt
"""

from ultralytics import YOLO
import torch
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_all_models():
    """Find all trained best.pt models."""
    models = {}

    searches = [
        ("runs/**/mega_model/weights/best.pt", "YOLO26m MEGA dataset"),
        ("runs/**/combined_model_max/weights/best.pt", "YOLO26m MAX training"),
        ("runs/**/combined_model_m/weights/best.pt", "YOLO26m combined"),
        ("runs/**/combined_model/weights/best.pt", "YOLO26n combined"),
        ("runs/**/navigation_model/weights/best.pt", "YOLO26n outdoor only"),
    ]

    for pattern, label in searches:
        paths = glob.glob(pattern, recursive=True)
        if paths:
            models[label] = paths[0]

    return models


def find_best_dataset():
    """Find best available dataset for evaluation."""
    for yaml_path in [
        "datasets/mega_dataset/data.yaml",
        "datasets/combined_dataset/data.yaml",
        "datasets/navigation_dataset/data.yaml",
    ]:
        if os.path.exists(yaml_path):
            return yaml_path
    return None


def evaluate_model(model_path, data_yaml=None, device=None, label=""):
    """Run YOLO validation and return metrics."""

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None

    if data_yaml is None:
        data_yaml = find_best_dataset()

    if data_yaml is None:
        print("ERROR: No dataset found!")
        return None

    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'

    name = label or os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    print(f"\n[EVAL] {name}")
    print(f"  Model:   {model_path}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Running validation...")

    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml, device=device, verbose=False)

        metrics = {
            'mAP50': round(float(results.box.map50), 4),
            'mAP50_95': round(float(results.box.map), 4),
            'precision': round(float(results.box.mp), 4),
            'recall': round(float(results.box.mr), 4),
        }

        print(f"\n  {'=' * 46}")
        print(f"  RESULTS: {name}")
        print(f"  {'=' * 46}")
        print(f"  mAP@50:      {metrics['mAP50']*100:6.2f}%")
        print(f"  mAP@50-95:   {metrics['mAP50_95']*100:6.2f}%")
        print(f"  Precision:   {metrics['precision']*100:6.2f}%")
        print(f"  Recall:      {metrics['recall']*100:6.2f}%")
        print(f"  {'=' * 46}")

        # Rating
        map50 = metrics['mAP50'] * 100
        if map50 >= 80:
            print(f"  EXCELLENT — {map50:.1f}% mAP@50")
        elif map50 >= 70:
            print(f"  VERY GOOD — {map50:.1f}% mAP@50")
        elif map50 >= 60:
            print(f"  GOOD — {map50:.1f}% mAP@50")
        elif map50 >= 50:
            print(f"  ACCEPTABLE — {map50:.1f}% mAP@50")
        else:
            print(f"  NEEDS MORE DATA/TRAINING — {map50:.1f}% mAP@50")

        return metrics

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def compare_all():
    """Compare all trained models side by side."""
    device = '0' if torch.cuda.is_available() else 'cpu'
    data_yaml = find_best_dataset()

    if data_yaml is None:
        print("ERROR: No dataset found!")
        return

    all_models = find_all_models()

    if not all_models:
        print("ERROR: No trained models found!")
        print("Train first: python training/train_mega.py")
        return

    print(f"\n[INFO] Found {len(all_models)} trained models")
    print(f"[INFO] Dataset: {data_yaml}")

    results = {}
    for label, path in all_models.items():
        r = evaluate_model(path, data_yaml, device, label)
        if r:
            results[label] = r

    if len(results) >= 2:
        print(f"\n{'=' * 72}")
        print(f"  FULL COMPARISON TABLE")
        print(f"{'=' * 72}")
        print(f"  {'Model':<35} {'mAP@50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8}")
        print(f"  {'-' * 69}")

        # Sort by mAP50
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mAP50'], reverse=True)
        for name, m in sorted_results:
            print(f"  {name:<35} {m['mAP50']*100:>7.1f}% {m['mAP50_95']*100:>9.1f}% "
                  f"{m['precision']*100:>7.1f}% {m['recall']*100:>7.1f}%")

        print(f"  {'=' * 69}")

        # Best model
        best_name, best_metrics = sorted_results[0]
        print(f"\n  BEST MODEL: {best_name}")
        print(f"  mAP@50: {best_metrics['mAP50']*100:.2f}%")

        # Show improvement from worst to best
        if len(sorted_results) >= 2:
            worst_name, worst_metrics = sorted_results[-1]
            improvement = (best_metrics['mAP50'] - worst_metrics['mAP50']) * 100
            print(f"  Improvement over baseline: +{improvement:.1f}% mAP@50")

        print(f"{'=' * 72}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true',
                        help='Compare all trained models')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model path to evaluate')
    args = parser.parse_args()

    if args.compare:
        compare_all()
    elif args.model:
        evaluate_model(args.model)
    else:
        # Auto find and evaluate best model
        models = find_all_models()
        if models:
            best_label = list(models.keys())[0]
            best_path = models[best_label]
            print(f"[AUTO] Evaluating best available: {best_label}")
            evaluate_model(best_path, label=best_label)
        else:
            print("No trained models found!")
            print("Train first: python training/train_mega.py")


            