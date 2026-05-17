"""
export_models.py — Export trained YOLO26 to ONNX and TFLite.
Auto-finds the trained model wherever it was saved.
"""

from ultralytics import YOLO
import shutil
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_trained_model():
    """Search for best.pt in all possible locations."""
    possible_paths = [
        "runs/train/navigation_model/weights/best.pt",
        "runs/detect/runs/train/navigation_model/weights/best.pt",
        "runs/detect/navigation_model/weights/best.pt",
        "runs/segment/navigation_model/weights/best.pt",
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p

    # Search everywhere in runs/
    found = glob.glob("runs/**/best.pt", recursive=True)
    if found:
        return found[0]

    return None


def export_all():
    """Export trained model to all formats."""

    model_path = find_trained_model()

    if model_path is None:
        print("ERROR: No trained model found!")
        print("")
        print("Searched in:")
        print("  runs/train/navigation_model/weights/best.pt")
        print("  runs/detect/runs/train/navigation_model/weights/best.pt")
        print("  runs/**/best.pt")
        print("")
        print("Train first: python training/train_custom.py")
        return

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  MODEL EXPORT")
    print("=" * 60)

    # Load model
    print(f"\n[LOAD] Found model: {model_path}")
    model = YOLO(model_path)

    num_classes = len(model.names)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[LOAD] Classes: {num_classes}")
    for idx, name in model.names.items():
        print(f"        {idx}: {name}")
    print(f"[LOAD] Size: {model_size:.1f} MB")

    # 1. Copy .pt
    print(f"\n{'─' * 40}")
    print(f"[1/3] Copying PyTorch model (.pt)")
    print(f"{'─' * 40}")

    dest_pt = os.path.join(output_dir, "navigation_model.pt")
    shutil.copy2(model_path, dest_pt)
    pt_size = os.path.getsize(dest_pt) / (1024 * 1024)
    print(f"  SAVED: {dest_pt} ({pt_size:.1f} MB)")
    print(f"  USE:   python main.py --model models/navigation_model.pt")

    # 2. Export ONNX
    print(f"\n{'─' * 40}")
    print(f"[2/3] Exporting ONNX model (for fast CPU)")
    print(f"{'─' * 40}")

    try:
        onnx_path = model.export(format="onnx")
        if onnx_path and os.path.exists(onnx_path):
            dest_onnx = os.path.join(output_dir, "navigation_model.onnx")
            shutil.copy2(onnx_path, dest_onnx)
            onnx_size = os.path.getsize(dest_onnx) / (1024 * 1024)
            print(f"  SAVED: {dest_onnx} ({onnx_size:.1f} MB)")
            print(f"  USE:   python main.py --model models/navigation_model.onnx")
        else:
            print(f"  WARNING: ONNX export did not produce file")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  FIX:   pip install onnx>=1.14.0")

    # 3. Export TFLite
    print(f"\n{'─' * 40}")
    print(f"[3/3] Exporting TFLite INT8 model (for Android)")
    print(f"{'─' * 40}")

    try:
        tflite_path = model.export(format="tflite", int8=True)
        if tflite_path and os.path.exists(tflite_path):
            dest_tflite = os.path.join(output_dir, "navigation_model_int8.tflite")
            shutil.copy2(tflite_path, dest_tflite)
            tflite_size = os.path.getsize(dest_tflite) / (1024 * 1024)
            print(f"  SAVED: {dest_tflite} ({tflite_size:.1f} MB)")
        else:
            print(f"  WARNING: TFLite export did not produce file")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  FIX:   pip install tensorflow>=2.15.0")
        print(f"  NOTE:  TFLite only needed for Android. Skip if laptop-only.")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EXPORT SUMMARY")
    print(f"{'=' * 60}")

    if os.path.exists(output_dir):
        total_size = 0
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / (1024 * 1024)
                total_size += size
                print(f"  {f:<40} {size:>6.1f} MB")
        print(f"  {'─' * 46}")
        print(f"  {'TOTAL':<40} {total_size:>6.1f} MB")

    print(f"\n  Laptop GPU   → models/navigation_model.pt")
    print(f"  Laptop CPU   → models/navigation_model.onnx")
    print(f"  Android      → models/navigation_model_int8.tflite")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    export_all()
