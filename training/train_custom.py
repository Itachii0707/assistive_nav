"""
train_custom.py — Fine-tune YOLO26 on Roboflow obstacle detection dataset.

WHAT THIS DOES:
1. Loads COCO pre-trained yolo26n.pt (80 classes, auto-downloads ~5MB)
2. Fine-tunes it on your 9-class navigation dataset
3. Saves best model to runs/train/navigation_model/weights/best.pt

HOW TO RUN:
    python training/train_custom.py

TIME:
    GPU: 15-30 minutes
    CPU: 1-3 hours
"""

from ultralytics import YOLO
import torch
import os
import sys

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train():
    """Fine-tune YOLO26 on navigation dataset."""

    # ===== CONFIGURATION =====
    DATA_YAML = "datasets/navigation_dataset/data.yaml"
    BASE_MODEL = "yolo26n.pt"
    EPOCHS = 50
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    PATIENCE = 10

    # ===== CHECK DATASET =====
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found!")
        print("")
        print("SOLUTION:")
        print("1. Run: python download_dataset.py")
        print("2. Make sure datasets/navigation_dataset/ folder exists")
        print("3. Make sure data.yaml is inside that folder")
        return None

    # Check train images exist
    train_dir = "datasets/navigation_dataset/train/images"
    if not os.path.exists(train_dir):
        print(f"ERROR: {train_dir} not found!")
        print("Dataset folder structure is wrong.")
        return None

    train_count = len(os.listdir(train_dir))
    if train_count == 0:
        print(f"ERROR: {train_dir} is empty!")
        return None

    # ===== PRINT HEADER =====
    print("=" * 60)
    print("  YOLO26 TRAINING — Navigation Obstacle Detection")
    print("=" * 60)

    # ===== DEVICE DETECTION =====
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[DEVICE] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print("[DEVICE] Using CUDA GPU — training will be FAST")
    else:
        device = 'cpu'
        print("\n[DEVICE] No GPU detected — using CPU")
        print("[DEVICE] Training will be SLOW (1-3 hours)")
        BATCH_SIZE = 8
        print(f"[DEVICE] Reduced batch size to {BATCH_SIZE} for CPU")

    # ===== DATASET INFO =====
    print(f"\n[DATASET] Config: {DATA_YAML}")
    print(f"[DATASET] Train images: {train_count}")

    valid_dir = "datasets/navigation_dataset/valid/images"
    if os.path.exists(valid_dir):
        val_count = len(os.listdir(valid_dir))
        print(f"[DATASET] Val images: {val_count}")

    # ===== LOAD MODEL =====
    print(f"\n[MODEL] Loading base model: {BASE_MODEL}")
    print("[MODEL] This is COCO pre-trained (80 classes)")
    print("[MODEL] Fine-tuning to learn 9 navigation classes")
    print("[MODEL] First run auto-downloads from Ultralytics (~5 MB)")

    try:
        model = YOLO(BASE_MODEL)
        print("[MODEL] Loaded successfully!")
    except Exception as e:
        print(f"[MODEL] ERROR loading model: {e}")
        print("[MODEL] Check internet connection for first download")
        return None

    # ===== START TRAINING =====
    print(f"\n[TRAIN] Configuration:")
    print(f"        Epochs:     {EPOCHS}")
    print(f"        Batch size: {BATCH_SIZE}")
    print(f"        Image size: {IMAGE_SIZE}")
    print(f"        Patience:   {PATIENCE} epochs (early stopping)")
    print(f"        Device:     {'GPU (CUDA)' if device == 0 else 'CPU'}")
    print(f"\n[TRAIN] Starting training NOW...")
    print(f"[TRAIN] This will take {'15-30 minutes' if device == 0 else '1-3 hours'}")
    print("-" * 60)

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            patience=PATIENCE,
            save=True,
            project="runs/train",
            name="navigation_model",
            exist_ok=True,
            plots=True,
            cache=True,
            workers=4 if device == 0 else 2,
            verbose=True,
        )
    except Exception as e:
        print(f"\n[TRAIN] ERROR during training: {e}")
        print("\nPossible fixes:")
        print("1. Reduce batch size: change BATCH_SIZE = 4")
        print("2. Reduce image size: change IMAGE_SIZE = 320")
        print("3. Check if data.yaml paths are correct")
        print("4. Check if images and labels match")
        return None

    # ===== TRAINING COMPLETE =====
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    best_path = "runs/train/navigation_model/weights/best.pt"
    last_path = "runs/train/navigation_model/weights/last.pt"

    if os.path.exists(best_path):
        best_size = os.path.getsize(best_path) / (1024 * 1024)
        print(f"\n[SAVED] Best model: {best_path} ({best_size:.1f} MB)")
    if os.path.exists(last_path):
        last_size = os.path.getsize(last_path) / (1024 * 1024)
        print(f"[SAVED] Last model: {last_path} ({last_size:.1f} MB)")

    print(f"\n[SAVED] Training plots: runs/train/navigation_model/")
    print(f"        - confusion_matrix.png")
    print(f"        - results.png")
    print(f"        - P_curve.png")
    print(f"        - R_curve.png")
    print(f"        - F1_curve.png")

    print(f"\n[NEXT STEPS]")
    print(f"  1. Evaluate: python evaluation/metrics.py")
    print(f"  2. Export:   python training/export_models.py")
    print(f"  3. Run app:  python main.py --model {best_path}")

    return results


if __name__ == "__main__":
    train()
