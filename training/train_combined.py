"""
train_combined.py — OPTIMIZED for 80%+ mAP.

Key fixes from previous run:
1. cache='disk'  — fixes RAM OOM, stable training
2. epochs=50, patience=15 — more training time
3. freeze=5 — unfreeze more layers for better learning
4. rect=False — proper mosaic augmentation
5. No close_mosaic issue

USAGE:
    python training/train_combined.py
"""

from ultralytics import YOLO
import torch
import os
import sys
import glob
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_best_yaml():
    options = [
        ("datasets/mega_dataset/data.yaml",     "MEGA dataset"),
        ("datasets/combined_dataset/data.yaml", "COMBINED dataset"),
        ("datasets/navigation_dataset/data.yaml","OUTDOOR dataset"),
    ]
    for path, label in options:
        if os.path.exists(path):
            print(f"[DATASET] Using: {label}")
            return path
    return None


def train():
    DATA_YAML = find_best_yaml()
    if DATA_YAML is None:
        print("ERROR: No dataset found!")
        return

    # ===== CONFIG =====
    BASE_MODEL  = "yolo26m.pt"
    EPOCHS      = 50         # More epochs — previous best was epoch 19 of 30
    PATIENCE    = 15         # More patience — was 7, stopped too early
    IMAGE_SIZE  = 416
    BATCH_SIZE  = 8

    print("=" * 65)
    print("  OPTIMIZED TRAINING — TARGET 80%+ mAP")
    print("=" * 65)

    # ===== DEVICE =====
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[DEVICE] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        if vram < 5.0:
            BATCH_SIZE = 8
        print(f"[FIX] cache='disk' — fixes RAM OOM from previous run")
        print(f"[FIX] patience=15 — previous run stopped too early at epoch 19")
        print(f"[FIX] freeze=5 — unfreezes more layers for better learning")
    else:
        device = 'cpu'
        BATCH_SIZE = 4

    # ===== DATASET INFO =====
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)

    nc    = data.get('nc', '?')
    names = data.get('names', {})
    base_dir   = os.path.dirname(DATA_YAML)
    train_dir  = os.path.join(base_dir, 'train', 'images')
    valid_dir  = os.path.join(base_dir, 'valid', 'images')
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    valid_count = len(os.listdir(valid_dir)) if os.path.exists(valid_dir) else 0

    print(f"\n[DATASET] Train: {train_count:,} images")
    print(f"[DATASET] Valid: {valid_count:,} images")
    print(f"[DATASET] Classes: {nc}")
    for idx in sorted(names.keys(), key=int):
        print(f"           {idx}: {names[idx]}")

    # ===== LOAD MODEL =====
    print(f"\n[MODEL] Loading {BASE_MODEL}...")
    try:
        model = YOLO(BASE_MODEL)
        print(f"[MODEL] Loaded!")
    except Exception as e:
        print(f"[MODEL] ERROR: {e}")
        return

    print(f"\n[TRAIN] Config:")
    print(f"  Epochs:      {EPOCHS} (patience={PATIENCE})")
    print(f"  Batch:       {BATCH_SIZE}")
    print(f"  Image size:  {IMAGE_SIZE}px")
    print(f"  Cache:       disk (fixes RAM issue)")
    print(f"  Freeze:      5 layers (was 10 — more layers trainable)")
    print(f"  AMP:         ON")
    print(f"\n[TRAIN] Estimated time: ~8-12 hours (disk cache, no RAM OOM)")
    print(f"[TRAIN] Starting NOW...\n")
    print("-" * 65)

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            patience=PATIENCE,
            save=True,
            save_period=5,
            project="runs/train",
            name="combined_model_v2",
            exist_ok=True,
            plots=True,

            # FIX 1: disk cache instead of RAM cache
            cache='disk',
            workers=8,
            amp=True,

            # Optimizer
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=3,
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,

            # FIX 2: freeze fewer layers — more of model trains
            freeze=5,

            # Augmentation
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            verbose=True,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[OOM] CUDA Out of Memory!")
            print(f"Fix: Change BATCH_SIZE = 8")
        else:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        return

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== DONE =====
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*65}")

    best_paths = glob.glob(
        "runs/**/combined_model_v2/weights/best.pt", recursive=True
    )

    if best_paths:
        best = best_paths[0]
        size = os.path.getsize(best) / (1024*1024)
        print(f"\n[SAVED] {best} ({size:.1f} MB)")
        print(f"\n[RUN]")
        print(f"  Web:     python run_web.py --dual --model {best}")
        print(f"  Desktop: python main.py --dual --model {best}")
        print(f"\n[EVALUATE]")
        print(f"  python evaluation/metrics.py --compare")

    return results


if __name__ == "__main__":
    train()

