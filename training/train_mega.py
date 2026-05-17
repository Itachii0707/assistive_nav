"""
train_mega.py — Train YOLO26m on mega dataset for maximum accuracy.
Uses all available datasets merged together.

USAGE:
    python training/train_mega.py

EXPECTED:
    mAP@50: 75-85% (depends on dataset quality and size)
    Time: 4-8 hours on RTX 4050
"""

from ultralytics import YOLO
import torch
import os
import sys
import glob
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_dataset_info(data_yaml):
    """Get dataset statistics."""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data


def train():
    # Try mega dataset first, fall back to combined
    DATA_YAML = None
    for yaml_path in [
        "datasets/mega_dataset/data.yaml",
        "datasets/combined_dataset/data.yaml",
    ]:
        if os.path.exists(yaml_path):
            DATA_YAML = yaml_path
            break

    if DATA_YAML is None:
        print("ERROR: No dataset found!")
        print("Run: python merge_all_datasets.py")
        return

    BASE_MODEL = "yolo26m.pt"
    EPOCHS = 100
    PATIENCE = 20
    IMAGE_SIZE = 960

    print("=" * 60)
    print("  MAXIMUM ACCURACY TRAINING — YOLO26m")
    print("=" * 60)
    print(f"\n[DATASET] Using: {DATA_YAML}")

    # Dataset info
    data_info = get_dataset_info(DATA_YAML)
    nc = data_info.get('nc', '?')
    names = data_info.get('names', {})
    print(f"[DATASET] Classes: {nc}")
    for idx in sorted(names.keys(), key=int):
        print(f"           {idx}: {names[idx]}")

    # Count images
    base_dir = os.path.dirname(DATA_YAML)
    for split in ['train', 'valid']:
        img_dir = os.path.join(base_dir, split, 'images')
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            print(f"[DATASET] {split}: {count} images")

    # Device
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[DEVICE] GPU: {gpu_name} ({vram:.1f} GB)")

        # Auto batch size based on VRAM
        if vram >= 8.0:
            BATCH_SIZE = 16
        elif vram >= 6.0:
            BATCH_SIZE = 12
        else:
            BATCH_SIZE = 8
            IMAGE_SIZE = 640

        print(f"[DEVICE] Batch: {BATCH_SIZE}, ImgSize: {IMAGE_SIZE}")
    else:
        device = 'cpu'
        BATCH_SIZE = 4
        IMAGE_SIZE = 640
        print("\n[DEVICE] CPU — very slow")

    # Load model
    print(f"\n[MODEL] Loading {BASE_MODEL}...")
    try:
        model = YOLO(BASE_MODEL)
        print(f"[MODEL] Loaded yolo26m (20.1M parameters)")
    except Exception as e:
        print(f"[MODEL] ERROR: {e}")
        return

    print(f"\n[TRAIN] Starting maximum accuracy training...")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"  Expected time: 4-8 hours")
    print(f"  Expected mAP@50: 75-85%")
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
            save_period=10,            # Save checkpoint every 10 epochs
            project="runs/train",
            name="mega_model",
            exist_ok=True,
            plots=True,

            # Speed
            cache=True,
            workers=8,
            amp=True,

            # Optimizer
            optimizer='AdamW',
            cos_lr=True,
            warmup_epochs=5,
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            momentum=0.937,

            # Transfer learning
            freeze=10,

            # Maximum augmentation
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.05,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            auto_augment='randaugment',

            verbose=True,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[OOM] CUDA Out of Memory!")
            print(f"Fix: Change BATCH_SIZE = 8, IMAGE_SIZE = 640")
        else:
            print(f"\n[ERROR] {e}")
        return
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Done
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    best_paths = glob.glob("runs/**/mega_model/weights/best.pt", recursive=True)
    if best_paths:
        best = best_paths[0]
        size = os.path.getsize(best) / (1024 * 1024)
        print(f"\n  Best model: {best} ({size:.1f} MB)")
        print(f"\n  EVALUATE:")
        print(f"    python evaluation/metrics.py --compare")
        print(f"\n  RUN:")
        print(f"    python main.py --dual --model {best}")
        print(f"    python run_web.py --dual --model {best}")

    return results


if __name__ == "__main__":
    train()


    