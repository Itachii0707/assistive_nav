"""
merge_all_datasets.py — Merge ALL datasets into mega_dataset.

Handles:
  - navigation_dataset (outdoor, 9 classes)
  - pascal_yolo (converted VOC, 11 classes)
  - homeobjects (indoor, 12 classes)
  - indoor_kaggle
  - pothole datasets
  - vehicle datasets
  - pedestrian datasets

USAGE:
    python merge_all_datasets.py
"""

import os
import shutil
import yaml
import glob
from collections import Counter


# ============================================================
# MASTER CLASSES — Final unified class list
# ============================================================
MASTER_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    4:  "bus",
    5:  "truck",
    6:  "traffic light",
    7:  "stop sign",
    8:  "bench",
    9:  "cat",
    10: "dog",
    11: "backpack",
    12: "umbrella",
    13: "bottle",
    14: "chair",
    15: "couch",
    16: "dining table",
    17: "laptop",
    18: "cell phone",
    19: "pothole",
    20: "electric pole",
    21: "uncovered manhole",
    22: "traffic signs",
    23: "door",
    24: "wall",
    25: "stairs",
    26: "elevator",
    27: "cabinet",
    28: "tree",
    29: "bicycle lane",
}

# ============================================================
# CLASS NAME → MASTER ID MAPPING
# Every possible name variant mapped here
# ============================================================
NAME_TO_ID = {
    # person
    "person": 0, "pedestrian": 0, "human": 0, "people": 0,
    "man": 0, "woman": 0, "child": 0, "rider": 0, "adult": 0,

    # bicycle
    "bicycle": 1, "bike": 1, "cycle": 1,

    # car
    "car": 2, "automobile": 2, "sedan": 2, "suv": 2,
    "vehicle": 2, "cars": 2,

    # motorcycle
    "motorcycle": 3, "motorbike": 3, "motor": 3, "scooter": 3,

    # bus
    "bus": 4,

    # truck
    "truck": 5, "van": 5, "lorry": 5, "pickup truck": 5,
    "pickup": 5,

    # traffic light
    "traffic light": 6, "trafficlight": 6, "traffic_light": 6,
    "traffic light(green)": 6, "traffic light(red)": 6,

    # stop sign
    "stop sign": 7, "stopsign": 7,

    # bench
    "bench": 8,

    # cat
    "cat": 9,

    # dog
    "dog": 10,

    # backpack
    "backpack": 11, "bag": 11, "handbag": 11, "suitcase": 11,
    "luggage": 11,

    # umbrella
    "umbrella": 12,

    # bottle
    "bottle": 13, "cup": 13,

    # chair
    "chair": 14, "seat": 14, "monoblock chair": 14,
    "office_chair": 14, "armchair": 14,

    # couch
    "couch": 15, "sofa": 15, "bed": 15,
    "sofa1": 15, "sofa2": 15, "sofa3": 15,

    # dining table
    "dining table": 16, "table": 16, "diningtable": 16,
    "dining_table": 16, "coffee_table": 16, "desk": 16,

    # laptop
    "laptop": 17, "computer": 17,

    # cell phone
    "cell phone": 18, "phone": 18, "mobile": 18, "cellphone": 18,

    # pothole
    "pothole": 19, "road damage": 19, "crack": 19,
    "potholes": 19, "road_damage": 19,

    # electric pole
    "electric pole": 20, "pole": 20, "utility pole": 20,
    "electric_pole": 20,

    # uncovered manhole
    "uncovered manhole": 21, "manhole": 21,

    # traffic signs
    "traffic signs": 22, "traffic sign": 22, "road sign": 22,
    "sign": 22, "traffic_sign": 22,

    # door
    "door": 23, "doors": 23, "open door": 23,
    "closed door": 23, "cabinetdoor": 23,

    # wall
    "wall": 24, "window": 24,

    # stairs
    "stairs": 25, "stair": 25, "staircase": 25,
    "stairway": 25, "steps": 25,

    # elevator
    "elevator": 26, "lift": 26, "escalator": 26,

    # cabinet
    "cabinet": 27, "wardrobe": 27, "cupboard": 27,
    "shelf": 27, "bookcase": 27,

    # tree
    "tree": 28, "bush": 28, "plant": 28,
    "potted plant": 28, "pottedplant": 28,

    # SKIP these completely (return None)
    "aeroplane": None, "airplane": None, "boat": None,
    "train": None, "horse": None, "cow": None, "sheep": None,
    "bird": None, "tvmonitor": None, "tv": None, "lamp": None,
    "photo frame": None, "refrigerator": None, "microwave": None,
    "oven": None, "toaster": None, "sink": None, "toilet": None,
    "book": None, "clock": None, "vase": None, "scissors": None,
    "teddy bear": None, "hair drier": None, "toothbrush": None,
    "sports ball": None, "kite": None, "baseball bat": None,
    "skateboard": None, "surfboard": None, "tennis racket": None,
    "wine glass": None, "fork": None, "knife": None, "spoon": None,
    "bowl": None, "banana": None, "apple": None, "sandwich": None,
    "orange": None, "broccoli": None, "carrot": None, "hot dog": None,
    "pizza": None, "donut": None, "cake": None, "frisbee": None,
    "skis": None, "snowboard": None, "baseball glove": None,
    "tie": None, "fire hydrant": None,
}


def get_class_id(name):
    """Map class name to master class ID. Returns None to skip."""
    name_clean = name.lower().strip()
    # Direct match
    if name_clean in NAME_TO_ID:
        return NAME_TO_ID[name_clean]
    # Partial match
    for key, val in NAME_TO_ID.items():
        if key and (key in name_clean or name_clean in key):
            return val
    return None


def find_yaml(dataset_dir):
    """Find data.yaml in dataset directory."""
    for name in ['data.yaml', 'dataset.yaml']:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            return p
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f == 'data.yaml':
                return os.path.join(root, f)
    return None


def read_classes(yaml_path):
    """Read class names from data.yaml."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def find_splits(dataset_dir):
    """Find train/valid/test image folders."""
    splits = {}
    for name in ['train', 'valid', 'val', 'test']:
        img_dir = os.path.join(dataset_dir, name, 'images')
        lbl_dir = os.path.join(dataset_dir, name, 'labels')
        if os.path.exists(img_dir):
            splits[name] = {'images': img_dir, 'labels': lbl_dir}
    return splits


def copy_and_remap(src_img, src_lbl, dst_img, dst_lbl,
                   class_remap, prefix, stats):
    """Copy images and labels with class remapping."""
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    if not os.path.exists(src_img):
        return 0

    copied = 0
    for img_file in os.listdir(src_img):
        if not img_file.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            continue

        new_img = f"{prefix}_{img_file}"
        shutil.copy2(
            os.path.join(src_img, img_file),
            os.path.join(dst_img, new_img)
        )

        base = os.path.splitext(img_file)[0]
        src_label = os.path.join(src_lbl, base + '.txt')

        if os.path.exists(src_label):
            new_lines = []
            with open(src_label, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_id = int(parts[0])
                    new_id = class_remap.get(old_id)
                    if new_id is None:
                        continue
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts))
                    stats['class_counts'][new_id] = \
                        stats['class_counts'].get(new_id, 0) + 1

            if new_lines:
                dst_label_path = os.path.join(
                    dst_lbl, f"{prefix}_{base}.txt"
                )
                with open(dst_label_path, 'w') as f:
                    f.write("\n".join(new_lines) + "\n")

        copied += 1
    return copied


def build_remap(yaml_path, dataset_name):
    """Build class ID remap from dataset classes to master classes."""
    classes = read_classes(yaml_path)
    remap = {}
    unmapped = []

    for old_id, name in classes.items():
        new_id = get_class_id(str(name))
        if new_id is not None:
            remap[old_id] = new_id
        else:
            unmapped.append(f"{old_id}:{name}")

    mapped_count = sum(1 for v in remap.values() if v is not None)
    print(f"  {dataset_name}: {len(classes)} classes → "
          f"{mapped_count} mapped, {len(unmapped)} skipped")
    if unmapped and len(unmapped) <= 5:
        print(f"    Skipped: {unmapped}")

    return remap


def merge():
    print("=" * 65)
    print("  MEGA DATASET MERGER — ALL DATASETS")
    print("=" * 65)

    OUTPUT_DIR = "datasets/mega_dataset"

    # ===== ALL DATASET DIRECTORIES =====
    ALL_DATASETS = [
        ("datasets/navigation_dataset", "nav"),
        ("datasets/extra/pascal_yolo", "voc"),
        ("datasets/extra/homeobjects", "home"),
        ("datasets/extra/homeobjects-3K", "home2"),
        ("datasets/extra/indoor_kaggle", "indkag"),
        ("datasets/extra/visdrone_yolo", "vis"),
        ("datasets/extra/vehicle_ped_yolo", "vehped"),
        ("datasets/extra/vehicle_yolo_fixed", "vehyolo"),
        ("datasets/extra/pedestrian_yolo", "pedk"),
        ("datasets/extra/pothole_yolo1", "pot1"),
        ("datasets/extra/pothole_yolo_fixed", "pot_yf"),
        ("datasets/extra/road_yolo3", "road3"),
    ]


    # Skip indoor_dataset — class names are meaningless numbers
    # Use homeobjects instead for indoor

    # Filter to existing ones with valid yaml
    existing = []
    print("\n  Scanning datasets:")
    for d, prefix in ALL_DATASETS:
        if os.path.exists(d):
            yaml_path = find_yaml(d)
            if yaml_path:
                # Count images
                img_count = 0
                for root, dirs, files in os.walk(d):
                    img_count += len([f for f in files
                                      if f.lower().endswith(
                                          ('.jpg','.jpeg','.png'))])
                if img_count > 0:
                    print(f"  FOUND: {d} ({img_count:,} images)")
                    existing.append((d, prefix, yaml_path))
                else:
                    print(f"  EMPTY: {d}")
            else:
                print(f"  NO YAML: {d}")
        else:
            print(f"  MISSING: {d}")

    if not existing:
        print("\nERROR: No datasets found!")
        return

    print(f"\n  Merging {len(existing)} datasets...")

    # ===== BUILD REMAPS =====
    print(f"\n[1/3] Building class remaps...")
    dataset_remaps = []
    for d, prefix, yaml_path in existing:
        remap = build_remap(yaml_path, os.path.basename(d))
        dataset_remaps.append((d, prefix, remap))

    # ===== COPY FILES =====
    print(f"\n[2/3] Copying and remapping files...")

    if os.path.exists(OUTPUT_DIR):
        print(f"  Removing old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    stats = {'class_counts': {}}
    total_copied = 0

    for d, prefix, remap in dataset_remaps:
        splits = find_splits(d)
        ds_total = 0

        for split_name, paths in splits.items():
            dst_split = 'valid' if split_name in ['val','valid'] else split_name
            count = copy_and_remap(
                paths['images'], paths['labels'],
                os.path.join(OUTPUT_DIR, dst_split, 'images'),
                os.path.join(OUTPUT_DIR, dst_split, 'labels'),
                remap, prefix, stats
            )
            ds_total += count

        print(f"  {os.path.basename(d)}: {ds_total:,} images")
        total_copied += ds_total

    # ===== CREATE data.yaml =====
    print(f"\n[3/3] Creating mega data.yaml...")

    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(MASTER_CLASSES),
        'names': {int(k): v for k, v in MASTER_CLASSES.items()},
    }

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # ===== SUMMARY =====
    print(f"\n{'=' * 65}")
    print(f"  MERGE COMPLETE!")
    print(f"{'=' * 65}")

    for split in ['train', 'valid']:
        img_dir = os.path.join(OUTPUT_DIR, split, 'images')
        if os.path.exists(img_dir):
            n = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
            print(f"  {split}: {n:,} images")

    print(f"  Total:   {total_copied:,} images")
    print(f"  Classes: {len(MASTER_CLASSES)}")

    print(f"\n  Top 10 classes by annotation count:")
    sorted_counts = sorted(stats['class_counts'].items(),
                          key=lambda x: x[1], reverse=True)
    for cls_id, count in sorted_counts[:10]:
        cls_name = MASTER_CLASSES.get(cls_id, f"class_{cls_id}")
        bar = "#" * min(count // 200, 25)
        print(f"    {cls_id:2d} {cls_name:<22} {count:6,} {bar}")

    if total_copied >= 25000:
        print(f"\n  Expected mAP@50: 82-88%")
    elif total_copied >= 15000:
        print(f"\n  Expected mAP@50: 78-84%")
    elif total_copied >= 10000:
        print(f"\n  Expected mAP@50: 74-80%")

    print(f"\n[NEXT] Run: python training/train_combined.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    merge()
