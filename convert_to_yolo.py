"""
convert_to_yolo.py — Fixed for all dataset structures.

Fixes:
1. pothole_andrew — images/ + annotations/ (Pascal VOC XML)
2. road_damage3   — already YOLO but WRONG class IDs (1,2,3 → pothole 19)
3. visdrone       — VisDrone images+labels, remap class IDs
4. vehicle_ped    — folder-per-class, no labels, assign from folder name
5. vehicle_yolo   — already YOLO train/valid/test structure
6. pedestrian_k   — Pascal VOC XML in Train/Test/Val structure
7. pothole_yolo   — already YOLO with train/valid structure
8. pascal_voc     — VOCdevkit structure (already done, skip)

USAGE:
    python convert_to_yolo.py
"""

import os
import shutil
import glob
import random
import xml.etree.ElementTree as ET
import yaml

# ============================================================
# MASTER CLASSES
# ============================================================
MASTER_CLASSES = {
    0: "person",        1: "bicycle",       2: "car",
    3: "motorcycle",    4: "bus",           5: "truck",
    6: "traffic light", 7: "stop sign",     8: "bench",
    9: "cat",           10: "dog",          11: "backpack",
    12: "umbrella",     13: "bottle",       14: "chair",
    15: "couch",        16: "dining table", 17: "laptop",
    18: "cell phone",   19: "pothole",      20: "electric pole",
    21: "uncovered manhole", 22: "traffic signs",
    23: "door",         24: "wall",         25: "stairs",
    26: "elevator",     27: "cabinet",      28: "tree",
    29: "bicycle lane",
}

VOC_MAP = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "motorbike": 3, "bus": 4, "truck": 5, "van": 5,
    "lorry": 5, "small lorry": 5,
    "trafficlight": 6, "traffic light": 6,
    "stopsign": 7, "stop sign": 7,
    "bench": 8, "cat": 9, "dog": 10,
    "backpack": 11, "umbrella": 12, "handbag": 11,
    "bottle": 13, "chair": 14, "sofa": 15, "couch": 15,
    "diningtable": 16, "dining table": 16,
    "laptop": 17, "cellphone": 18, "cell phone": 18,
    "pothole": 19, "road damage": 19, "crack": 19,
    "pottedplant": 28, "potted plant": 28,
    "aeroplane": None, "boat": None, "train": None,
    "horse": None, "cow": None, "sheep": None,
    "bird": None, "tvmonitor": None,
}

# VisDrone class → master ID
VISDRONE_REMAP = {
    0: None,  1: 0,   2: 0,   3: 1,
    4: 2,     5: 5,   6: 4,   7: 5,
    8: 3,     9: 1,   10: None, 11: None,
}

# Vehicle folder name → master class ID
VEHICLE_FOLDER_MAP = {
    "bus": 4, "car": 2, "lorry": 5, "small lorry": 5,
    "motorcycles": 3, "motorcycle": 3,
    "pedestrian": 0, "truck": 5, "van": 5,
}


def get_voc_id(name):
    n = name.lower().strip().replace("-", "").replace("_", "")
    if n in VOC_MAP: return VOC_MAP[n]
    n2 = name.lower().strip()
    if n2 in VOC_MAP: return VOC_MAP[n2]
    return None


def write_yaml(output_dir):
    data = {
        'path': os.path.abspath(output_dir).replace('\\', '/'),
        'train': 'train/images',
        'val':   'valid/images',
        'nc': len(MASTER_CLASSES),
        'names': {int(k): v for k, v in MASTER_CLASSES.items()},
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def count_imgs(folder):
    if not os.path.exists(folder): return 0
    return sum(1 for r, d, files in os.walk(folder)
               for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))


# ============================================================
# CONVERTER A: Pascal VOC XML (separate ann_dir + img_dir)
# ============================================================
def convert_voc_pair(ann_dir, img_dir, output_dir, prefix, val_ratio=0.15):
    """Convert Pascal VOC XML annotations to YOLO format."""
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    if not xml_files:
        return 0

    for sp in ['train', 'valid']:
        os.makedirs(os.path.join(output_dir, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sp, 'labels'), exist_ok=True)

    random.seed(42)
    random.shuffle(xml_files)
    val_set = set(xml_files[:max(1, int(len(xml_files) * val_ratio))])

    converted = 0
    class_counts = {}

    for xml_file in xml_files:
        try:
            tree = ET.parse(os.path.join(ann_dir, xml_file))
            root = tree.getroot()

            fn_el = root.find('filename')
            if fn_el is None or not fn_el.text:
                continue
            fname = fn_el.text.strip()

            img_path = None
            base = os.path.splitext(fname)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                for candidate in [
                    os.path.join(img_dir, fname),
                    os.path.join(img_dir, base + ext),
                ]:
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path: break

            if not img_path:
                continue

            sz = root.find('size')
            if sz is None: continue
            iw = int(sz.find('width').text)
            ih = int(sz.find('height').text)
            if iw <= 0 or ih <= 0: continue

            yolo_lines = []
            for obj in root.findall('object'):
                name_el = obj.find('name')
                if name_el is None: continue
                cls_id = get_voc_id(name_el.text)
                if cls_id is None: continue
                diff = obj.find('difficult')
                if diff is not None and int(diff.text) == 1: continue
                bb = obj.find('bndbox')
                if bb is None: continue
                xmin = float(bb.find('xmin').text)
                ymin = float(bb.find('ymin').text)
                xmax = float(bb.find('xmax').text)
                ymax = float(bb.find('ymax').text)
                if xmax <= xmin or ymax <= ymin: continue
                cx = max(0.0, min(1.0, ((xmin+xmax)/2)/iw))
                cy = max(0.0, min(1.0, ((ymin+ymax)/2)/ih))
                bw = max(0.001, min(1.0, (xmax-xmin)/iw))
                bh = max(0.001, min(1.0, (ymax-ymin)/ih))
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            if not yolo_lines:
                continue

            split = "valid" if xml_file in val_set else "train"
            stem = f"{prefix}_{converted:06d}"
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path,
                         os.path.join(output_dir, split, 'images', stem + ext))
            with open(os.path.join(output_dir, split, 'labels', stem + '.txt'), 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted += 1

        except Exception:
            pass

    if converted > 0:
        write_yaml(output_dir)
        for cid, cnt in sorted(class_counts.items()):
            print(f"    {cid}: {MASTER_CLASSES.get(cid, '?')} — {cnt:,}")

    return converted


# ============================================================
# CONVERTER B: Road Damage — fix WRONG class IDs
# road_damage3 class IDs 1,2,3 = crack types → all map to pothole (19)
# ============================================================
def fix_road_damage_labels(source_dir, output_dir):
    """
    road_damage3 flat folder: images + txt labels in same directory.
    Class IDs in labels:
      1 = alligator crack  → pothole (19)
      2 = longitudinal crack → pothole (19)
      3 = transverse crack → pothole (19)
    All map to our master class pothole = 19.
    """
    print(f"\n{'='*50}")
    print(f"[ROAD DAMAGE FIX] {source_dir}")
    print(f"  Remapping class IDs 1,2,3 → pothole (19)")

    all_imgs = []
    for ext in ['.jpg', '.jpeg', '.png']:
        all_imgs += glob.glob(os.path.join(source_dir, f"*{ext}"))
        all_imgs += glob.glob(os.path.join(source_dir, f"*{ext.upper()}"))

    print(f"  Images found: {len(all_imgs)}")

    if not all_imgs:
        print(f"  No images found!")
        return 0

    for sp in ['train', 'valid']:
        os.makedirs(os.path.join(output_dir, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sp, 'labels'), exist_ok=True)

    random.seed(42)
    random.shuffle(all_imgs)
    val_n = max(1, int(len(all_imgs) * 0.15))
    val_set = set(all_imgs[:val_n])

    converted = 0
    skipped_no_label = 0

    for img_path in all_imgs:
        lbl_path = os.path.splitext(img_path)[0] + '.txt'

        if not os.path.exists(lbl_path):
            skipped_no_label += 1
            continue

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_id = int(parts[0])
            # Remap ALL crack types to pothole (19)
            if old_id in [0, 1, 2, 3]:
                parts[0] = '19'
                new_lines.append(' '.join(parts))

        if not new_lines:
            continue

        split = "valid" if img_path in val_set else "train"
        stem = f"road_{converted:06d}"
        ext = os.path.splitext(img_path)[1]

        shutil.copy2(img_path,
                     os.path.join(output_dir, split, 'images', stem + ext))
        with open(os.path.join(output_dir, split, 'labels', stem + '.txt'), 'w') as f:
            f.write('\n'.join(new_lines))
        converted += 1

    print(f"  Converted: {converted:,} (all mapped to pothole class 19)")
    print(f"  Skipped (no label): {skipped_no_label}")

    if converted > 0:
        write_yaml(output_dir)

    return converted


# ============================================================
# CONVERTER C: VisDrone — remap class IDs
# ============================================================
def convert_visdrone(source_dir, output_dir):
    """VisDrone labels already YOLO format but need class ID remapping."""
    print(f"\n{'='*50}")
    print(f"[VISDRONE] {source_dir}")

    for sp in ['train', 'valid']:
        os.makedirs(os.path.join(output_dir, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sp, 'labels'), exist_ok=True)

    converted = 0

    split_dirs = []
    for root, dirs, files in os.walk(source_dir):
        if 'images' in dirs and 'labels' in dirs:
            split_dirs.append(root)

    print(f"  Found {len(split_dirs)} split folder(s)")

    for split_root in split_dirs:
        img_dir = os.path.join(split_root, 'images')
        lbl_dir = os.path.join(split_root, 'labels')
        folder_name = os.path.basename(split_root).lower()
        dst_split = 'valid' if 'val' in folder_name else 'train'

        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {os.path.basename(split_root)}: {len(img_files)} imgs → {dst_split}")

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            base = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(lbl_dir, base + '.txt')

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                old_id = int(parts[0])
                new_id = VISDRONE_REMAP.get(old_id)
                if new_id is None: continue
                parts[0] = str(new_id)
                new_lines.append(' '.join(parts))

            if not new_lines:
                continue

            stem = f"vis_{converted:06d}"
            ext = os.path.splitext(img_file)[1]
            shutil.copy2(img_path,
                         os.path.join(output_dir, dst_split, 'images', stem + ext))
            with open(os.path.join(output_dir, dst_split, 'labels', stem + '.txt'), 'w') as f:
                f.write('\n'.join(new_lines))
            converted += 1

    print(f"  Converted: {converted:,}")
    if converted > 0:
        write_yaml(output_dir)

    return converted


# ============================================================
# CONVERTER D: Vehicle Ped — folder-per-class, no labels
# ============================================================
def convert_vehicle_ped(source_dir, output_dir):
    """Veri Seti/Bus/ Car/ etc — images only, assign class from folder."""
    print(f"\n{'='*50}")
    print(f"[VEHICLE PED] {source_dir}")

    veri_seti = os.path.join(source_dir, "Veri Seti")
    if not os.path.exists(veri_seti):
        print(f"  Veri Seti folder not found")
        return 0

    for sp in ['train', 'valid']:
        os.makedirs(os.path.join(output_dir, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sp, 'labels'), exist_ok=True)

    converted = 0
    random.seed(42)

    for folder_name in os.listdir(veri_seti):
        folder_path = os.path.join(veri_seti, folder_name)
        if not os.path.isdir(folder_path):
            continue

        cls_id = VEHICLE_FOLDER_MAP.get(folder_name.lower())
        if cls_id is None:
            print(f"  Skipping unknown folder: {folder_name}")
            continue

        imgs = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {folder_name}: {len(imgs)} imgs → class {cls_id} "
              f"({MASTER_CLASSES.get(cls_id, '?')})")

        random.shuffle(imgs)
        val_set = set(imgs[:max(1, int(len(imgs) * 0.15))])

        for img_file in imgs:
            img_path = os.path.join(folder_path, img_file)
            split = "valid" if img_file in val_set else "train"
            stem = f"vehped_{converted:06d}"
            ext = os.path.splitext(img_file)[1]

            shutil.copy2(img_path,
                         os.path.join(output_dir, split, 'images', stem + ext))
            with open(os.path.join(output_dir, split, 'labels', stem + '.txt'), 'w') as f:
                f.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")
            converted += 1

    print(f"  Converted: {converted:,}")
    if converted > 0:
        write_yaml(output_dir)

    return converted


# ============================================================
# CONVERTER E: Already YOLO with train/valid/test structure
# ============================================================
def copy_yolo_dataset(source_dir, output_dir):
    """Copy dataset that already has correct YOLO structure."""
    print(f"\n{'='*50}")
    print(f"[COPY YOLO] {source_dir}")

    total = 0
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_dir, split, 'images')
        if os.path.exists(img_dir):
            n = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total += n
            print(f"  {split}: {n} images")

    if total == 0:
        print(f"  No images found")
        return 0

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    write_yaml(output_dir)
    print(f"  Copied {total:,} images")
    return total


# ============================================================
# MAIN
# ============================================================
def convert_all():
    print("=" * 65)
    print("  FORMAT CONVERTER — Fixed for All Structures")
    print("=" * 65)

    total = 0

    # ===== 1. PASCAL VOC (already done) =====
    existing = count_imgs("datasets/extra/pascal_yolo")
    if existing > 1000:
        print(f"\n[PASCAL VOC] Already done — {existing:,} images")
        total += existing

    # ===== 2. POTHOLE ANDREW (already done) =====
    existing = count_imgs("datasets/extra/pothole_yolo1")
    if existing > 50:
        print(f"\n[POTHOLE ANDREW] Already done — {existing:,} images")
        total += existing
    elif os.path.exists("datasets/extra/pothole_andrew/annotations"):
        print(f"\n{'='*50}")
        print(f"[POTHOLE ANDREW]")
        n = convert_voc_pair(
            "datasets/extra/pothole_andrew/annotations",
            "datasets/extra/pothole_andrew/images",
            "datasets/extra/pothole_yolo1",
            "pot"
        )
        print(f"  Converted: {n:,}")
        total += n

    # ===== 3. ROAD DAMAGE 3 — FIX WRONG CLASS IDs =====
    road_out = "datasets/extra/road_yolo3"

    # Always redo this one to fix wrong labels from previous run
    if os.path.exists(road_out):
        print(f"\n[ROAD DAMAGE] Removing old wrong-label version...")
        shutil.rmtree(road_out)

    if os.path.exists("datasets/extra/road_damage3"):
        n = fix_road_damage_labels("datasets/extra/road_damage3", road_out)
        total += n
    else:
        print(f"\n[SKIP] road_damage3 not found")

    # ===== 4. VISDRONE — remap class IDs =====
    existing = count_imgs("datasets/extra/visdrone_yolo")
    if existing > 100:
        print(f"\n[VISDRONE] Already done — {existing:,} images")
        total += existing
    else:
        n = convert_visdrone("datasets/extra/visdrone", "datasets/extra/visdrone_yolo")
        total += n

    # ===== 5. VEHICLE PED — folder-per-class =====
    existing = count_imgs("datasets/extra/vehicle_ped_yolo")
    if existing > 50:
        print(f"\n[VEHICLE PED] Already done — {existing:,} images")
        total += existing
    elif os.path.exists("datasets/extra/vehicle_ped/Veri Seti"):
        n = convert_vehicle_ped(
            "datasets/extra/vehicle_ped",
            "datasets/extra/vehicle_ped_yolo"
        )
        total += n

    # ===== 6. VEHICLE YOLO — already train/valid/test =====
    existing = count_imgs("datasets/extra/vehicle_yolo_fixed")
    if existing > 50:
        print(f"\n[VEHICLE YOLO] Already done — {existing:,} images")
        total += existing
    elif os.path.exists("datasets/extra/vehicle_yolo/VehiclesDetectionDataset"):
        n = copy_yolo_dataset(
            "datasets/extra/vehicle_yolo/VehiclesDetectionDataset",
            "datasets/extra/vehicle_yolo_fixed"
        )
        total += n

    # ===== 7. PEDESTRIAN K — Pascal VOC XML =====
    existing = count_imgs("datasets/extra/pedestrian_yolo")
    if existing > 50:
        print(f"\n[PEDESTRIAN] Already done — {existing:,} images")
        total += existing
    else:
        print(f"\n{'='*50}")
        print(f"[PEDESTRIAN]")
        ped_out = "datasets/extra/pedestrian_yolo"
        ped_total = 0
        for split_folder, pfx in [
            ("datasets/extra/pedestrian_k/Train/Train", "ped_train"),
            ("datasets/extra/pedestrian_k/Test/Test",   "ped_test"),
            ("datasets/extra/pedestrian_k/Val/Val",     "ped_val"),
        ]:
            ann = os.path.join(split_folder, "Annotations")
            img = os.path.join(split_folder, "JPEGImages")
            if os.path.exists(ann) and os.path.exists(img):
                print(f"  Processing {os.path.basename(split_folder)}...")
                n = convert_voc_pair(ann, img, ped_out, pfx)
                ped_total += n
                print(f"    {n:,} images")
        print(f"  Total pedestrian: {ped_total:,}")
        total += ped_total

    # ===== 8. POTHOLE YOLO — already YOLO =====
    existing = count_imgs("datasets/extra/pothole_yolo_fixed")
    if existing > 50:
        print(f"\n[POTHOLE YOLO] Already done — {existing:,} images")
        total += existing
    elif os.path.exists("datasets/extra/pothole_yolo/train/images"):
        n = copy_yolo_dataset(
            "datasets/extra/pothole_yolo",
            "datasets/extra/pothole_yolo_fixed"
        )
        total += n

    # ===== SUMMARY =====
    print(f"\n{'='*65}")
    print(f"  CONVERSION COMPLETE")
    print(f"  Total images: {total:,}")
    print(f"\n  Converted folders:")
    for folder in [
        "datasets/extra/pascal_yolo",
        "datasets/extra/pothole_yolo1",
        "datasets/extra/road_yolo3",
        "datasets/extra/visdrone_yolo",
        "datasets/extra/vehicle_ped_yolo",
        "datasets/extra/vehicle_yolo_fixed",
        "datasets/extra/pedestrian_yolo",
        "datasets/extra/pothole_yolo_fixed",
    ]:
        n = count_imgs(folder)
        if n > 0:
            print(f"    {os.path.basename(folder)}: {n:,}")

    print(f"\n[NEXT] Run: python merge_all_datasets.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    convert_all()
