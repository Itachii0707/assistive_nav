"""
detector.py — YOLO26 detection wrapper with maximum accuracy settings.
"""

import numpy as np
import platform

from core.config import (
    CONFIDENCE_THRESHOLD, NAVIGATION_CLASSES_COCO,
    YOLO_INPUT_SIZE, PREFERRED_GPU_MODEL,
    USE_TTA, INFERENCE_SIZE
)


def get_device():
    """Auto-detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[DEVICE] GPU: {device_name} ({vram:.1f} GB VRAM)")
            return '0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[DEVICE] Apple Silicon GPU")
            return 'mps'
    except ImportError:
        pass
    print("[DEVICE] CPU only")
    return 'cpu'


def is_android():
    """Check if running on Android."""
    try:
        import android
        return True
    except ImportError:
        return platform.machine() in ['aarch64', 'armv7l']


class Detector:
    """
    YOLO26 detector — maximum accuracy configuration.

    Modes:
      COCO only (default) — 36+ objects, no training needed
      Custom only (--model) — 9 navigation objects
      DUAL (--dual --model) — both combined, maximum detection
    """

    def __init__(self, custom_model_path=None, dual_mode=False):
        self.device = get_device()
        self.model = None
        self.custom_model = None
        self.model_name = ""
        self.is_android = is_android()
        self._is_custom_model = False
        self._dual_mode = dual_mode
        self._use_tta = USE_TTA and self.device == '0'
        self._infer_size = INFERENCE_SIZE
        self._load_model(custom_model_path)
        self._warmup()

        print(f"\n[ACCURACY SETTINGS]")
        print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  Input size: {self._infer_size}x{self._infer_size}")
        print(f"  TTA (Test Time Augmentation): {'ON' if self._use_tta else 'OFF'}")
        print(f"  FP16 (Half precision): {'ON' if self.device == '0' else 'OFF'}")
        print()

    def _load_model(self, custom_model_path=None):
        """Load model(s) for maximum accuracy."""
        from ultralytics import YOLO

        # ===== DUAL MODE =====
        if self._dual_mode and custom_model_path:
            coco_name = self._get_best_coco_model()
            self.model = YOLO(coco_name)
            print(f"[DETECTOR] COCO model: {coco_name}")

            self.custom_model = YOLO(custom_model_path)
            self.model_name = f"DUAL: {coco_name} + {custom_model_path}"
            self._is_custom_model = False

            print(f"[DETECTOR] Custom model: {custom_model_path} ({len(self.custom_model.names)} classes)")
            print(f"[DETECTOR] DUAL MODE — Maximum detection")
            return

        # ===== CUSTOM ONLY =====
        if custom_model_path:
            self.model = YOLO(custom_model_path)
            self.model_name = custom_model_path
            self._is_custom_model = True
            print(f"[DETECTOR] Custom model: {custom_model_path} ({len(self.model.names)} classes)")
            for idx, name in self.model.names.items():
                print(f"           {idx}: {name}")
            return

        # ===== COCO ONLY =====
        self._is_custom_model = False
        coco_name = self._get_best_coco_model()
        self.model = YOLO(coco_name)
        self.model_name = coco_name
        print(f"[DETECTOR] Loaded {coco_name}")

    def _get_best_coco_model(self):
        """Pick highest accuracy model for current hardware."""
        import os

        if self.is_android:
            return "yolo26n.pt"

        if self.device == '0':
            for name in [PREFERRED_GPU_MODEL, "yolo26m.pt", "yolo26s.pt", "yolo26n.pt"]:
                if os.path.exists(f"models/{name}"):
                    return f"models/{name}"
            return PREFERRED_GPU_MODEL

        elif self.device == 'mps':
            return "yolo26s.pt"

        else:
            if os.path.exists("models/yolo26n.onnx"):
                return "models/yolo26n.onnx"
            return "yolo26n.pt"

    def _warmup(self):
        """Warm up model with dummy frame."""
        try:
            dummy = np.zeros((self._infer_size, self._infer_size, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, imgsz=self._infer_size)
            if self.custom_model:
                self.custom_model.predict(dummy, verbose=False, imgsz=self._infer_size)
            print("[DETECTOR] Warmup complete")
        except Exception as e:
            print(f"[DETECTOR] Warmup failed: {e}")

    def _run_inference(self, model, frame, filter_classes=None):
        """Run inference with maximum accuracy settings."""
        use_half = self.device == '0'

        predict_args = {
            'conf': CONFIDENCE_THRESHOLD,
            'verbose': False,
            'device': self.device if not self.is_android else None,
            'half': use_half,
            'imgsz': self._infer_size,
            'augment': self._use_tta,
        }

        if filter_classes is not None:
            predict_args['classes'] = filter_classes

        return model.predict(frame, **predict_args)

    def _extract_detections(self, results, frame_h, frame_w, frame_area):
        """
        Extract detection dicts from YOLO results.
        All values converted to native Python types — no numpy int64/float32.
        """
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    # Convert to native Python int — fixes JSON serialization error
                    coords = box.xyxy[0].cpu().numpy()
                    x1 = int(coords[0])
                    y1 = int(coords[1])
                    x2 = int(coords[2])
                    y2 = int(coords[3])

                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = str(results[0].names.get(cls_id, f"class_{cls_id}"))

                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = float(bbox_area / frame_area) if frame_area > 0 else 0.0

                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)

                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': (cx, cy),
                        'area_ratio': area_ratio
                    })

        return detections

    def _remove_duplicates(self, coco_dets, custom_dets):
        """Remove duplicates when both models detect same object."""
        combined = list(coco_dets)

        for cd in custom_dets:
            cx_c, cy_c = cd['center']
            is_duplicate = False

            for i, existing in enumerate(combined):
                cx_e, cy_e = existing['center']
                if abs(cx_c - cx_e) < 50 and abs(cy_c - cy_e) < 50:
                    if cd['confidence'] > existing['confidence']:
                        combined[i] = cd
                    is_duplicate = True
                    break

            if not is_duplicate:
                combined.append(cd)

        return combined

    def detect(self, frame):
        """Run detection with maximum accuracy on frame."""
        if frame is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        all_detections = []

        # Run primary model
        try:
            if self._is_custom_model:
                results = self._run_inference(self.model, frame)
            else:
                results = self._run_inference(
                    self.model, frame,
                    filter_classes=NAVIGATION_CLASSES_COCO
                )
            all_detections = self._extract_detections(
                results, frame_h, frame_w, frame_area
            )
        except Exception as e:
            print(f"[DETECTOR] Primary error: {e}")

        # Run custom model (dual mode)
        if self._dual_mode and self.custom_model:
            try:
                custom_results = self._run_inference(self.custom_model, frame)
                custom_dets = self._extract_detections(
                    custom_results, frame_h, frame_w, frame_area
                )
                all_detections = self._remove_duplicates(all_detections, custom_dets)
            except Exception as e:
                print(f"[DETECTOR] Custom error: {e}")

        return all_detections

    def get_info(self):
        tta = " +TTA" if self._use_tta else ""
        size = f" {self._infer_size}px"
        if self._dual_mode:
            return f"DUAL MODE | {self.device}{tta}{size}"
        model_type = "Custom" if self._is_custom_model else "COCO"
        return f"{self.model_name} | {self.device} | {model_type}{tta}{size}"

    