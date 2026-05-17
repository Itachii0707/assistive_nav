"""
main.py — Assistive Navigation System entry point.

MODES:
  python main.py                                    # COCO model (36+ objects)
  python main.py --model path/to/best.pt            # Custom model
  python main.py --dual --model path/to/best.pt     # DUAL mode (maximum detection)
  python main.py --mode kivy                        # Kivy UI

CONTROLS:
  q — Quit
  b — Print benchmark stats
"""

import os
import sys
import time
import threading
import gc
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_INPUT_SIZE,
    FRAME_SKIP_THRESHOLD_FPS, LOW_RES_WIDTH, LOW_RES_HEIGHT,
    MEMORY_WARNING_PERCENT, DETECTION_PERSISTENCE_FRAMES,
    COLOR_HIGH_RISK, COLOR_MEDIUM_RISK, COLOR_LOW_RISK, COLOR_ZONE_LINE,
    ZONE_LEFT_BOUNDARY, ZONE_RIGHT_BOUNDARY,
    FONT_SCALE, FONT_THICKNESS,
)
from core.detector import Detector, is_android
from core.decision import assign_zones, suggest_direction
from core.proximity import add_proximity
from core.risk_classifier import add_risk_levels
from core.alert_manager import AlertManager
from core.scene_describer import describe_scene, generate_alert
from evaluation.benchmark import Benchmark
from evaluation.logger import DetectionLogger


class NavigationSystem:
    """Main navigation system."""

    def __init__(self, custom_model=None, dual_mode=False):
        print("=" * 50)
        print("  ASSISTIVE NAVIGATION SYSTEM")
        print("=" * 50)

        print("\n[INIT] Loading detector...")
        self.detector = Detector(custom_model_path=custom_model, dual_mode=dual_mode)

        print("[INIT] Opening camera...")
        from platform_layer.desktop.camera_desktop import DesktopCamera
        self.camera = DesktopCamera(camera_index=0)
        self.camera.open()
        self.camera.start()

        print("[INIT] Initializing TTS...")
        from platform_layer.desktop.tts_desktop import DesktopTTS
        self.tts = DesktopTTS()

        self.alert_mgr = AlertManager(tts_engine=self.tts)
        self.bench = Benchmark()
        self.logger = DetectionLogger()

        self._running = False
        self.latest_detections = []
        self.latest_direction = "Initializing..."
        self.latest_scene = ""
        self.det_lock = threading.Lock()
        self.frame_count = 0
        self.prev_detections = []

        print("[INIT] System ready!\n")

    def _detection_loop(self):
        """Detection thread — runs YOLO on frames continuously."""
        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            t_start = time.perf_counter()

            detections = self.detector.detect(frame)

            t_infer = (time.perf_counter() - t_start) * 1000
            self.bench.record_inference(t_infer)

            frame_h, frame_w = frame.shape[:2]
            detections = assign_zones(detections, frame_w)
            detections = add_proximity(detections)
            detections = add_risk_levels(detections)
            detections = self._smooth(detections)

            direction = suggest_direction(detections)
            scene = describe_scene(detections)

            # Generate full sentence alert
            # Alert manager speaks it every 2 seconds automatically
            alert = generate_alert(detections, direction)
            self.alert_mgr.set_alert(alert)

            t_total = (time.perf_counter() - t_start) * 1000
            self.bench.record_pipeline(t_total)

            with self.det_lock:
                self.latest_detections = detections
                self.latest_direction = direction
                self.latest_scene = scene

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.logger.log(detections, direction, self.bench.get_fps())

            if self.frame_count % 100 == 0:
                mem = self.bench.get_memory_usage()
                if mem > MEMORY_WARNING_PERCENT:
                    gc.collect()

    def _smooth(self, detections):
        self.prev_detections = detections
        return detections

    def _draw_overlay(self, frame, detections, direction, scene):
        h, w = frame.shape[:2]

        z1 = int(w * ZONE_LEFT_BOUNDARY)
        z2 = int(w * ZONE_RIGHT_BOUNDARY)
        cv2.line(frame, (z1, 0), (z1, h), COLOR_ZONE_LINE, 1)
        cv2.line(frame, (z2, 0), (z2, h), COLOR_ZONE_LINE, 1)

        cv2.putText(frame, "LEFT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE_LINE, 1)
        cv2.putText(frame, "CENTER", (z1 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE_LINE, 1)
        cv2.putText(frame, "RIGHT", (z2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONE_LINE, 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            risk = det.get('risk_level', 'low')
            color = (COLOR_HIGH_RISK if risk == 'high'
                     else COLOR_MEDIUM_RISK if risk == 'medium'
                     else COLOR_LOW_RISK)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

            info = f"{risk.upper()} | {det.get('proximity', '')}"
            cv2.putText(frame, info, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        fps = self.bench.get_fps()
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Objects: {len(detections)}", (w - 160, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        dir_color = (0, 0, 255) if direction == "stop" else (0, 255, 0)
        cv2.rectangle(frame, (0, h - 70), (w, h - 35), (0, 0, 0), -1)
        cv2.putText(frame, direction.upper(), (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, dir_color, 2)

        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, scene, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        model_info = self.detector.get_info()
        cv2.putText(frame, model_info, (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        return frame

    def run(self):
        self._running = True

        det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        det_thread.start()

        print("[RUN] Press 'q' to quit, 'b' for benchmark\n")

        while self._running:
            t0 = self.bench.start_frame()

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            with self.det_lock:
                dets = self.latest_detections.copy()
                direction = self.latest_direction
                scene = self.latest_scene

            display = self._draw_overlay(frame, dets, direction, scene)

            self.bench.end_frame(t0)

            cv2.imshow("Assistive Navigation", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('b'):
                self.bench.print_report()

        self.shutdown()

    def shutdown(self):
        print("\n[SHUTDOWN] Stopping...")
        self._running = False
        self.alert_mgr.stop()
        self.camera.stop()
        self.logger.save()
        cv2.destroyAllWindows()

        stats = self.logger.get_stats()
        if stats:
            print(f"\n[STATS] Frames: {stats['total_frames']} | "
                  f"Avg FPS: {stats['avg_fps']} | "
                  f"Avg Objects: {stats['avg_objects']}")

        self.bench.print_report()
        print("[SHUTDOWN] Done.")


def run_desktop(custom_model=None, dual_mode=False):
    system = NavigationSystem(custom_model=custom_model, dual_mode=dual_mode)
    try:
        system.run()
    except KeyboardInterrupt:
        system.shutdown()
    except Exception as e:
        print(f"[FATAL] {e}")
        system.shutdown()
        raise


def run_kivy(custom_model=None, dual_mode=False):
    os.environ['KIVY_NO_ARGS'] = '1'
    os.environ['KIVY_NO_CONSOLELOG'] = '1'

    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.clock import Clock
    from kivy.graphics.texture import Texture
    from kivy.lang import Builder

    Builder.load_file('ui/main_ui.kv')

    class NavigationApp(BoxLayout):
        pass

    class AssistiveNavApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.nav_system = None

        def build(self):
            self.root = NavigationApp()
            return self.root

        def on_start(self):
            threading.Thread(target=self._init_system, daemon=True).start()

        def _init_system(self):
            self.nav_system = NavigationSystem(
                custom_model=custom_model, dual_mode=dual_mode
            )
            self.nav_system._running = True
            threading.Thread(target=self.nav_system._detection_loop, daemon=True).start()
            Clock.schedule_interval(self._update_ui, 1.0 / 30)

        def _update_ui(self, dt):
            if not self.nav_system:
                return

            frame = self.nav_system.camera.read()
            if frame is None:
                return

            with self.nav_system.det_lock:
                dets = self.nav_system.latest_detections.copy()
                direction = self.nav_system.latest_direction
                scene = self.nav_system.latest_scene

            display = self.nav_system._draw_overlay(frame, dets, direction, scene)

            buf = cv2.flip(display, 0)
            buf = buf.tobytes()
            texture = Texture.create(size=(display.shape[1], display.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.root.ids.camera_feed.texture = texture
            self.root.ids.fps_label.text = f"FPS: {self.nav_system.bench.get_fps():.0f}"
            self.root.ids.direction_label.text = direction.upper()
            self.root.ids.scene_label.text = scene
            self.root.ids.model_label.text = self.nav_system.detector.model_name
            self.root.ids.device_label.text = self.nav_system.detector.device

            if direction == "stop":
                self.root.ids.direction_label.color = (1, 0, 0, 1)
            elif direction == "path clear":
                self.root.ids.direction_label.color = (0, 1, 0, 1)
            else:
                self.root.ids.direction_label.color = (1, 0.65, 0, 1)

        def on_stop(self):
            if self.nav_system:
                self.nav_system.shutdown()

    AssistiveNavApp().run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assistive Navigation System")
    parser.add_argument('--mode', choices=['desktop', 'kivy'], default='desktop',
                        help='UI mode: desktop (OpenCV) or kivy')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to custom YOLO model')
    parser.add_argument('--dual', action='store_true',
                        help='DUAL mode: COCO + custom model together')
    args = parser.parse_args()

    if args.mode == 'kivy' or is_android():
        run_kivy(custom_model=args.model, dual_mode=args.dual)
    else:
        run_desktop(custom_model=args.model, dual_mode=args.dual)

        