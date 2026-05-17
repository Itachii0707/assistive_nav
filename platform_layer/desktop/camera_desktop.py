"""
camera_desktop.py — OpenCV webcam handler with threaded capture.
"""

import cv2
import threading
import platform
import time

from core.config import CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS, BUFFER_SIZE


class DesktopCamera:
    """Threaded webcam capture using OpenCV."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.fps = 0.0

    def open(self):
        system = platform.system()

        if system == 'Windows':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        elif system == 'Linux':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

        print(f"[CAMERA] Opened camera {self.camera_index} "
              f"({int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
        return True

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        retry_count = 0
        prev_time = time.time()
        frame_count = 0

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                retry_count += 1
                if retry_count > 5:
                    print("[CAMERA] Too many failed reads, attempting reconnect...")
                    self._reconnect()
                    retry_count = 0
                time.sleep(0.01)
                continue

            retry_count = 0
            with self.lock:
                self.frame = frame

            frame_count += 1
            elapsed = time.time() - prev_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                prev_time = time.time()

        self.cap.release()

    def _reconnect(self):
        if self.cap:
            self.cap.release()
        time.sleep(1)
        try:
            self.open()
        except RuntimeError:
            print("[CAMERA] Reconnect failed")

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_fps(self):
        return self.fps

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[CAMERA] Stopped")

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
    