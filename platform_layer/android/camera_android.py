"""
camera_android.py — Android camera handler using Kivy Camera widget.
"""

import threading


class AndroidCamera:
    """Camera handler for Android using Kivy's Camera widget."""

    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self._running = False
        self.fps = 0.0
        self.camera_widget = None

    def set_camera_widget(self, widget):
        self.camera_widget = widget

    def open(self):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.INTERNET,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])
            print("[CAMERA] Android permissions requested")
            return True
        except ImportError:
            print("[CAMERA] Not running on Android")
            return False

    def start(self):
        self._running = True
        if self.camera_widget:
            self.camera_widget.play = True

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_fps(self):
        return self.fps

    def stop(self):
        self._running = False
        if self.camera_widget:
            self.camera_widget.play = False

    def is_opened(self):
        return self._running
    