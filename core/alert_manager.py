"""
alert_manager.py — Desktop TTS alert manager.

Speaks every 5 seconds. Never repeats same message back-to-back.
Path clear spoken less frequently (every 10 seconds).
"""

import time
import threading
from core.config import ALERT_SPEAK_INTERVAL, PATH_CLEAR_INTERVAL


class AlertManager:
    """Reliable alert manager for pyttsx3 desktop TTS."""

    def __init__(self, tts_engine=None):
        self.tts_engine = tts_engine
        self.lock = threading.Lock()
        self._running = True

        self.current_alert      = ""
        self.last_spoken_time   = 0.0
        self.last_spoken_message = ""
        self.last_path_clear_time = 0.0

        # 5 seconds — enough for pyttsx3 to finish full sentence
        self.SPEAK_INTERVAL      = max(ALERT_SPEAK_INTERVAL, 5.0)
        # Path clear is less urgent — speak every 10 seconds
        self.PATH_CLEAR_INTERVAL = max(PATH_CLEAR_INTERVAL, 10.0)

        self._thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._thread.start()

    def set_tts(self, tts_engine):
        self.tts_engine = tts_engine

    def update_scene(self, detections):
        """Kept for compatibility."""
        pass

    def set_alert(self, message):
        """Set current alert. Called every detection frame."""
        with self.lock:
            self.current_alert = message

    def add_alert(self, message, priority=0, class_name=None,
                  direction=None, detections=None):
        """Set current alert. Speaks on next cycle."""
        with self.lock:
            self.current_alert = message

    def _speaker_loop(self):
        """Runs forever. Speaks current alert on interval."""
        while self._running:
            now = time.time()

            with self.lock:
                message = self.current_alert

            if message:
                is_path_clear = "path is clear" in message.lower()

                if is_path_clear:
                    # Path clear — speak every PATH_CLEAR_INTERVAL
                    if now - self.last_path_clear_time >= self.PATH_CLEAR_INTERVAL:
                        self._speak(message)
                        self.last_path_clear_time = time.time()
                        self.last_spoken_time     = time.time()
                        self.last_spoken_message  = message
                else:
                    # Obstacle alert — speak every SPEAK_INTERVAL
                    # Skip if same message already spoken (no repeating same warning)
                    time_ok    = (now - self.last_spoken_time >= self.SPEAK_INTERVAL)
                    message_ok = (message != self.last_spoken_message)

                    if time_ok and message_ok:
                        self._speak(message)
                        self.last_spoken_time    = time.time()
                        self.last_spoken_message = message

            time.sleep(0.1)

    def _speak(self, message):
        """Speak message using pyttsx3."""
        if self.tts_engine is None:
            print(f"[ALERT] {message}")
            return

        try:
            self.tts_engine.speak(message)
        except Exception as e:
            print(f"[ALERT] TTS error: {e}")
            try:
                self.tts_engine._init_engine()
            except Exception:
                pass

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)



            