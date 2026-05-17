"""
alert_manager.py — Speaks EVERY 2 seconds. Full sentences. Never stops.

EXAMPLES OF WHAT IT SAYS:
  "Person detected ahead, please move left"
  "Chair detected on your right, please be careful"
  "Car detected ahead, please stop immediately"
  "Path is clear, you may continue walking"
"""

import time
import threading


class AlertManager:
    """Simple, reliable alert manager. Speaks every 2 seconds."""

    def __init__(self, tts_engine=None):
        self.tts_engine = tts_engine
        self.lock = threading.Lock()
        self._running = True

        # Current alert to speak
        self.current_alert = ""
        self.last_spoken_time = 0.0
        self.last_spoken_message = ""

        # Speak interval
        self.SPEAK_INTERVAL = 2.0  # Speak every 2 seconds

        # Start speaker thread
        self._thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._thread.start()

    def set_tts(self, tts_engine):
        self.tts_engine = tts_engine

    def update_scene(self, detections):
        """Not used. Kept for compatibility."""
        pass

    def set_alert(self, message):
        """Set the current alert message. Called every detection frame."""
        with self.lock:
            self.current_alert = message

    def add_alert(self, message, priority=0, class_name=None,
                  direction=None, detections=None):
        """Set current alert. Speaks on next cycle."""
        with self.lock:
            self.current_alert = message

    def _speaker_loop(self):
        """Runs forever. Speaks current alert every 2 seconds."""
        while self._running:
            now = time.time()

            # Check if 2 seconds passed since last speech
            if now - self.last_spoken_time >= self.SPEAK_INTERVAL:
                message = ""
                with self.lock:
                    message = self.current_alert

                if message and len(message) > 0:
                    self._speak(message)
                    self.last_spoken_time = time.time()
                    self.last_spoken_message = message

            time.sleep(0.1)  # Check 10 times per second

    def _speak(self, message):
        """Speak message using pyttsx3."""
        if self.tts_engine is None:
            print(f"[ALERT] {message}")
            return

        try:
            self.tts_engine.speak(message)
        except Exception as e:
            print(f"[ALERT] TTS error: {e}")
            # Try to reinitialize TTS
            try:
                self.tts_engine._init_engine()
            except Exception:
                pass

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)

            