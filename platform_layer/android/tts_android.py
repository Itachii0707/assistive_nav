"""
tts_android.py — Android native TTS using Plyer.
"""

import threading


class AndroidTTS:
    """Android text-to-speech using plyer."""

    def __init__(self):
        self._available = False
        self.lock = threading.Lock()
        self._init_tts()

    def _init_tts(self):
        try:
            from plyer import tts as plyer_tts
            self._tts = plyer_tts
            self._available = True
            print("[TTS] Android TTS initialized via Plyer")
        except Exception as e:
            print(f"[TTS] Android TTS init failed: {e}")
            self._available = False

    def speak(self, message):
        if not self._available:
            print(f"[TTS] (disabled) {message}")
            return
        with self.lock:
            try:
                self._tts.speak(message)
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def is_available(self):
        return self._available

    def stop(self):
        pass
    