"""
tts_desktop.py — pyttsx3 TTS that NEVER hangs.
Creates fresh engine each speak call to avoid blocking forever.
"""

import threading


class DesktopTTS:
    """Reliable offline TTS. Fresh engine each time prevents hangs."""

    def __init__(self):
        self.lock = threading.Lock()
        self._available = False
        self._rate = 175
        self._volume = 1.0
        self._init_engine()

    def _init_engine(self):
        """Test if pyttsx3 works."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._available = True
            print("[TTS] pyttsx3 initialized")
        except Exception as e:
            print(f"[TTS] Failed: {e}")
            self._available = False

    def speak(self, message):
        """Speak a message. Creates fresh engine each time to prevent hanging."""
        if not self._available:
            print(f"[TTS] (disabled) {message}")
            return

        with self.lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', self._rate)
                engine.setProperty('volume', self._volume)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def set_rate(self, rate):
        self._rate = rate

    def set_volume(self, volume):
        self._volume = volume

    def is_available(self):
        return self._available

    def stop(self):
        pass

    