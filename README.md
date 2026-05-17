# Assistive Navigation System for the Visually Impaired

Real-time obstacle detection + audio guidance using YOLO26.

## Quick Start

pip install -r requirements.txt
python test_yolo26.py
python test_camera.py
python test_tts.py
python main.py

## Controls
- q — Quit
- b — Print benchmark stats

## Architecture
Camera → YOLO26 → Zone Mapping → Proximity → Risk → Direction → TTS
3 threads: Camera | Detection | UI+TTS