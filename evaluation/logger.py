"""
logger.py — Detection history logger with timestamps.
"""

import time
import json


class DetectionLogger:
    """Logs detection events with timestamps to file."""

    def __init__(self, log_file="detection_log.json"):
        self.log_file = log_file
        self.entries = []

    def log(self, detections, direction, fps):
        entry = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'epoch': time.time(),
            'fps': round(fps, 1),
            'direction': direction,
            'num_objects': len(detections),
            'objects': [
                {
                    'class': d['class_name'],
                    'confidence': round(d['confidence'], 2),
                    'zone': d.get('zone', ''),
                    'risk': d.get('risk_level', ''),
                    'proximity': d.get('proximity', ''),
                }
                for d in detections
            ]
        }
        self.entries.append(entry)
        if len(self.entries) > 1000:
            self.entries = self.entries[-500:]

    def save(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.entries, f, indent=2)
            print(f"[LOGGER] Saved {len(self.entries)} entries to {self.log_file}")
        except Exception as e:
            print(f"[LOGGER] Save error: {e}")

    def get_stats(self):
        if not self.entries:
            return {}
        avg_fps = sum(e['fps'] for e in self.entries) / len(self.entries)
        avg_objects = sum(e['num_objects'] for e in self.entries) / len(self.entries)
        directions = {}
        for e in self.entries:
            d = e['direction']
            directions[d] = directions.get(d, 0) + 1
        return {
            'total_frames': len(self.entries),
            'avg_fps': round(avg_fps, 1),
            'avg_objects': round(avg_objects, 1),
            'direction_counts': directions,
        }
    