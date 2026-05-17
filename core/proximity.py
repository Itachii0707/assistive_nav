"""
proximity.py — Estimate obstacle distance using bounding box area ratio.
"""

from core.config import (
    PROXIMITY_CRITICAL, PROXIMITY_CLOSE, PROXIMITY_MEDIUM, PROXIMITY_FAR
)


def estimate_proximity(area_ratio):
    """Classify proximity based on bbox area as fraction of frame."""
    if area_ratio >= PROXIMITY_CRITICAL:
        return "critical"
    elif area_ratio >= PROXIMITY_CLOSE:
        return "close"
    elif area_ratio >= PROXIMITY_MEDIUM:
        return "medium"
    elif area_ratio >= PROXIMITY_FAR:
        return "far"
    else:
        return "distant"


def add_proximity(detections):
    """Add proximity info to each detection."""
    for det in detections:
        det['proximity'] = estimate_proximity(det.get('area_ratio', 0))
    return detections
