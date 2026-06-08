"""
proximity.py — Estimate obstacle distance using bounding box area ratio.
"""

import math

from core.config import (
    PROXIMITY_CRITICAL, PROXIMITY_CLOSE, PROXIMITY_MEDIUM, PROXIMITY_FAR,
    FOCAL_LENGTH_PX, AVERAGE_CLASS_HEIGHTS
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


def estimate_distance_m(class_name, bbox, pitch_deg=0.0):
    """Estimate actual distance to object in meters using camera projection with tilt correction."""
    x1, y1, x2, y2 = bbox
    bbox_h = max(1, y2 - y1)
    
    # Get typical physical height in meters
    known_h = AVERAGE_CLASS_HEIGHTS.get(class_name.lower(), 0.6)
    
    # Convert pitch tilt to radians
    pitch_rad = math.radians(abs(pitch_deg))
    cos_tilt = math.cos(pitch_rad)
    
    # Compensate for camera foreshortening due to device pitch tilt
    if cos_tilt > 0.2:
        effective_focal = FOCAL_LENGTH_PX / cos_tilt
    else:
        effective_focal = FOCAL_LENGTH_PX
    
    # Distance = (Known_Height * Effective_Focal) / Pixel_Height
    distance = (known_h * effective_focal) / bbox_h
    return max(0.1, min(15.0, round(float(distance), 1)))


def add_proximity(detections, pitch_deg=0.0):
    """Add proximity and distance info to each detection."""
    for det in detections:
        det['proximity'] = estimate_proximity(det.get('area_ratio', 0))
        det['distance_m'] = estimate_distance_m(det.get('class_name', 'object'), det['bbox'], pitch_deg=pitch_deg)
        det['distance_str'] = f"{det['distance_m']}m"
    return detections


