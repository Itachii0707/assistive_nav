"""
decision.py — Zone mapping + safe direction suggestion.
"""

from core.config import ZONE_LEFT_BOUNDARY, ZONE_RIGHT_BOUNDARY


def get_zone(cx, frame_width):
    """Determine zone based on center x coordinate."""
    ratio = cx / frame_width if frame_width > 0 else 0.5
    if ratio < ZONE_LEFT_BOUNDARY:
        return "left"
    elif ratio > ZONE_RIGHT_BOUNDARY:
        return "right"
    else:
        return "center"


def assign_zones(detections, frame_width):
    """Add zone info to each detection."""
    for det in detections:
        cx, _ = det['center']
        det['zone'] = get_zone(cx, frame_width)
    return detections


def suggest_direction(detections):
    """
    Determine safe navigation direction based on zone occupancy and risk.

    Returns:
        str: "move left", "move right", "path clear", or "stop"
    """
    if not detections:
        return "path clear"

    zone_risk = {'left': 0, 'center': 0, 'right': 0}
    has_critical = False

    for det in detections:
        zone = det.get('zone', 'center')
        risk = det.get('risk_level', 'low')

        if risk == 'high':
            zone_risk[zone] += 3
            if det.get('area_ratio', 0) >= 0.20:
                has_critical = True
        elif risk == 'medium':
            zone_risk[zone] += 1

    if has_critical:
        return "stop"

    if all(v == 0 for v in zone_risk.values()):
        return "path clear"

    if zone_risk['center'] > 0:
        if zone_risk['left'] <= zone_risk['right']:
            return "move left"
        else:
            return "move right"

    if zone_risk['left'] > 0 and zone_risk['right'] == 0:
        return "move right"
    if zone_risk['right'] > 0 and zone_risk['left'] == 0:
        return "move left"

    return "path clear"