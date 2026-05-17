"""
risk_classifier.py — Classify obstacles into Low / Medium / High risk.
"""


def classify_risk(detection):
    """Classify risk based on proximity and zone."""
    proximity = detection.get('proximity', 'distant')
    zone = detection.get('zone', 'center')

    if proximity in ('critical', 'close'):
        return 'high'
    elif proximity == 'medium':
        if zone == 'center':
            return 'high'
        else:
            return 'medium'
    elif proximity == 'far':
        if zone == 'center':
            return 'medium'
        else:
            return 'low'
    else:
        return 'low'


def add_risk_levels(detections):
    """Add risk_level to each detection."""
    for det in detections:
        det['risk_level'] = classify_risk(det)
    return detections
