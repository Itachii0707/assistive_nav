"""
scene_describer.py — Generate FULL SENTENCE descriptions and alerts.

EXAMPLES:
  "Person detected ahead of you, please move to your left"
  "Car and person detected ahead of you, please stop immediately"
  "Chair detected on your right, please be careful"
  "Path is clear, you may continue walking"
"""

from collections import Counter


def describe_scene(detections):
    """Generate scene description for display."""
    if not detections:
        return "Path is clear"

    zone_objects = {'left': [], 'center': [], 'right': []}
    for det in detections:
        zone = det.get('zone', 'center')
        zone_objects[zone].append(det['class_name'])

    parts = []
    for zone, label in [('center', 'ahead'), ('left', 'on left'), ('right', 'on right')]:
        items = zone_objects[zone]
        if not items:
            continue
        counts = Counter(items)
        for obj_name, count in counts.most_common():
            if count == 1:
                parts.append(f"{obj_name} {label}")
            else:
                parts.append(f"{count} {obj_name}s {label}")

    if not parts:
        return "Path is clear"

    return ", ".join(parts[:3])


def generate_alert(detections, direction):
    """
    Generate FULL SENTENCE alert for TTS.

    Returns natural speech like:
      "Person detected ahead of you, please move to your left"
      "Warning! Car very close ahead of you, please stop immediately"
      "Path is clear, you may continue walking"
    """

    # === No obstacles ===
    if not detections or direction == "path clear":
        return "Path is clear, you may continue walking"

    # === Find most critical object ===
    high_risk = [d for d in detections if d.get('risk_level') == 'high']
    medium_risk = [d for d in detections if d.get('risk_level') == 'medium']

    if high_risk:
        obj = high_risk[0]
    elif medium_risk:
        obj = medium_risk[0]
    else:
        obj = detections[0]

    obj_name = obj['class_name']
    zone = obj.get('zone', 'center')
    proximity = obj.get('proximity', 'medium')
    risk = obj.get('risk_level', 'low')

    # === Build location phrase ===
    if zone == 'center':
        location = "ahead of you"
    elif zone == 'left':
        location = "on your left"
    else:
        location = "on your right"

    # === Count other objects ===
    total = len(detections)
    other_names = set()
    for d in detections:
        if d['class_name'] != obj_name:
            other_names.add(d['class_name'])

    # === Build object phrase ===
    if other_names and total > 1:
        others = list(other_names)[:2]
        object_phrase = f"{obj_name} and {' and '.join(others)}"
    else:
        if total > 1:
            same_count = sum(1 for d in detections if d['class_name'] == obj_name)
            if same_count > 1:
                object_phrase = f"{same_count} {obj_name}s"
            else:
                object_phrase = obj_name
        else:
            object_phrase = obj_name

    # === Build full sentence based on direction and risk ===
    if direction == "stop":
        if proximity == "critical":
            alert = f"Warning! {object_phrase} very close {location}, please stop immediately"
        elif proximity == "close":
            alert = f"Caution! {object_phrase} close {location}, please stop now"
        else:
            alert = f"{object_phrase} detected {location}, please stop"

    elif direction == "move left":
        if risk == "high":
            alert = f"Caution! {object_phrase} detected {location}, please move to your left"
        elif risk == "medium":
            alert = f"{object_phrase} detected {location}, move to your left to avoid"
        else:
            alert = f"{object_phrase} detected {location}, please move left"

    elif direction == "move right":
        if risk == "high":
            alert = f"Caution! {object_phrase} detected {location}, please move to your right"
        elif risk == "medium":
            alert = f"{object_phrase} detected {location}, move to your right to avoid"
        else:
            alert = f"{object_phrase} detected {location}, please move right"

    else:
        if risk == "high":
            alert = f"Warning! {object_phrase} detected {location}, please be careful"
        elif risk == "medium":
            alert = f"{object_phrase} detected {location}, please be careful"
        else:
            alert = f"{object_phrase} nearby {location}, please be aware"

    return alert

