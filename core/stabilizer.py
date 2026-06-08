"""
stabilizer.py — Temporal smoothing and object tracking for stable detections.
"""

def _iou(a, b):
    """Intersection over Union of two bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1  = max(ax1, bx1)
    iy1  = max(ay1, by1)
    ix2  = min(ax2, bx2)
    iy2  = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id, det):
        self.track_id = track_id
        self.class_name = det['class_name']
        self.class_id = det['class_id']
        self.bbox = list(det['bbox'])  # [x1, y1, x2, y2]
        self.confidence = det['confidence']
        self.age = 1
        self.missing_count = 0
        
        # Enriched properties
        self.zone = det.get('zone', 'center')
        self.proximity = det.get('proximity', 'distant')
        self.risk_level = det.get('risk_level', 'low')
        self.distance_m = det.get('distance_m', 0.0)
        self.distance_str = det.get('distance_str', '')
        self.area_ratio = det.get('area_ratio', 0.0)

    def update(self, det, alpha=0.6):
        # Bbox coordinate smoothing using Exponential Moving Average (EMA)
        self.bbox[0] = int(self.bbox[0] * (1 - alpha) + det['bbox'][0] * alpha)
        self.bbox[1] = int(self.bbox[1] * (1 - alpha) + det['bbox'][1] * alpha)
        self.bbox[2] = int(self.bbox[2] * (1 - alpha) + det['bbox'][2] * alpha)
        self.bbox[3] = int(self.bbox[3] * (1 - alpha) + det['bbox'][3] * alpha)
        
        self.confidence = self.confidence * (1 - alpha) + det['confidence'] * alpha
        self.age += 1
        self.missing_count = 0
        
        # Update properties
        self.zone = det.get('zone', self.zone)
        self.proximity = det.get('proximity', self.proximity)
        self.risk_level = det.get('risk_level', self.risk_level)
        self.distance_m = det.get('distance_m', self.distance_m)
        self.distance_str = det.get('distance_str', self.distance_str)
        self.area_ratio = det.get('area_ratio', self.area_ratio)


class DetectionsStabilizer:
    """
    Stabilizes real-time object detections across consecutive video frames.
    Performs object tracking, box smoothing (EMA), occlusion recovery, 
    and confirms/discards temporary detections.
    """
    def __init__(self, min_confirm_frames=2, max_missing_frames=3, smooth_alpha=0.6):
        self.tracks = []
        self.next_track_id = 1
        self.min_confirm_frames = min_confirm_frames
        self.max_missing_frames = max_missing_frames
        self.smooth_alpha = smooth_alpha

    def update(self, detections):
        """
        Updates active tracks and returns stabilized, smoothed detections.
        """
        # Match detections with active tracks using IoU
        matched_detects = [False] * len(detections)
        
        for track in self.tracks:
            best_iou = 0.0
            best_idx = -1
            
            for idx, det in enumerate(detections):
                if matched_detects[idx]:
                    continue
                if det['class_name'].lower() != track.class_name.lower():
                    continue
                
                iou_val = _iou(track.bbox, det['bbox'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx
            
            # If match found with high IoU, update track
            if best_idx != -1 and best_iou > 0.35:
                track.update(detections[best_idx], alpha=self.smooth_alpha)
                matched_detects[best_idx] = True
            else:
                track.missing_count += 1
                track.age = 0
        
        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if not matched_detects[idx]:
                self.tracks.append(Track(self.next_track_id, det))
                self.next_track_id += 1
                
        # Filter and drop dead tracks (exceeded max missing frames)
        self.tracks = [t for t in self.tracks if t.missing_count <= self.max_missing_frames]
        
        # Build stabilized detection payload
        stabilized = []
        for track in self.tracks:
            # Confirm tracks seen for at least N consecutive frames
            # or if they are highly confident or critical
            is_confirmed = (track.age >= self.min_confirm_frames or 
                            track.confidence > 0.65 or 
                            track.proximity in ('critical', 'close'))
            
            if is_confirmed and track.missing_count == 0:
                # Calculate center based on smoothed bbox
                w = track.bbox[2] - track.bbox[0]
                h = track.bbox[3] - track.bbox[1]
                cx = track.bbox[0] + w // 2
                cy = track.bbox[1] + h // 2
                
                stabilized.append({
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'confidence': track.confidence,
                    'bbox': tuple(track.bbox),
                    'center': (cx, cy),
                    'area_ratio': track.area_ratio,
                    'zone': track.zone,
                    'proximity': track.proximity,
                    'risk_level': track.risk_level,
                    'distance_m': track.distance_m,
                    'distance_str': track.distance_str
                })
                
        return stabilized
