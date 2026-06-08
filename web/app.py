"""
app.py — Flask web server for Assistive Navigation System.
"""

import os
import sys
import time
import base64
import numpy as np
import cv2

# Use gevent for proper WebSocket support
from gevent import monkey
monkey.patch_all()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from core.config import CONFIDENCE_THRESHOLD
from core.detector import Detector
from core.decision import assign_zones, suggest_direction
from core.proximity import add_proximity
from core.risk_classifier import add_risk_levels
from core.scene_describer import describe_scene, generate_alert


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'assistive-nav-secret'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False
)

detector = None
frame_count = 0
last_alert_time = 0
last_alert_text = ""
stabilizers = {}



def to_python(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    Fixes: Object of type int64 is not JSON serializable
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def init_detector(custom_model=None, dual_mode=False):
    global detector
    print("[WEB] Initializing detector...")
    detector = Detector(custom_model_path=custom_model, dual_mode=dual_mode)
    print("[WEB] Detector ready!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'detector': detector is not None,
        'model': detector.model_name if detector else None,
        'device': detector.device if detector else None,
    })


@socketio.on('connect')
def handle_connect():
    print(f"[WEB] Client connected: {request.sid}")
    emit('server_info', {
        'model': detector.model_name if detector else 'None',
        'device': detector.device if detector else 'None',
        'info': detector.get_info() if detector else 'Not loaded',
    })


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WEB] Client disconnected: {request.sid}")
    stabilizers.pop(request.sid, None)


@socketio.on('frame')
def handle_frame(data):
    global frame_count, last_alert_time, last_alert_text

    if detector is None:
        emit('error', {'message': 'Detector not initialized'})
        return

    t_start = time.perf_counter()

    try:
        # Decode base64 JPEG from browser
        img_data = data.get('image', '')
        low_light = data.get('low_light', False)
        pitch = float(data.get('pitch', 0.0))
        send_time = data.get('send_time')
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Run detection with optional low-light preprocessing
        detections = detector.detect(frame, low_light_mode=low_light)

        # Enrich detections
        frame_h, frame_w = frame.shape[:2]
        detections = assign_zones(detections, frame_w)
        detections = add_proximity(detections, pitch_deg=pitch)
        detections = add_risk_levels(detections)

        # Apply temporal stabilizer to smooth boxes and mitigate flickering
        sid = request.sid
        if sid not in stabilizers:
            from core.stabilizer import DetectionsStabilizer
            stabilizers[sid] = DetectionsStabilizer()
        detections = stabilizers[sid].update(detections)

        # Decision
        direction = suggest_direction(detections)
        scene = describe_scene(detections)
        alert = generate_alert(detections, direction)

        # FPS calculation
        t_total = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / t_total if t_total > 0 else 0

        frame_count += 1

        # Alert cooldown — 4.5 seconds so speech finishes before next alert
        # Also skip if same alert repeating
        now = time.time()
        should_speak = False
        if alert and now - last_alert_time >= 4.5 and alert != last_alert_text:
            should_speak = True
            last_alert_time = now
            last_alert_text = alert

        # Build detection list — convert ALL numpy types to Python native
        det_list = []
        for det in detections:
            bbox = det['bbox']
            det_list.append({
                'class_name': str(det['class_name']),
                'confidence': float(round(det['confidence'], 2)),
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'zone': str(det.get('zone', '')),
                'risk_level': str(det.get('risk_level', 'low')),
                'proximity': str(det.get('proximity', '')),
                'area_ratio': float(round(det.get('area_ratio', 0), 4)),
                'distance_m': float(det.get('distance_m', 0.0)),
                'distance_str': str(det.get('distance_str', '')),
            })

        result = {
            'detections': det_list,
            'direction': str(direction),
            'scene': str(scene),
            'alert': str(alert) if should_speak else None,
            'fps': float(round(fps, 1)),
            'inference_ms': float(round(t_total, 1)),
            'num_objects': int(len(detections)),
            'frame_width': int(frame_w),
            'frame_height': int(frame_h),
            'send_time': send_time,
        }

        emit('result', result)

    except Exception as e:
        print(f"[WEB] Error: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})


def run_server(host='0.0.0.0', port=5000, custom_model=None, dual_mode=False):
    init_detector(custom_model=custom_model, dual_mode=dual_mode)

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print("\n" + "=" * 60)
    print("  ASSISTIVE NAVIGATION — WEB SERVER")
    print("=" * 60)
    print(f"\n  PC browser:     http://localhost:{port}")
    print(f"  Phone browser:  http://{local_ip}:{port}")
    print(f"\n  Phone MUST be on same WiFi as this laptop")
    print("=" * 60 + "\n")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        use_reloader=False
    )


if __name__ == '__main__':
    run_server()
    