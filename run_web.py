"""
run_web.py — Start Assistive Navigation web server.

USAGE:
    python run_web.py                                    # COCO model
    python run_web.py --model path/to/best.pt            # Custom model
    python run_web.py --dual --model path/to/best.pt     # DUAL mode
    python run_web.py --port 8080                        # Custom port

THEN OPEN:
    PC:     http://localhost:5000
    Phone:  http://YOUR_LAPTOP_IP:5000  (same WiFi)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import run_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assistive Navigation Web Server")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to custom YOLO model')
    parser.add_argument('--dual', action='store_true',
                        help='DUAL mode: COCO + custom model')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host (default: 0.0.0.0)')
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        custom_model=args.model,
        dual_mode=args.dual,
    )
    