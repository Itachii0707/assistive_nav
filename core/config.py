"""
config.py — ALL constants in one place.
Change any setting here — affects entire system automatically.
"""

import os

# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.20      # Minimum confidence to accept detection
YOLO_INPUT_SIZE = 640            # Default input size
INFERENCE_SIZE = 960             # High accuracy input size (bigger = more accurate)
PREFERRED_GPU_MODEL = "yolo26m.pt"  # Best model for GPU
USE_TTA = False                  # Test Time Augmentation (YOLO26 doesn't support)

# ============================================================
# COCO NAVIGATION CLASSES — 37 classes relevant for navigation
# These are filtered from the 80-class COCO dataset
# Only these classes detected by the COCO model (yolo26m)
# ============================================================

NAVIGATION_CLASSES_COCO = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    12,  # parking meter
    13,  # bench
    15,  # cat
    16,  # dog
    17,  # horse
    19,  # cow
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    36,  # skateboard
    39,  # bottle
    41,  # cup
    43,  # knife
    45,  # bowl
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    61,  # toilet
    62,  # tv
    63,  # laptop
    67,  # cell phone
    72,  # refrigerator
    73,  # book
    75,  # vase
    76,  # scissors
]

# Custom model — all classes (no filter needed, model already trained on navigation)
NAVIGATION_CLASSES_CUSTOM = None  # None = detect all classes

# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0                 # Default webcam index
CAMERA_WIDTH = 640               # Capture width
CAMERA_HEIGHT = 480              # Capture height
TARGET_FPS = 30                  # Target capture FPS
BUFFER_SIZE = 1                  # Frame buffer size (1 = latest frame only)

# ============================================================
# ZONE BOUNDARIES — Frame divided into 3 zones
# LEFT: 0 to 33%, CENTER: 33 to 66%, RIGHT: 66 to 100%
# ============================================================

ZONE_LEFT_BOUNDARY = 0.33        # Where left zone ends
ZONE_RIGHT_BOUNDARY = 0.66       # Where right zone begins

# ============================================================
# PROXIMITY THRESHOLDS — Based on bbox area ratio
# bbox area / frame area → proximity tier
# ============================================================

PROXIMITY_CRITICAL = 0.20        # >20% of frame → VERY CLOSE → STOP
PROXIMITY_CLOSE = 0.15           # >15% of frame → CLOSE → urgent warning
PROXIMITY_MEDIUM = 0.05          # >5% of frame → MEDIUM → caution
PROXIMITY_FAR = 0.02             # >2% of frame → FAR → low priority
                                 # <2% of frame → DISTANT → informational

# ============================================================
# TTS SETTINGS
# ============================================================

TTS_RATE = 175                   # Speech rate (words per minute)
TTS_VOLUME = 1.0                 # Volume (0.0 to 1.0)
TTS_MIN_GAP = 1.0                # Minimum gap between TTS calls (seconds)
MAX_TTS_QUEUE_SIZE = 3           # Max alerts in queue before flushing

# Alert manager settings
ALERT_SPEAK_INTERVAL = 2.0       # Speak alert every 2 seconds
PATH_CLEAR_INTERVAL = 5.0        # "Path is clear" spoken once per 5 seconds

# ============================================================
# RISK CLASSIFICATION COLORS — BGR format for OpenCV
# ============================================================

COLOR_HIGH_RISK = (0, 0, 255)    # Red — HIGH risk
COLOR_MEDIUM_RISK = (0, 145, 255)  # Orange — MEDIUM risk
COLOR_LOW_RISK = (0, 230, 118)   # Green — LOW risk
COLOR_ZONE_LINE = (0, 229, 255)  # Cyan — zone divider lines

# ============================================================
# DISPLAY SETTINGS
# ============================================================

FONT_SCALE = 0.5
FONT_THICKNESS = 1
WINDOW_NAME = "Assistive Navigation System"

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

FRAME_SKIP_THRESHOLD_FPS = 10   # Below this FPS → reduce resolution
LOW_RES_WIDTH = 320              # Low resolution width when FPS drops
LOW_RES_HEIGHT = 240             # Low resolution height when FPS drops
MEMORY_WARNING_PERCENT = 85      # RAM usage above this → trigger GC

# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

DETECTION_PERSISTENCE_FRAMES = 3  # Keep detection visible for N frames

# ============================================================
# LOGGING AND EVALUATION
# ============================================================

LOG_FILE = "detection_log.json"
MAX_LOG_ENTRIES = 1000
BENCHMARK_ROLLING_WINDOW = 30    # Rolling average over N frames

# ============================================================
# MODEL PATHS
# ============================================================

MODELS_DIR = "models"
RUNS_DIR = "runs"

# Auto-find best trained model paths
def get_best_model_path():
    """Returns path to best trained model. Prefers yolo26m trained model."""
    import glob

    # Prefer max accuracy model (yolo26m trained)
    max_paths = glob.glob("runs/**/combined_model_max/weights/best.pt", recursive=True)
    if max_paths:
        return max_paths[0]

    # Then yolo26m basic
    m_paths = glob.glob("runs/**/combined_model_m/weights/best.pt", recursive=True)
    if m_paths:
        return m_paths[0]

    # Then yolo26n combined
    n_paths = glob.glob("runs/**/combined_model/weights/best.pt", recursive=True)
    if n_paths:
        return n_paths[0]

    # Then any best.pt
    all_paths = glob.glob("runs/**/best.pt", recursive=True)
    if all_paths:
        return all_paths[0]

    return None


def get_best_data_yaml():
    """Returns path to best available dataset yaml."""
    options = [
        "datasets/mega_dataset/data.yaml",
        "datasets/combined_dataset/data.yaml",
        "datasets/navigation_dataset/data.yaml",
    ]
    for path in options:
        if os.path.exists(path):
            return path
    return None


# ============================================================
# WEB APP SETTINGS
# ============================================================

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_FPS = 5                      # Frames per second sent from browser to server
WEB_JPEG_QUALITY = 0.8           # JPEG compression quality
WEB_ALERT_INTERVAL = 2.0         # Speak alert every 2 seconds on web

# ============================================================
# ANDROID SETTINGS
# ============================================================

ANDROID_CAMERA_INDEX = 0
ANDROID_PREVIEW_WIDTH = 640
ANDROID_PREVIEW_HEIGHT = 480

# ============================================================
# DATASET PATHS
# ============================================================

OUTDOOR_DATASET = "datasets/navigation_dataset"
INDOOR_DATASET = "datasets/indoor_dataset"
COMBINED_DATASET = "datasets/combined_dataset"
MEGA_DATASET = "datasets/mega_dataset"
EXTRA_DATASETS_DIR = "datasets/extra"

