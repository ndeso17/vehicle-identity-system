import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models Directory
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# Model Paths
MODEL_PLAT_NOMOR_PRIMARY_PATH = os.path.join(MODELS_DIR, 'tnkb.pt')
MODEL_PLAT_NOMOR_FALLBACK_PATH = os.path.join(MODELS_DIR, 'plat_nomor.pt')
MODEL_OCR_CHAR_PATH = os.path.join(MODELS_DIR, 'character_ocr.pt')
MODEL_YOLO_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')

# Detection Settings
CONF_THRESHOLD_PLAT = 0.1   # Plate-first low threshold per spec
CONF_THRESHOLD_VEHICLE = 0.4
CONF_THRESHOLD_PERSON = 0.4

# Classes for YOLOv8 (COCO)
TARGET_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
TARGET_PERSON_CLASS = 0

# Color Detection Settings
HSV_SATURATION_THRESHOLD = 50
HSV_VALUE_THRESHOLD = 50

# Visualization Settings (BGR format for OpenCV)
COLOR_PLATE_BOX = (0, 165, 255)     # ORANGE (BGR)
COLOR_PLATE_BOX_ORIGINAL = (0, 0, 255)  # RED for original YOLO bbox
COLOR_VEHICLE_BOX = (0, 255, 0)     # GREEN for vehicles
COLOR_DRIVER_BOX = (255, 0, 255)    # MAGENTA for drivers
COLOR_CABIN_DEBUG = (255, 255, 0)   # CYAN-like for cabin debug overlays
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Device
DEVICE = 'cpu'

# ========================================
# IDENTITY MATCHING CONFIGURATION
# ========================================

# Priority 1: Plate OCR
PLATE_PRIMARY_CONF = 0.7    # Confidence above which plate is trusted as primary key
PLATE_FALLBACK_CONF = 0.3   # Below this, rely on visual features only

# Priority 2: Face Embedding
FACE_SIM_THRESHOLD = 0.65   # Cosine similarity threshold for face match
FACE_EMBEDDING_METHOD = 'auto'  # 'face_recognition', 'appearance_hash', or 'auto'

# Priority 3: Visual Features (type + color + size)
VISUAL_MATCH_THRESHOLD = 0.8  # Threshold for visual feature matching

# Clustering/Matching
CLUSTER_MATCH_THRESHOLD = 0.5  # Min weighted score to reuse existing cluster

# Identity Priority Weights (for weighted scoring when multiple features available)
WEIGHT_PLATE = 3.0      # Highest weight for plate match
WEIGHT_FACE = 2.0       # Strong signal for driver identity
WEIGHT_TYPE = 0.5       # Weak signal (many cars/motorcycles)
WEIGHT_COLOR = 0.5      # Weak signal (colors vary by lighting)
WEIGHT_TIME = 0.5       # Temporal proximity bonus

# Time window for temporal matching (seconds)
TIME_WINDOW_STRONG = 600    # 10 mins - high time score
TIME_WINDOW_WEAK = 3600     # 1 hour - low time score

# ========================================
# STORAGE PATHS
# ========================================

STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
CROPS_FOLDER = os.path.join(STATIC_FOLDER, 'crops')
FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'frames')
ANNOTATED_FOLDER = os.path.join(STATIC_FOLDER, 'annotated')

# Legacy paths (for backward compatibility)
REGISTRY_PATH = os.path.join(BASE_DIR, 'results', 'vehicle_registry.json')
RAW_DETECTIONS_PATH = os.path.join(BASE_DIR, 'results', 'raw_detections.jsonl')
CLUSTERING_DECISIONS_PATH = os.path.join(BASE_DIR, 'results', 'clustering_decisions.jsonl')
MANUAL_CORRECTIONS_PATH = os.path.join(BASE_DIR, 'results', 'manual_corrections.jsonl')

# ========================================
# UI SETTINGS
# ========================================

VEHICLE_UI_PER_PAGE = 24  # Vehicles per page in gallery view
OBSERVATIONS_PER_PAGE = 50  # Observations per page in table view

# ========================================
# DATABASE
# ========================================

DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'vehicle_identity.db')}"

# ========================================
# ENSURE DIRECTORIES EXIST
# ========================================

def ensure_directories():
    """Create required directories if they don't exist"""
    dirs = [
        os.path.join(BASE_DIR, 'static'),
        CROPS_FOLDER,
        FRAMES_FOLDER,
        ANNOTATED_FOLDER,
        os.path.join(BASE_DIR, 'results'),
        os.path.join(BASE_DIR, 'instance'),
        os.path.join(BASE_DIR, 'Views'),
        os.path.join(BASE_DIR, 'Views', 'admin'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Run on import
ensure_directories()
