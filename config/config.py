"""
Configuration file for the Music Classifier project.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MANUAL_LABELS_DIR = os.path.join(DATA_DIR, "manual_labels")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")

# Taxonomy
ENERGY_LABELS = ["warm-up", "building", "peak", "intense", "closing"]
VIBE_LABELS = [
    "deep",
    "dark",
    "melodic",
    "hypnotic",
    "atmospheric",
    "trippy",
    "breaks",
    "dubby",
    "analog/lofi",
    "clean/digital",
    "IGNORE THIS",
]

# Training parameters
SAMPLE_BATCH_SIZE = 20  # Number of tracks to sample per batch for labeling
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_F1_ENERGY = 0.70
MIN_F1_VIBE = 0.60

# Sampling exclusion patterns (regex patterns to exclude files/paths)
# Examples: r"remix", r"live", r"radio edit", r"demo", r"instrumental"
EXCLUDE_PATTERNS = [
    r"docetism",
    # Add your exclusion patterns here
    # r"remix",
    # r"live",
    # r"radio edit",
]

# Feature extraction
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024

# Model parameters
N_ESTIMATORS = 200
MAX_DEPTH = 15

# Prediction display settings
PREDICTION_VIBE_THRESHOLD = 0.6  # Show vibe as predicted if confidence > 60%
PREDICTION_AUTO_ACCEPT_ENERGY = 0.8  # Auto-accept energy if confidence > 80%
PREDICTION_AUTO_ACCEPT_VIBE = 0.75  # Auto-accept vibe if confidence > 75%
