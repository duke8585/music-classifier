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

# Taxonomy - RELABEL v2.0 (4 orthogonal dimensions)
# Each dimension is single-select
ENERGY_LABELS = ["intro/outro", "mid", "peak"]
BASS_WEIGHT_LABELS = ["light", "balanced", "heavy"]
RHYTHM_LABELS = ["straight", "breaks", "sparse"]
VIBE_LABELS = ["hypnotic", "melodic", "atmospheric"]

# Legacy taxonomy (for backwards compatibility with old labels)
LEGACY_ENERGY_LABELS = ["warm-up", "building", "peak", "intense", "closing"]
LEGACY_VIBE_LABELS = [
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
]

# Training parameters
SAMPLE_BATCH_SIZE = 300  # Number of tracks to sample per batch for labeling
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model performance targets for RELABEL v2.0
MIN_F1_ENERGY = 0.70  # 3-class: intro/outro, mid, peak
MIN_F1_BASS_WEIGHT = 0.65  # 3-class: light, balanced, heavy
MIN_F1_RHYTHM = 0.75  # 3-class: straight, breaks, sparse
MIN_F1_VIBE = 0.60  # 3-class: hypnotic, melodic, atmospheric

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

# Prediction display settings (for future re-implementation)
# PREDICTION_VIBE_THRESHOLD = 0.6  # Show vibe as predicted if confidence > 60%
# PREDICTION_AUTO_ACCEPT_ENERGY = 0.8  # Auto-accept energy if confidence > 80%
# PREDICTION_AUTO_ACCEPT_VIBE = 0.75  # Auto-accept vibe if confidence > 75%
