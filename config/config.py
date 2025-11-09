"""
Configuration file for the Music Classifier project.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MANUAL_LABELS_DIR = os.path.join(DATA_DIR, 'manual_labels')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

# Taxonomy
ENERGY_LABELS = ['warm-up', 'building', 'peak', 'intense', 'closing']
VIBE_LABELS = ['dark', 'melodic', 'hypnotic', 'dubby', 'atmospheric', 'trippy', 'analog/lofi', 'clean/digital']

# MTG-Jamendo Dataset
MTG_JAMENDO_API_BASE = "https://mtg.github.io/mtg-jamendo-dataset"
MTG_JAMENDO_METADATA_URL = "https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging.tsv"

# Training parameters
MANUAL_LABEL_TARGET = 300  # Target number of manually labeled tracks
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_F1_ENERGY = 0.70
MIN_F1_VIBE = 0.60

# Feature extraction
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024

# Model parameters
N_ESTIMATORS = 200
MAX_DEPTH = 15
