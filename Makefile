.PHONY: help sample label extract train predict typecheck clean format

# Virtual environment activation
VENV := .venv
PYTHON := . $(VENV)/bin/activate && python
RUFF := . $(VENV)/bin/activate && ruff
MYPY := . $(VENV)/bin/activate && mypy
PIP := . $(VENV)/bin/activate && pip

# Default parameters
MUSIC_DIR ?= /Users/maxr/iCloudDrive/bandcamp_exports
EXTENSIONS ?=
XML_PATH ?=
FEATURES_FILE ?= data/features/training_features.json

help:
	@echo "Music Classifier - Available commands:"
	@echo ""
	@echo "  make sample     - Generate random sample of 10 AIFF files for labeling"
	@echo "  make label      - Start the labeling web app"
	@echo "  make extract    - Extract features from labeled tracks"
	@echo "  make train      - Train models (FEATURES_FILE=path/to/features.json)"
	@echo "  make predict    - Run predictions on music library"
	@echo "  make typecheck  - Run mypy type checking on all Python files"
	@echo "  make format     - Format all Python files with ruff"
	@echo "  make clean      - Clean up generated files"
	@echo ""
	@echo "Note: All commands automatically activate the .venv virtual environment"

setup:
	@echo "Creating virtual environment..."
	@python -m venv $(VENV)
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete! You can now run:"
	@echo "  make sample    - Generate sample tracks"
	@echo "  make label     - Start labeling"

sample:
	@echo "Generating random sample of tracks..."
	@$(PYTHON) src/phase1/generate_sample_list.py

label:
	@echo "Starting labeling webapp..."
	@$(PYTHON) src/phase1/label_app.py & echo $$! > /tmp/label_app.pid
	@sleep 1
	@echo "Opening http://localhost:5001 in Safari..."
	@open -a Safari http://localhost:5001 || open http://localhost:5001
	@wait

extract:
	@echo "Extracting features from labeled tracks..."
	@$(PYTHON) src/phase2/feature_extraction.py

train:
	@echo "Training models with features from: $(FEATURES_FILE)"
	@$(PYTHON) src/phase2/train_models.py --features-file "$(FEATURES_FILE)"

predict:
	@echo "Running predictions..."
	@$(PYTHON) src/phase3/predict.py

typecheck:
	@echo "Running mypy type checking..."
	@$(MYPY) src/ config/ --config-file mypy.ini
	@echo "Type checking complete!"

format:
	@echo "Formatting Python files with ruff..."
	@$(RUFF) format .
	@$(RUFF) check --fix .
	@echo "Done"

clean:
	@echo "Cleaning up..."
	@rm -f data/sample_tracks.json
	@rm -f data/manual_labels.json
	@echo "Done"
