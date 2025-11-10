# Electronic Music Mood/Vibe Classifier for DJ Library

## Project Overview
Build an ML-powered system to automatically tag ~6,000 electronic music tracks with mood and energy labels.

**Taxonomy:**
- Energy (5): warm-up, building, peak, intense, closing
- Vibe (8): dark, melodic, hypnotic, dubby, atmospheric, trippy, analog/lofi, clean/digital

## 4 Phases (6 weeks)

### Phase 1: Manual Labeling (Week 1-2)
- **Implemented**: Flask-based web labeling tool at http://localhost:5001
- Generate random sample: `make sample` (creates data/sample_tracks.json)
- Start labeling: `make label` (auto-opens Safari)
- Features: Audio playback with speed control, keyboard shortcuts, autoplay
- User labels 200-300 tracks from personal library (~6,000 AIFF files in iCloud)
- Output: data/manual_labels.json

### Phase 2: Training (Week 3-4)
- Extract features with Essentia (electronic music optimized)
- Train 2 models: Energy classifier + Vibe classifier
- Start with Random Forest/XGBoost
- Target: F1 > 0.70 (energy), F1 > 0.60 (vibe)

### Phase 3: Inference (Week 5)
- Batch process 6,000 tracks
- Output predictions.json with confidence scores
- CSV for easy review/filtering

### Phase 4: Rekordbox Integration (Week 6)
- Parse Rekordbox XML export
- Write tags to MyTag fields
- Reimport to Rekordbox

## Tech Stack
- **Flask**: Web-based labeling UI (Safari for AIFF support)
- **Essentia**: Feature extraction (electronic music optimized)
- **scikit-learn/XGBoost**: Classification
- **librosa**: Audio processing (AIFF compatible)
- **Ruff**: Code formatting and linting

## Development Workflow
- **Virtual Environment**: All make commands automatically activate `.venv` before running Python/Ruff
- **Always format code at the end of each implementation cycle**: Run `make format` to ensure consistent code style
- Ruff configuration is in `ruff.toml` (100 char line length, Python 3.10+)
- Format command: `make format` (runs `ruff format .` and `ruff check --fix .`)
- All Python commands in Makefile source `.venv/bin/activate` first

## Success Criteria
User can search "dark, hypnotic peak-time tracks" in Rekordbox and find relevant results immediately.
