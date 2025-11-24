# Electronic Music Mood/Vibe Classifier for DJ Library

## Project Overview
Build an ML-powered system to automatically tag ~6,000 electronic music tracks with mood and energy labels.

**Taxonomy (RELABEL v2.0):**
- Energy (3): intro/outro, mid, peak
- Bass Weight (3): light, balanced, heavy
- Rhythm (3): straight, breaks, sparse
- Vibe (3): hypnotic, melodic, atmospheric

**Total: 81 combinations** (3^4) - optimized for DJ workflow and ML training

See [LABELING.md](LABELING.md) for complete taxonomy documentation.

## 4 Phases (6 weeks)

### Phase 1: Manual Labeling (Week 1-2)
- **Implemented**: Flask-based web labeling tool at http://localhost:5001 (RELABEL v2.0)
- Generate random sample: `make sample` (creates data/sample_tracks.json)
- Start labeling: `make label` (auto-opens Safari)
- Features: Radio button UI, audio playback with speed control, keyboard shortcuts (1-3/QWE/ASD/YXC + J/L seek), autoplay
- User labels 200-300 tracks from personal library (~6,000 AIFF files in iCloud)
- Output: data/manual_labels.json (format: energy, bass_weight, rhythm, vibe)

### Phase 2: Training (Week 3-4)
- Extract features with Essentia (electronic music optimized)
- Train 4 models: Energy, Bass Weight, Rhythm, and Vibe classifiers
- Start with Random Forest/XGBoost
- Target: F1 > 0.70 (energy), F1 > 0.65 (bass), F1 > 0.75 (rhythm), F1 > 0.60 (vibe)

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
User can search "atmospheric, heavy bass mid-energy tracks" or "hypnotic peak-time straight 4/4" in Rekordbox and find relevant results immediately.

## Taxonomy Design Philosophy
The RELABEL v2.0 taxonomy uses 4 orthogonal dimensions:
1. **Energy**: Set positioning (when to play in a DJ set)
2. **Bass Weight**: Low-frequency presence (measurable from 20-80Hz energy)
3. **Rhythm**: Drum pattern characteristics (onset regularity, percussion density)
4. **Vibe**: Textural and spatial qualities (self-similarity, reverb, harmonic content)

This design prioritizes:
- **Orthogonality**: Independent dimensions reduce labeling ambiguity
- **Measurability**: All dimensions correlate with audio features for ML training
- **DJ Workflow**: Tags map to practical DJ selection criteria
- **Collection Fit**: Optimized for dub techno, deep house, atmospheric, hypnotic, downtempo
