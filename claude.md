# Electronic Music Mood/Vibe Classifier for DJ Library

## Project Overview
Build an ML-powered system to automatically tag ~6,000 electronic music tracks with mood and energy labels.

**Taxonomy:**
- Energy (5): warm-up, building, peak, intense, closing
- Vibe (8): dark, melodic, hypnotic, dubby, atmospheric, trippy, analog/lofi, clean/digital

## 4 Phases (6 weeks)

### Phase 1: Labeling Tool (Week 1-2)
- Download MTG-Jamendo metadata (~16k electronic tracks)
- Build GUI labeling tool (Tkinter or web-based)
- User labels 200-300 tracks manually
- Output: data/manual_labels.json

### Phase 2: Training (Week 3-4)
- Extract features with Essentia (electronic music optimized)
- Train 2 models: Energy classifier + Vibe classifier
- Start with Random Forest/XGBoost
- Bootstrap with MTG-Jamendo if needed
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
- Essentia: Feature extraction
- scikit-learn/XGBoost: Classification
- librosa: Audio processing
- Tkinter or Flask: Labeling UI

## Success Criteria
User can search "dark, hypnotic peak-time tracks" in Rekordbox and find relevant results immediately.
