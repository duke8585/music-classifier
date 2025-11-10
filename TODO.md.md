# TODO

## Completed ✅

- ✅ Labeler: AIFF files from `/Users/maxr/iCloudDrive/bandcamp_exports/**/*.aiff`
  - Implemented as Flask web app with Safari support
  - Random sample generation via `make sample`
  - Features: Audio playback, speed control, keyboard shortcuts, autoplay

- ✅ Makefile commands for workflow steps
  - `make sample` - Generate random sample
  - `make label` - Start labeling webapp (auto-opens Safari)
  - `make clean` - Cleanup generated files

- ✅ Removed all MTG-Jamendo references
  - Deleted download_metadata.py
  - Removed from config.py, main.py, Makefile
  - Workflow uses only user's own music library

- ✅ General cleanup
  - Removed old tkinter labeling tool
  - Updated .gitignore for AIFF and sample_tracks.json
  - Fixed all labeling bugs
  - Successfully labeled 10 sample tracks

## Next Steps

### Phase 2: Feature Extraction & Training
- Implement feature extraction with Essentia/librosa (AIFF compatible)
- Train energy classifier (5 classes)
- Train vibe classifier (8 classes, multi-label)
- Target: F1 > 0.70 (energy), F1 > 0.60 (vibe)

### Phase 3: Inference
- Batch process ~6,000 AIFF files
- Generate predictions with confidence scores

### Phase 4: Rekordbox Integration
- Parse Rekordbox XML export
- Write predictions to MyTag fields
- Test import workflow
