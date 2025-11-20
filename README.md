# Electronic Music Mood/Vibe Classifier

ML-powered system to automatically tag electronic music tracks with mood and energy labels for DJ library organization.

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create your labels file (copy from example)
cp data/manual_labels.example.json data/manual_labels.json

# Edit with your tracks and labels
nano data/manual_labels.json
```

### 2. Validate Your Labels
```bash
# Run validation before training
python validate_labels.py

# Should output: ✓ READY TO TRAIN!
```

### 3. Train Models
```bash
# Train classifiers
python train.py

# Models will be saved to models/
```

## Labels Format

Create `data/manual_labels.json`:
```json
{
  "track_id": {
    "energy": "peak",
    "vibe": ["dark", "hypnotic"],
    "audio_path": "/absolute/path/to/track.mp3"
  }
}
```

**Valid Labels:**
- **Energy** (5): `warm-up`, `building`, `peak`, `intense`, `closing`
- **Vibe** (8): `dark`, `melodic`, `hypnotic`, `dubby`, `atmospheric`, `trippy`, `analog/lofi`, `clean/digital`

## Minimum Requirements

- **50+ labeled tracks** (200-300 recommended)
- **10+ samples per class** (for balanced training)
- Audio files in MP3, WAV, or FLAC format

## Training Not Working?

**See [TRAINING_DEBUG.md](TRAINING_DEBUG.md) for detailed troubleshooting guide**

Common issues:
1. Labels file missing or malformed → Run `validate_labels.py`
2. Audio files not found → Use absolute paths
3. Missing dependencies → `pip install -r requirements.txt`
4. Insufficient data → Label more tracks
5. Low F1 scores → More data + balanced classes

## What Gets Trained

The script trains **2 separate models**:
1. **Energy Classifier**: Predicts warm-up, building, peak, intense, or closing
2. **Vibe Classifier**: Predicts dark, melodic, hypnotic, etc.

Both use Random Forest with extensive validation and cross-validation.

## Success Criteria

- Energy model: **F1 > 0.70**
- Vibe model: **F1 > 0.60**

## Project Phases

- [x] **Phase 1**: Labeling Tool (Week 1-2)
- [x] **Phase 2**: Training (Week 3-4) ← YOU ARE HERE
- [ ] **Phase 3**: Inference (Week 5)
- [ ] **Phase 4**: Rekordbox Integration (Week 6)

## Files

- `train.py` - Main training script
- `validate_labels.py` - Pre-training validation
- `requirements.txt` - Python dependencies
- `TRAINING_DEBUG.md` - Comprehensive troubleshooting guide
- `data/manual_labels.example.json` - Labels file template
- `claude.md` - Original project specification

## Tech Stack

- **Essentia**: Feature extraction (optimized for electronic music)
- **scikit-learn**: Random Forest classification
- **librosa**: Audio processing utilities

## Getting Help

1. Run `python validate_labels.py` - catches 90% of issues
2. Check `TRAINING_DEBUG.md` - detailed solutions for common problems
3. Review training output - detailed logs show exactly what went wrong

## License

This is a personal project for organizing a DJ library.
