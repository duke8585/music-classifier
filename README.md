# Electronic Music Mood/Vibe Classifier for DJ Library

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Automatically tag your DJ music library with mood and energy labels using machine learning. Built specifically for electronic music, optimized for Rekordbox integration.

## Overview

This system analyzes audio files and predicts:

**Energy Levels (5 categories):**
- warm-up
- building
- peak
- intense
- closing

**Vibes (8 categories):**
- dark
- melodic
- hypnotic
- dubby
- atmospheric
- trippy
- analog/lofi
- clean/digital

**Target:** Automatically classify ~6,000 tracks after training on 200-300 manually labeled examples.

## Project Structure

```
music-classifier/
├── config/
│   └── config.py              # Configuration settings
├── data/
│   ├── manual_labels/         # Manual labels and metadata
│   ├── features/              # Extracted audio features
│   ├── models/                # Trained ML models
│   └── predictions/           # Inference results
├── src/
│   ├── phase1/                # Phase 1: Manual labeling
│   │   ├── label_app.py       # Flask web app for labeling
│   │   ├── generate_sample_list.py  # Random sample generator
│   │   └── convert_sample_to_wav.py  # Optional AIFF→WAV converter
│   ├── phase2/                # Training pipeline
│   ├── phase3/                # Inference system
│   └── phase4/                # Rekordbox integration
├── templates/
│   └── label.html             # Labeling webapp UI
├── tests/                     # Unit tests
├── Makefile                   # Build commands
├── main.py                    # CLI entry point
└── requirements.txt           # Dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Audio files in supported formats (AIFF, MP3, WAV, FLAC, M4A)
- Safari browser (for AIFF playback in labeling tool)
- (Optional) Rekordbox for DJ library integration

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd music-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note on Essentia:** Essentia is the preferred library for electronic music feature extraction. If installation fails, the system will fall back to librosa.

```bash
# For Essentia (optional but recommended):
pip install essentia
```

## Quick Start Guide

### Phase 1: Manual Labeling (Week 1-2)

#### Step 1: Generate Sample Tracks
```bash
make sample
```

This randomly selects 10 AIFF files from your music library for labeling. Edit the path in [src/phase1/generate_sample_list.py](src/phase1/generate_sample_list.py) to point to your music directory.

Default: `/Users/maxr/iCloudDrive/bandcamp_exports/**/*.aiff`

#### Step 2: Start Labeling
```bash
make label
```

This launches a Flask web app at http://localhost:5001 and opens it in Safari. The interface includes:

**Note:** Safari is required for AIFF playback. If you prefer Chrome/Firefox, first convert samples to WAV:
```bash
python src/phase1/convert_sample_to_wav.py
```
- Audio player with speed control (±10% playback rate)
- Energy level buttons (keyboard shortcuts: 1-5)
- Vibe buttons with multi-select (keyboard shortcuts: QWERTASD)
- Autoplay on track navigation
- Progress tracking

Labels are saved to `data/manual_labels.json`

**Target:** Label 200-300 tracks for good model performance.

**Keyboard Shortcuts:**
- `1-6`: Select energy level
- `asdfghjkl` (home row): Toggle vibes 1-10
- `xcvbnm` (bottom row): Toggle remaining vibes
- `Enter`: Save and advance to next track
- `Arrow keys`: Navigate between tracks
- `Space`: Play/pause

**Tips:**
- Focus on diverse examples across all energy levels
- Label tracks you know well from your DJ sets
- Use speed control to quickly assess energy level
- Vibes are multi-select - choose all that apply

### Phase 2: Training (Week 3-4)

#### Step 2.1: Extract Features
```bash
python main.py extract
```

Extracts audio features from labeled tracks using Essentia or librosa:
- Rhythm features (BPM, beat strength)
- Spectral features (brightness, texture)
- Tonal features (key, harmony)
- Timbral features (MFCCs)
- Dynamic features (energy, loudness)

Output: `data/features/training_features.json`

#### Step 2.2: Train Models
```bash
python main.py train
```

Trains two classifiers:
1. **Energy Classifier:** Multi-class classification (5 classes)
2. **Vibe Classifier:** Multi-label classification (8 classes)

Uses XGBoost (if available) or Random Forest with:
- 80/20 train/test split
- Cross-validation
- Feature importance analysis
- Performance metrics (F1 scores, accuracy)

**Success Criteria:**
- Energy classifier: F1 > 0.70
- Vibe classifier: F1 > 0.60

Output: Trained models in `data/models/`

### Phase 3: Inference (Week 5)

#### Batch Process Your Music Library
```bash
python main.py predict /path/to/your/music/library
```

Optional: Specify file extensions
```bash
python main.py predict /path/to/music --extensions "*.mp3" "*.flac"
```

This will:
- Find all audio files in the directory (recursive)
- Extract features for each track
- Predict energy and vibe labels with confidence scores
- Generate output files:
  - `data/predictions/predictions.json` (detailed)
  - `data/predictions/predictions.csv` (for review)

**Example output:**
```json
{
  "filename": "track001.mp3",
  "energy": "peak",
  "energy_confidence": 0.87,
  "vibes": ["dark", "hypnotic"],
  "energy_top3": [
    {"label": "peak", "confidence": 0.87},
    {"label": "intense", "confidence": 0.09},
    {"label": "building", "confidence": 0.03}
  ]
}
```

### Phase 4: Rekordbox Integration (Week 6)

#### Step 4.1: Export Rekordbox Library
1. Open Rekordbox
2. File → Library → Export Collection in xml format
3. Save as `rekordbox.xml`

#### Step 4.2: Integrate Predictions
```bash
python main.py rekordbox /path/to/rekordbox.xml
```

This will:
- Parse your Rekordbox library
- Match predictions to tracks by file path
- Write tags to MyTag/Comments fields
- Create a backup of the original XML
- Generate a new `rekordbox_tagged.xml`

**Tag Format Options:**

Default format: `"peak | dark, hypnotic"`

You can customize by editing `src/phase4/rekordbox_integration.py`:
- `energy_vibes`: "peak | dark, hypnotic" (default)
- `energy_only`: "peak"
- `vibes_only`: "dark, hypnotic"
- `detailed`: "E:peak(0.87) | V:dark,hypnotic"

#### Step 4.3: Import to Rekordbox
1. **Backup your Rekordbox library first!**
   - File → Library → Backup Library

2. **Option A: Replace XML** (Recommended)
   - Close Rekordbox completely
   - Replace `rekordbox.xml` with `rekordbox_tagged.xml`
   - Restart Rekordbox

3. **Option B: Import Collection**
   - File → Library → Import Collection
   - Select `rekordbox_tagged.xml`

4. **Verify Tags**
   - Check a few tracks to ensure tags appear in Comments field
   - Enable Comments column: File → Preferences → View

## Using the Tags in DJ Sets

### Rekordbox Search Examples

```
Search: "peak"
→ Find all peak-time tracks

Search: "dark, hypnotic"
→ Find dark hypnotic tracks

Search: "closing"
→ Find closing tracks

Search: "peak" + BPM filter 128-132
→ Find peak-time techno
```

### Advanced Workflow

1. **Set Preparation:**
   - Filter by energy level for different set sections
   - Combine with BPM and key for harmonic mixing
   - Use vibe tags to maintain mood consistency

2. **Track Discovery:**
   - Search for "melodic, atmospheric" for deep house sets
   - Find "dark, intense" for techno peak times
   - Discover "dubby, hypnotic" for minimal sets

## CLI Reference

```bash
# Phase 1: Manual Labeling
make sample                      # Generate random sample from your music library
make label                       # Start Flask labeling webapp (opens Safari)

# Phase 2: Training
python main.py extract           # Extract features from labeled tracks
python main.py train             # Train classification models

# Phase 3: Inference
python main.py predict <music_dir>               # Predict labels for library
python main.py predict <music_dir> --extensions "*.aiff" "*.mp3"

# Phase 4: Rekordbox Integration
python main.py rekordbox <xml_path>              # Integrate with Rekordbox
python main.py rekordbox <xml_path> --predictions <json_path>

# Cleanup
make clean                       # Remove generated samples and labels
```

## Configuration

Edit [config/config.py](config/config.py) to customize your taxonomy. Changes are automatically reflected in the labeling webapp.

```python
# Taxonomy
ENERGY_LABELS = ['warm-up', 'building', 'peak', 'intense', 'closing']
VIBE_LABELS = ['dark', 'melodic', 'hypnotic', 'dubby',
               'atmospheric', 'trippy', 'analog/lofi', 'clean/digital']

# Training parameters
MANUAL_LABEL_TARGET = 300  # Target number of labeled tracks
TEST_SIZE = 0.2            # Train/test split ratio
N_ESTIMATORS = 200         # Number of trees in forest
MAX_DEPTH = 15             # Max tree depth

# Feature extraction
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024
```

## Improving Model Performance

### 1. Label More Tracks
- Aim for 300+ labeled tracks
- Ensure balanced representation across all energy levels
- Include diverse examples for each vibe

### 2. Label Quality
- Be consistent with your labeling criteria
- Review and correct existing labels
- Use tracks you know well

### 3. Feature Engineering
- Edit `src/phase2/feature_extraction.py`
- Add genre-specific features
- Experiment with different audio descriptors

### 4. Model Tuning
- Edit `src/phase2/train_models.py`
- Adjust hyperparameters (n_estimators, max_depth)
- Try different algorithms (SVM, Neural Networks)
- Use ensemble methods

### 5. Use Pre-labeled Data (Optional)
- If you have existing tagged music from other sources
- Map their tags to your taxonomy
- Pre-train on larger dataset, fine-tune on your labels

## Troubleshooting

### Issue: Essentia installation fails
**Solution:** Use librosa as fallback. The system will automatically detect and use librosa.

### Issue: Feature extraction is slow
**Solution:**
- Process in smaller batches
- Use multiprocessing (edit `inference.py`)
- Extract features from representative segments instead of full tracks

### Issue: Low model accuracy
**Solution:**
- Label more tracks (aim for 300+)
- Check label consistency
- Review misclassified examples
- Add more informative features

### Issue: Tracks don't match in Rekordbox
**Solution:**
- Ensure file paths are consistent
- Check that predictions used same files as Rekordbox library
- Use absolute paths in predictions

### Issue: Tags don't appear in Rekordbox
**Solution:**
- Enable Comments column in Rekordbox
- Try re-importing the library
- Check Rekordbox version (format may vary)

## Tech Stack

- **Flask:** Web-based labeling tool
- **Essentia:** Audio feature extraction (electronic music optimized, AIFF compatible)
- **librosa:** Alternative audio processing (AIFF compatible)
- **scikit-learn:** Machine learning algorithms
- **XGBoost:** Gradient boosting (if available)
- **pandas:** Data manipulation
- **lxml:** XML processing for Rekordbox

## Performance Benchmarks

Expected performance with 300 labeled tracks:

| Metric | Energy Classifier | Vibe Classifier |
|--------|------------------|-----------------|
| F1 Score (macro) | 0.70-0.80 | 0.60-0.70 |
| Accuracy | 0.75-0.85 | 0.55-0.65 |
| Inference Speed | ~2-3 sec/track | ~2-3 sec/track |

## Roadmap

- [x] Web-based labeling interface with Flask
- [x] Audio playback with speed control
- [x] Keyboard shortcuts for efficient labeling
- [ ] Implement active learning for smart sample selection
- [ ] Add support for Traktor and Serato
- [ ] Create pre-trained models for common genres
- [ ] Implement confidence-based filtering
- [ ] Add multi-genre support
- [ ] Optional WAV conversion for broader browser support

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Essentia team for audio analysis tools optimized for electronic music
- librosa team for AIFF-compatible audio processing
- Electronic music DJ community for feature inspiration

## Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review phase-specific documentation in code comments

---

**Built for DJs, by DJs. Happy mixing! 🎧**
