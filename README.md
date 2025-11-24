# Electronic Music Mood/Vibe Classifier for DJ Library

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Automatically tag your DJ music library with mood and energy labels using machine learning. Built specifically for electronic music, optimized for Rekordbox integration.

## Overview

This system analyzes audio files and predicts:

> **Note:** The taxonomy is fully customizable. You can easily extend or modify the energy levels and vibes by editing [config/config.py](config/config.py) to match your DJ style and music genre.

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

**Quick setup:**
```bash
git clone <repository-url>
cd music-classifier
make setup
```

This will create a virtual environment and install all dependencies.

**Manual setup (alternative):**
```bash
# 1. Clone the repository
git clone <repository-url>
cd music-classifier

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
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

Labels are saved to `data/manual_labels/manual_labels.json`. See the **Automatic Label Backups** section below for information about how your labeling work is automatically protected.

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

#### Advanced Labeling Features

**Track Position Display:**
The interface shows "Track X of Y" so you know exactly where you are in the labeling process. The counter updates automatically as you navigate through tracks.

**Previous Label Display:**
When you navigate to a track that's already been labeled, the interface displays:
- Previously saved energy level
- Previously saved vibes
- Buttons are automatically pre-selected with existing labels
- This allows you to easily update labels or skip already-labeled tracks

**Sample Batch Management:**
Each time you run `make sample`, the previous `sample_tracks.json` is automatically backed up with incremental numbering (e.g., `sample_tracks.bak.1.json`, `sample_tracks.bak.2.json`, etc.). This ensures you never lose track of any of your previous sample batches.

**Automatic Label Backups:**
Your precious labeling work is automatically protected:
- Each time you start the labeling tool (`make label`), an incremental backup is created (e.g., `manual_labels.bak.1.json`, `manual_labels.bak.2.json`, etc.)
- This protects your work if something goes wrong during a labeling session
- All previous backups are preserved, allowing you to restore from any backup
- You can restore from any backup by copying it back to `manual_labels.json`

**Exclude Patterns:**
Filter out unwanted tracks when generating samples. See the **Configuration** section below for details on customizing exclude patterns and other settings.

**Sample Batch Size:**
Adjust how many tracks to generate per batch in [config/config.py](config/config.py):

```python
SAMPLE_BATCH_SIZE = 100  # Default: 100 tracks per batch
```

### Phase 2: Training (Week 3-4)

#### Step 2.1: Extract Features
```bash
make extract
# Or: python main.py extract
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
make train
# Or: python main.py train
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
make predict MUSIC_DIR=/path/to/your/music/library
# Or: python main.py predict /path/to/your/music/library
```

Optional: Specify file extensions
```bash
make predict MUSIC_DIR=/path/to/music EXTENSIONS="*.mp3 *.flac"
# Or: python main.py predict /path/to/music --extensions "*.mp3" "*.flac"
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

### Makefile Commands (Recommended)

```bash
# Setup
make setup                                        # Create venv and install dependencies
make help                                         # Show all available commands

# Phase 1: Manual Labeling
make sample                                       # Generate random sample from music library
make label                                        # Start Flask labeling webapp (opens Safari)

# Phase 2: Training
make extract                                      # Extract features from labeled tracks
make train                                        # Train classification models

# Phase 3: Inference
make predict MUSIC_DIR=/path/to/music            # Predict labels for entire library
make predict MUSIC_DIR=/path/to/music EXTENSIONS="*.mp3 *.flac"

# Phase 4: Rekordbox Integration
make rekordbox XML_PATH=/path/to/rekordbox.xml   # Integrate with Rekordbox

# Code Quality
make format                                       # Format code with Ruff

# Cleanup
make clean                                        # Remove generated samples and labels
```

### Python CLI (Alternative)

All commands can also be run directly via Python if you prefer:

```bash
# Phase 1: Manual Labeling
python src/phase1/generate_sample_list.py
python src/phase1/label_app.py

# Phase 2: Training
python main.py extract
python main.py train

# Phase 3: Inference
python main.py predict /path/to/music
python main.py predict /path/to/music --extensions "*.aiff" "*.mp3"

# Phase 4: Rekordbox Integration
python main.py rekordbox /path/to/rekordbox.xml
python main.py rekordbox /path/to/rekordbox.xml --predictions /path/to/predictions.json
```

## Configuration

Edit [config/config.py](config/config.py) to customize your taxonomy and sampling behavior. Changes are automatically reflected in the labeling webapp.

```python
# Taxonomy
ENERGY_LABELS = ['warm-up', 'building', 'peak', 'intense', 'closing']
VIBE_LABELS = ['dark', 'melodic', 'hypnotic', 'dubby',
               'atmospheric', 'trippy', 'analog/lofi', 'clean/digital']

# Training parameters
SAMPLE_BATCH_SIZE = 100    # Number of tracks per sampling batch
TEST_SIZE = 0.2            # Train/test split ratio
N_ESTIMATORS = 200         # Number of trees in forest
MAX_DEPTH = 15             # Max tree depth

# Sampling exclusion patterns
# Case-insensitive regex patterns that match against the full file path
# When you run `make sample`, files matching these patterns are excluded
EXCLUDE_PATTERNS = [
    r"docetism",      # Exclude specific artists/albums
    # r"remix",       # Uncomment to exclude remixes
    # r"live",        # Uncomment to exclude live recordings
    # r"radio edit",  # Uncomment to exclude radio edits
]
# Examples:
#   r"artist_name"     - Exclude all tracks by specific artist
#   r"\\(remix\\)"     - Exclude all remixes (note: escape special regex chars)
#   r"live|bootleg"    - Exclude live recordings OR bootlegs

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
- [x] Track position indicator (X of Y)
- [x] Display previously saved labels
- [x] Automatic sample batch backups
- [x] Regex-based file exclusion patterns
- [x] Configurable sample batch size
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
