# Training Script Debug Guide

## Common Issues and Solutions

### 1. Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'essentia'`

**Solution**:
```bash
pip install -r requirements.txt
```

**Note**: Essentia can be tricky to install. If pip fails, try:
```bash
# On Ubuntu/Debian:
apt-get install python3-essentia

# Or use conda:
conda install -c mtg essentia
```

---

### 2. Labels File Not Found
**Error**: `Labels file not found: data/manual_labels.json`

**Solution**: Create the labels file in the correct format:
```json
{
  "track001": {
    "energy": "peak",
    "vibe": ["dark", "hypnotic"],
    "audio_path": "/path/to/track001.mp3"
  },
  "track002": {
    "energy": "building",
    "vibe": ["melodic", "atmospheric"],
    "audio_path": "/path/to/track002.mp3"
  }
}
```

**Valid Labels**:
- Energy: `warm-up`, `building`, `peak`, `intense`, `closing`
- Vibe: `dark`, `melodic`, `hypnotic`, `dubby`, `atmospheric`, `trippy`, `analog/lofi`, `clean/digital`

---

### 3. Audio Files Not Found
**Error**: `audio file not found at /path/to/track.mp3`

**Causes**:
- Incorrect file paths in `manual_labels.json`
- Audio files moved/deleted
- Relative vs absolute path issues

**Solution**:
- Use absolute paths in `manual_labels.json`
- Or use paths relative to where you run the script
- Verify files exist: `ls -l /path/to/your/tracks/`

---

### 4. Insufficient Training Data
**Error**: `Insufficient training data! Need at least 50 valid labeled tracks`

**Solution**:
- Label more tracks (minimum 50, recommended 200-300)
- Use the labeling tool from Phase 1
- Consider bootstrapping with MTG-Jamendo dataset

---

### 5. Class Imbalance Warning
**Warning**: `Energy class imbalance: minimum 5 samples`

**Impact**: Model will perform poorly on underrepresented classes

**Solution**:
- Label more tracks for underrepresented classes
- Aim for at least 10-20 samples per class
- For best results: balanced distribution

---

### 6. Feature Extraction Failures
**Error**: `Failed to extract features from track_id: ...`

**Causes**:
- Corrupted audio files
- Unsupported audio format
- Empty/silent audio files
- Too short audio files

**Solutions**:
- Check audio file integrity: `ffmpeg -i file.mp3 -f null -`
- Convert to supported format: `ffmpeg -i input.flac -ar 44100 output.mp3`
- Remove corrupted tracks from labels file

---

### 7. NaN/Inf Values in Features
**Warning**: `NaN values detected in features`

**Cause**: Audio files with issues (silence, DC offset, etc.)

**Solution**: Script automatically handles this with `np.nan_to_num()`
But check your audio files for quality issues.

---

### 8. Low F1 Scores
**Result**: `F1 score 0.45 below target 0.70`

**Solutions**:
1. **More Data**: Label 200-300 tracks minimum
2. **Balanced Classes**: Ensure each class has similar sample counts
3. **Better Labels**: Review and fix ambiguous/incorrect labels
4. **Feature Engineering**:
   - Add more audio features
   - Use deep learning features (VGGish, OpenL3)
5. **Hyperparameter Tuning**:
   - Adjust RandomForest parameters
   - Try XGBoost instead

---

### 9. Memory Issues
**Error**: `MemoryError` or system crashes

**Causes**:
- Too many tracks processed at once
- Large audio files
- Insufficient RAM

**Solutions**:
```python
# Process in batches
# Reduce frame processing
# Use smaller frame_size in feature extraction
```

---

### 10. Audio Format Issues
**Error**: `Could not decode audio`

**Supported Formats**: MP3, WAV, FLAC, OGG, M4A

**Solution**:
```bash
# Convert problematic files
for f in *.flac; do
  ffmpeg -i "$f" -ar 44100 -ac 1 "${f%.flac}.mp3"
done
```

---

## Running the Script

### Basic Usage
```bash
python train.py
```

### Custom Paths
```bash
python train.py --data-dir /path/to/data --models-dir /path/to/models
```

### Different Labels File
```bash
python train.py --labels-file my_labels.json
```

---

## What the Script Does

1. **Environment Validation**: Checks all dependencies are installed
2. **Label Loading**: Loads and validates `data/manual_labels.json`
3. **Label Validation**:
   - Checks for required fields
   - Validates label values
   - Verifies audio files exist
   - Reports class distribution
4. **Feature Extraction**:
   - Loads audio files
   - Extracts spectral features (centroid, rolloff)
   - Extracts timbral features (MFCCs)
   - Extracts rhythm features (BPM, beats)
5. **Model Training**:
   - Trains separate Random Forest models for Energy and Vibe
   - 80/20 train/test split
   - 5-fold cross-validation
   - Generates detailed metrics
6. **Model Evaluation**:
   - Accuracy scores
   - F1 scores (macro and weighted)
   - Classification reports
   - Confusion matrices
7. **Model Saving**: Saves trained models to `models/` directory

---

## Output Files

After successful training:
- `models/energy_classifier.joblib` - Energy classification model
- `models/vibe_classifier.joblib` - Vibe classification model

---

## Next Steps After Training

1. Review classification reports for each model
2. Check if F1 scores meet targets:
   - Energy: F1 > 0.70
   - Vibe: F1 > 0.60
3. If scores are low, refer to "Low F1 Scores" section above
4. Once satisfied, proceed to Phase 3 (Inference)

---

## Quick Debugging Checklist

- [ ] All dependencies installed? (`pip list | grep essentia`)
- [ ] Labels file exists? (`ls data/manual_labels.json`)
- [ ] Valid JSON format? (`python -m json.tool data/manual_labels.json`)
- [ ] At least 50 labeled tracks?
- [ ] Audio files exist and are readable?
- [ ] Audio files in supported format (MP3, WAV, FLAC)?
- [ ] At least 10 samples per class?
- [ ] Sufficient disk space for models? (`df -h`)
- [ ] Sufficient RAM? (`free -h`)

---

## Getting Help

If you encounter issues not covered here:
1. Check the detailed log output from `train.py`
2. Look for specific error messages
3. Verify your data format matches the examples
4. Try with a small subset of data first (10-20 tracks)
