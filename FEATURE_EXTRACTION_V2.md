# Music Classifier - Future Improvements

## Option B: Deep Learning Embeddings

### Context
Current approach uses hand-crafted audio features (MFCCs, spectral features) which struggle to capture subjective/perceptual concepts like "hypnotic" or "dubby". Deep learning models pre-trained on music can extract high-level semantic embeddings that better represent these concepts.

### Approach
Use pre-trained music embedding models to convert audio → feature vectors:

1. **OpenL3** - General-purpose audio embeddings
   - Pre-trained on AudioSet
   - 512-dimensional embeddings
   - Captures high-level audio patterns
   - Python library: `pip install openl3`

2. **VGGish** - Google's audio embedding model
   - Pre-trained on YouTube-8M
   - 128-dimensional embeddings
   - Already has Essentia integration: `TensorflowInputVGGish`

3. **MusiCNN** - Music-specific CNN
   - Trained specifically on music
   - Essentia has: `TensorflowInputMusiCNN`
   - Best for music genre/mood

### Implementation Steps
1. Install TensorFlow + model libraries
2. Extract embeddings for all 314 tracks (or use Essentia's built-in extractors)
3. Use embeddings as features instead of/in addition to MFCCs
4. Retrain XGBoost on embeddings
5. Compare F1 scores

### Estimated Effort
- ~1 day to implement
- ~2-4 hours to extract embeddings for 314 tracks
- Should see F1 improvement of 0.10-0.15 if successful

---

## Option C: Custom Electronic Music Features

### Context
Electronic music has specific characteristics not captured by standard audio features:
- Sub-bass punch (kick drums, bass lines)
- Reverb/delay (dub aesthetic)
- Repetitive loops (hypnotic quality)
- Build/drop structure (energy evolution)
- Breakbeat patterns vs 4/4

### Custom Features to Implement

#### 1. Sub-Bass Energy (for "deep" and "dubby")
```python
def extract_subbass_energy(audio, sr=44100):
    \"\"\"Energy in 20-60Hz range (sub-bass/kick fundamental)\"\"\"
    # Apply bandpass filter 20-60Hz
    # Compute RMS energy over time
    # Return mean, std, max
```

#### 2. Bass Punch Detection
```python
def extract_bass_punch(audio, sr=44100):
    \"\"\"Detect kick drum transients in sub-bass range\"\"\"
    # Onset detection in 20-100Hz
    # Count strong onsets
    # Return: punches per minute, regularity
```

#### 3. Reverb Estimation
```python
def estimate_reverb(audio, sr=44100):
    \"\"\"Estimate reverb tail length and wet/dry ratio\"\"\"
    # Autocorrelation analysis
    # Decay time estimation
    # Return: reverb_time, wet_dry_ratio
```

#### 4. Self-Similarity (for "hypnotic")
```python
def compute_self_similarity(audio, sr=44100):
    \"\"\"Measure how repetitive the track is\"\"\"
    # Compute chroma/MFCC features over time
    # Self-similarity matrix
    # Return: repetition_score, loop_length
```

#### 5. Energy Trajectory
```python
def extract_energy_evolution(audio, sr=44100):
    \"\"\"Track energy changes over time (build/drop detection)\"\"\"
    # RMS energy in 8-bar segments
    # Detect rises, falls, plateaus
    # Return: has_build, has_drop, energy_variance
```

### Implementation Steps
1. Write feature extraction functions
2. Integrate into `feature_extraction.py`
3. Re-extract features for all tracks
4. Retrain models
5. Measure improvement

### Estimated Effort
- ~2 days to implement and test all functions
- ~30 minutes to re-extract for 314 tracks
- Should see F1 improvement of 0.05-0.10

---

## Priority Recommendation
1. Try **Option B (Deep Learning Embeddings)** first - faster, proven approach
2. If embeddings don't reach F1>0.60, implement **Option C (Custom Features)**
3. Combine both if needed (embeddings + custom features)
