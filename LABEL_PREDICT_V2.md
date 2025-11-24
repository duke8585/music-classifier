# RELABEL v2.0 Pipeline Implementation Plan

## Overview
This document outlines the remaining work to update the entire ML pipeline from the old 2-model taxonomy (energy + multi-label vibes) to the new RELABEL v2.0 4-model taxonomy (energy, bass_weight, rhythm, vibe - all single-label).

**Status:** Phase 1 (Manual Labeling) is complete. Phase 2-4 need updates.

---

## Phase 1: Manual Labeling ✅ COMPLETE

### What's Done:
- ✅ New taxonomy defined in [config/config.py](config/config.py)
- ✅ Radio button UI with keyboard shortcuts (1-3/QWE/ASD/YXC + J/L seek)
- ✅ Updated label storage format: `{energy, bass_weight, rhythm, vibe}`
- ✅ Backwards compatibility with legacy labels
- ✅ Comprehensive labeling guide: [LABELING.md](LABELING.md)
- ✅ Make label opens Safari first, runs Flask in foreground (no duplicate processes)

### What's Skipped (For Now):
- ⏸️ Prediction display in labeler (will be re-implemented after training works)

---

## Phase 2: Feature Extraction & Training 🔧 NEEDS UPDATE

### 2.1 Update Config (`config/config.py`)

**Changes Needed:**

1. **Remove old model targets:**
   ```python
   # DELETE THESE:
   MIN_F1_ENERGY = 0.70
   MIN_F1_VIBE = 0.60
   ```

2. **Add new model targets:**
   ```python
   # Model performance targets for RELABEL v2.0
   MIN_F1_ENERGY = 0.70        # 3-class: intro/outro, mid, peak
   MIN_F1_BASS_WEIGHT = 0.65   # 3-class: light, balanced, heavy
   MIN_F1_RHYTHM = 0.75        # 3-class: straight, breaks, sparse
   MIN_F1_VIBE = 0.60          # 3-class: hypnotic, melodic, atmospheric
   ```

3. **Remove old prediction thresholds:**
   ```python
   # DELETE THESE (or comment out for later):
   PREDICTION_VIBE_THRESHOLD = 0.6
   PREDICTION_AUTO_ACCEPT_ENERGY = 0.8
   PREDICTION_AUTO_ACCEPT_VIBE = 0.75
   ```

---

### 2.2 Update Feature Extraction (`src/phase2/feature_extraction.py`)

**What to Check:**

1. **Remove `--simplified` flag and related logic:**
   - The simplified taxonomy is now the main taxonomy
   - Remove any command-line argument handling for `--simplified`
   - Remove conditional logic that switches between taxonomies

2. **Label Format Handling:**
   - Script currently expects: `{"energy": "...", "vibes": [...]}`
   - Needs to handle: `{"energy": "...", "bass_weight": "...", "rhythm": "...", "vibe": "..."}`
   - Add backwards compatibility check

3. **Label Validation:**
   - Ensure all 4 dimensions are present for new format
   - Skip tracks with missing dimensions
   - Log how many old vs new format labels are found

4. **Feature Extraction Logic:**
   - Current Essentia features should work for all 4 dimensions
   - No changes needed if features are comprehensive
   - Verify features are saved per-track, not per-label-type

**Expected Output:**
- `data/features/training_features.json` with format:
  ```json
  {
    "filename": "track.aiff",
    "features": [...],
    "energy": "mid",
    "bass_weight": "heavy",
    "rhythm": "straight",
    "vibe": "atmospheric"
  }
  ```

---

### 2.3 Update Training Script (`src/phase2/train_models.py`)

**Major Changes:**

1. **Remove `--simplified` flag and related logic:**
   - The simplified taxonomy is now the main taxonomy
   - Remove any command-line argument handling for `--simplified`
   - Remove conditional logic that switches between taxonomies

2. **Train 4 Separate Models Instead of 2:**
   ```python
   # OLD:
   # - energy_model (RandomForest, 5 classes)
   # - vibe_model (MultiLabelBinarizer + RandomForest, 10 classes)

   # NEW:
   # - energy_model (RandomForest, 3 classes)
   # - bass_weight_model (RandomForest, 3 classes)
   # - rhythm_model (RandomForest, 3 classes)
   # - vibe_model (RandomForest, 3 classes)
   ```

3. **Remove Multi-Label Handling:**
   - Old vibe classifier used `MultiLabelBinarizer`
   - New classifiers are all single-label (simpler!)
   - Use standard `LabelEncoder` for all 4

4. **Update Model Saving:**
   - Save 4 separate model files:
     - `data/models/energy_model.pkl`
     - `data/models/bass_weight_model.pkl`
     - `data/models/rhythm_model.pkl`
     - `data/models/vibe_model.pkl`

5. **Update Evaluation Metrics:**
   - Print F1 scores for all 4 models
   - Compare against new thresholds (MIN_F1_ENERGY, MIN_F1_BASS_WEIGHT, etc.)
   - Generate confusion matrices for all 4 dimensions

6. **Update Class Balance Handling:**
   - Check class distribution for all 4 dimensions
   - Use `class_weight='balanced'` if needed
   - Log class counts for each dimension

**Expected Output:**
```
Training Energy Classifier...
  Class distribution: intro/outro: 50, mid: 120, peak: 80
  F1 Score: 0.73 ✓ (target: 0.70)

Training Bass Weight Classifier...
  Class distribution: light: 60, balanced: 100, heavy: 90
  F1 Score: 0.68 ✓ (target: 0.65)

Training Rhythm Classifier...
  Class distribution: straight: 180, breaks: 40, sparse: 30
  F1 Score: 0.78 ✓ (target: 0.75)

Training Vibe Classifier...
  Class distribution: hypnotic: 90, melodic: 80, atmospheric: 80
  F1 Score: 0.62 ✓ (target: 0.60)

All models saved to data/models/
```

---

### 2.4 Testing Phase 2

**Manual Test Steps:**

1. **Generate small test dataset:**
   - Label 30-50 tracks with new taxonomy
   - Ensure variety across all dimensions

2. **Run feature extraction:**
   ```bash
   make extract
   ```
   - Verify no errors
   - Check `data/features/training_features.json` format

3. **Run training:**
   ```bash
   make train
   ```
   - Verify 4 models are created
   - Check F1 scores are reasonable (may be low with small dataset)
   - Inspect confusion matrices

4. **Test with full labeled dataset (200-300 tracks):**
   - Re-run `make extract && make train`
   - Verify F1 scores meet targets

---

## Phase 3: Inference 🔧 NEEDS UPDATE

### 3.1 Update Prediction Script (`src/phase3/predict.py`)

**Changes Needed:**

1. **Load 4 Models Instead of 2:**
   ```python
   energy_model = load_model('data/models/energy_model.pkl')
   bass_weight_model = load_model('data/models/bass_weight_model.pkl')
   rhythm_model = load_model('data/models/rhythm_model.pkl')
   vibe_model = load_model('data/models/vibe_model.pkl')
   ```

2. **Update Prediction Logic:**
   ```python
   # OLD:
   # predictions = {
   #   "energy": "peak",
   #   "vibe_details": [{"label": "hypnotic", "confidence": 0.8, "predicted": True}, ...]
   # }

   # NEW:
   # predictions = {
   #   "energy": {"label": "mid", "confidence": 0.75},
   #   "bass_weight": {"label": "heavy", "confidence": 0.82},
   #   "rhythm": {"label": "straight", "confidence": 0.90},
   #   "vibe": {"label": "atmospheric", "confidence": 0.68}
   # }
   ```

3. **Update Output Format:**
   - Remove multi-label threshold logic
   - Each dimension returns top prediction + confidence
   - Optionally include top-3 for each dimension

4. **Update CSV Export:**
   - Columns: `filename, energy, energy_conf, bass_weight, bass_conf, rhythm, rhythm_conf, vibe, vibe_conf`
   - Easy to filter by confidence thresholds

**Expected Output:**
```json
[
  {
    "filename": "track.aiff",
    "energy": {"label": "mid", "confidence": 0.75, "top3": [...]},
    "bass_weight": {"label": "heavy", "confidence": 0.82, "top3": [...]},
    "rhythm": {"label": "straight", "confidence": 0.90, "top3": [...]},
    "vibe": {"label": "atmospheric", "confidence": 0.68, "top3": [...]}
  }
]
```

---

### 3.2 Testing Phase 3

**Manual Test Steps:**

1. **Run predictions on test tracks:**
   ```bash
   make predict
   ```
   - Verify predictions.json format is correct
   - Check CSV export

2. **Spot-check predictions:**
   - Listen to 5-10 tracks
   - Compare predictions to your own judgment
   - Note any obvious misclassifications

---

## Phase 4: Rekordbox Integration 🔧 NEEDS MINIMAL UPDATE

### 4.1 Update Rekordbox Export (`src/phase4/rekordbox_export.py`)

**Changes Needed:**

1. **Tag Format:**
   - OLD: `energy: peak, vibes: hypnotic, melodic`
   - NEW: `energy: mid | bass: heavy | rhythm: straight | vibe: atmospheric`
   - Use pipe separator for readability

2. **MyTag Fields:**
   - Combine all 4 dimensions into a single tag string
   - Or use separate MyTag fields if Rekordbox supports multiple

**Expected Output:**
- Updated Rekordbox XML with MyTag fields populated

---

## Phase 5: Re-Enable Prediction Display in Labeler 🔧 OPTIONAL

**After Phase 2-3 are working:**

### 5.1 Update Label UI (`templates/label.html`)

**Currently Removed:**
- Confidence badges on radio buttons
- "Accept Predictions" button
- Prediction threshold info

**What to Re-Implement:**

1. **Show Predictions as Subtle Hints:**
   - Add small confidence badge next to predicted options
   - Format: `[Q] Light - Minimal sub-bass (AI: 65%)`
   - Only show if confidence > 60%

2. **Auto-fill Predictions:**
   - On page load, if predictions exist and confidence > 80%, pre-select
   - Still allow manual override
   - Visual indicator that it's auto-filled

3. **"Accept All Predictions" Button:**
   - Shows only if all 4 dimensions have high-confidence predictions
   - Fills in all 4 dimensions at once
   - User can still adjust before saving

### 5.2 Update Backend (`src/phase1/label_app.py`)

**Update `/get_prediction/<filename>` route:**
```python
# OLD format:
# {
#   "energy": "peak",
#   "energy_top3": [...],
#   "vibe_details": [...]
# }

# NEW format:
# {
#   "energy": {"label": "mid", "confidence": 0.75},
#   "bass_weight": {"label": "heavy", "confidence": 0.82},
#   "rhythm": {"label": "straight", "confidence": 0.90},
#   "vibe": {"label": "atmospheric", "confidence": 0.68}
# }
```

---

## Implementation Order

### Immediate (While Labeling):
1. ✅ Fix `make label` (already done)
2. ✅ Create this implementation plan

### Next Session (New Context):
1. **Update Makefile** (2 min)
   - Remove `extract-simplified` and `train-simplified` targets
   - Update help text

2. **Update config.py** (5 min)
   - Remove old MIN_F1 vars
   - Add new MIN_F1 vars for 4 models
   - Remove prediction thresholds

3. **Update feature_extraction.py** (15 min)
   - Remove `--simplified` flag
   - Handle new label format
   - Add backwards compatibility
   - Test with small dataset

4. **Update train_models.py** (30 min)
   - Remove `--simplified` flag
   - Train 4 separate models
   - Remove multi-label logic
   - Update evaluation metrics
   - Test with small dataset

5. **Test Phase 2** (15 min)
   - Run `make extract && make train` with full labeled data
   - Verify F1 scores
   - Inspect confusion matrices

6. **Update predict.py** (20 min)
   - Load 4 models
   - Update prediction format
   - Update CSV export

7. **Test Phase 3** (10 min)
   - Run `make predict` on test tracks
   - Spot-check results

8. **Update rekordbox_export.py** (10 min)
   - Update tag format
   - Test XML export

9. **(Optional) Re-enable prediction display in labeler** (30 min)
   - Update label.html
   - Update label_app.py
   - Test in browser

**Total Estimated Time:** 2-3 hours

---

## Risk Assessment

### Low Risk:
- ✅ Config updates (straightforward)
- ✅ Feature extraction (minimal changes)
- ✅ Rekordbox export (just formatting)

### Medium Risk:
- ⚠️ Training script refactor (biggest change, but well-defined)
- ⚠️ Prediction script update (depends on training working)

### High Risk:
- 🔴 Model performance (depends on label quality & quantity)
  - Mitigation: Start with 200+ labeled tracks
  - Mitigation: Use class_weight='balanced' for imbalanced classes
  - Mitigation: Check confusion matrices to identify problem classes

---

## Success Criteria

### Phase 2 (Training):
- ✅ All 4 models train without errors
- ✅ F1 scores meet or exceed targets:
  - Energy: F1 > 0.70
  - Bass Weight: F1 > 0.65
  - Rhythm: F1 > 0.75
  - Vibe: F1 > 0.60
- ✅ Confusion matrices show reasonable performance
- ✅ Class distributions are logged

### Phase 3 (Inference):
- ✅ Predictions run on all 6,000 tracks without errors
- ✅ predictions.json and CSV are correctly formatted
- ✅ Spot-checks show reasonable predictions

### Phase 4 (Rekordbox):
- ✅ XML export works
- ✅ Tags are readable in Rekordbox
- ✅ Can search by combined tags

---

## Notes for Next Session

- All Phase 1 files are already updated for new taxonomy
- Focus on Phase 2 (training) first - this is the critical path
- Test with small dataset (30-50 tracks) before full training
- Consider active learning if model performance is poor (label more of the misclassified tracks)
- Prediction display in labeler is optional - can be added later if needed

---

## File Checklist

### Files to Update:
- [ ] `Makefile` (remove extract-simplified and train-simplified targets)
- [ ] `config/config.py` (remove old MIN_F1, add new ones)
- [ ] `src/phase2/feature_extraction.py` (remove --simplified flag, handle new label format)
- [ ] `src/phase2/train_models.py` (remove --simplified flag, train 4 models not 2)
- [ ] `src/phase3/predict.py` (load 4 models, update format)
- [ ] `src/phase4/rekordbox_export.py` (update tag format)
- [ ] *(Optional)* `templates/label.html` (re-enable predictions)
- [ ] *(Optional)* `src/phase1/label_app.py` (update prediction route)

### Files Already Updated (Phase 1):
- ✅ `config/config.py` (taxonomy definitions)
- ✅ `templates/label.html` (radio button UI)
- ✅ `src/phase1/label_app.py` (new label format)
- ✅ `LABELING.md` (documentation)
- ✅ `CLAUDE.md` (project overview)
- ✅ `Makefile` (make label fix)

---

## References

- **Taxonomy Documentation:** [LABELING.md](LABELING.md)
- **Project Overview:** [CLAUDE.md](CLAUDE.md)
- **Config:** [config/config.py](config/config.py)
- **Current Labels:** `data/manual_labels.json`
- **Training Features:** `data/features/training_features.json`
- **Model Files:** `data/models/*.pkl`
- **Predictions:** `data/predictions/predictions.json`
