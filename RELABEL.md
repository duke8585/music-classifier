# Alternative Taxonomy: Orthogonal Dimensions

## Context

Current taxonomy uses overlapping, subjective labels that are hard for ML models to learn from audio features. This alternative approach uses **orthogonal dimensions** - each dimension captures a different, measurable aspect of the music.

### Why Orthogonal Dimensions?

1. **No overlap** - Each dimension is independent
2. **Measurable** - Can be detected from audio features
3. **Practical** - Easy to search/filter (e.g., "deep + hypnotic + mid")
4. **Smaller search space** - 3⁴ = 81 combinations vs 2¹⁰ = 1024

---

## Proposed Taxonomy

### Dimension 1: Energy Level (3 classes)

Captures: BPM, percussion intensity, frequency spectrum fullness

| Label | Description | Audio Cues | Measurable Features |
|-------|-------------|------------|---------------------|
| **ambient** | Beatless, minimal percussion, atmospheric | No clear beats, <100 BPM or no BPM | Beat confidence < 0.3, low spectral energy |
| **mid** | Steady groove, consistent energy | 110-125 BPM, regular kicks | BPM 110-125, beat confidence > 0.7 |
| **peak** | Maximum energy, full spectrum | 125-135+ BPM, loud and full | BPM > 125, high loudness, full spectrum |

**Labeling guidance:**
- ambient: Intro/outro tracks, soundscapes, no danceable beat
- mid: Main grooves, tracks you'd play for 1-2 hours straight
- peak: Peak-time bangers, maximum dance floor energy

---

### Dimension 2: Bass Character (3 classes)

Captures: Sub-bass presence, frequency distribution

| Label | Description | Audio Cues | Measurable Features |
|-------|-------------|------------|---------------------|
| **deep** | Sub-bass dominant, heavy low-end | Strong sub-bass, kick punch | Energy in 20-80Hz > threshold |
| **balanced** | Even frequency distribution | Bass present but not dominant | Balanced spectral energy |
| **bright** | Mid/high focus, minimal bass | Crisp percussion, thin bass | Low energy in <80Hz range |

**Labeling guidance:**
- deep: You feel the bass in your chest, sub-heavy dub techno
- balanced: Standard house/techno with normal bass
- bright: Minimal techno, high-hat focused, thin sound

---

### Dimension 3: Rhythmic Style (3 classes)

Captures: Drum pattern type, syncopation

| Label | Description | Audio Cues | Measurable Features |
|-------|-------------|------------|---------------------|
| **straight** | 4/4 kick pattern | Boom-boom-boom-boom | Regular onsets, 4/4 pattern |
| **breaks** | Breakbeat, shuffled, syncopated | Amen break, jungle drums | Irregular onsets, syncopation |
| **minimal** | Sparse percussion | Few elements, space between hits | Low onset density |

**Labeling guidance:**
- straight: Classic techno/house 4/4
- breaks: Breakbeat, UK bass, jungle, footwork
- minimal: Ambient techno, spacious, < 10 percussion hits per bar

---

### Dimension 4: Vibe/Texture (3 classes)

Captures: Temporal evolution + timbral character

| Label | Description | Audio Cues | Measurable Features |
|-------|-------------|------------|---------------------|
| **hypnotic** | Repetitive, minimal variation | Loops, mantric, trance-inducing | High self-similarity, long loops |
| **melodic** | Clear melodic content | Hooks, chords, memorable melodies | Strong pitch content, harmonic structure |
| **experimental** | Evolving, textured, unpredictable | Soundscapes, noise, constantly changing | Low self-similarity, high spectral complexity |

**Labeling guidance:**
- hypnotic: Feels like a loop, repetitive, trance-like
- melodic: You can hum/remember the melody or chord progression
- experimental: Weird, textured, soundscape-y, keeps surprising you

---

## Example Combinations

| Track Type | Energy | Bass | Rhythm | Vibe |
|------------|--------|------|--------|------|
| Classic dub techno | mid | deep | straight | hypnotic |
| Melodic house | mid | balanced | straight | melodic |
| Ambient intro | ambient | balanced | minimal | experimental |
| Peak-time techno | peak | deep | straight | hypnotic |
| Breakbeat roller | mid | deep | breaks | melodic |
| Minimal techno | mid | balanced | minimal | hypnotic |
| Experimental ambient | ambient | bright | minimal | experimental |
| UK bass | mid | deep | breaks | experimental |

---

## Labeling Strategy

### Phase 1: Test Relabeling (50 tracks)
1. Pick 50 diverse tracks
2. Label with this new taxonomy
3. Extract features + train
4. Check if F1 improves
5. If F1 > 0.60, commit to full relabel

### Phase 2: Full Relabeling (300 tracks)
1. Create updated labeling tool with 4 dropdowns
2. Label 300 tracks (can reuse current track selection)
3. Takes ~2-3 hours (faster than current multi-select approach)
4. Expected F1: 0.65-0.75 (orthogonal dimensions + measurable features)

### Phase 3: Expansion
1. Once models work well, label remaining ~5,700 tracks
2. Use active learning: model suggests labels, you confirm/correct
3. Full library tagged in 1-2 weeks

---

## Key Advantages

1. **Faster labeling** - 4 dropdowns vs multi-select checkboxes
2. **Less cognitive load** - Each dimension is independent
3. **Better ML performance** - Features directly map to labels
4. **More flexible searching** - Combine dimensions: "deep + hypnotic + mid + straight" = dub techno
5. **Scalable** - Easy to add dimensions later (e.g., "vocal presence", "organic vs synthetic")

---

## Discussion Points

1. Do these 4 dimensions capture what you care about for DJ selection?
2. Are there tracks that don't fit this taxonomy?
3. Would you add/remove/modify any dimensions?
4. Should we test on 50 tracks first or commit to full relabel?

---

## Migration Path

If we decide to use this:

1. Keep current 314 labels as backup
2. Relabel same 314 tracks with new taxonomy
3. Train both models, compare F1
4. Choose best performing taxonomy
5. Optionally: train a "translator" model (old labels → new labels) for the remaining 5,700 tracks
