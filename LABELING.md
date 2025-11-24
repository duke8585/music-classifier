# Music Labeling Guide - RELABEL Taxonomy v2.0

## Overview

This guide documents the refined taxonomy for labeling electronic music tracks in your DJ library. The system uses **4 orthogonal dimensions** to capture the essential characteristics of each track, optimized for DJ workflow and machine learning classification.

---

## Design Principles

### 1. Orthogonality
Each dimension captures an independent aspect of the music:
- **Energy**: Intensity and set positioning
- **Bass Weight**: Low-frequency presence
- **Rhythm**: Drum pattern characteristics
- **Vibe**: Textural and spatial qualities

### 2. Measurability
All dimensions correlate with extractable audio features:
- Energy → BPM, loudness, spectral flux
- Bass Weight → 20-80Hz energy distribution
- Rhythm → Onset regularity, percussion density
- Vibe → Self-similarity, spectral variance, reverb depth

### 3. Efficiency
- **81 total combinations** (3^4) vs. previous 5,120 combinations
- **Single-select dropdowns** → faster labeling
- **Clear decision boundaries** → less ambiguity

### 4. DJ-Centric
Designed for your collection profile:
- Dub techno, deep house/techno
- Atmospheric and hypnotic styles
- Downtempo and ambient
- Minimal "dark" hard techno

---

## Complete Taxonomy

### Dimension 1: Energy (Set Position)
Captures intensity and where the track fits in a DJ set.

| Label | Description | Audio Features | Examples |
|-------|-------------|----------------|----------|
| **intro/outro** | Low intensity, set bookends, beatless or minimal percussion | BPM < 100, low loudness, sparse onsets | Ambient pads, downtempo, sound design pieces |
| **mid** | Main body energy, groovy and sustained | BPM 110-125, moderate loudness | Deep house, dub techno, groovy techno |
| **peak** | High intensity, dancefloor moments | BPM 125-135+, high loudness, dense arrangement | Peak-time techno, driving house |

**Labeling Tips:**
- Think about **when in your set** you'd play this track
- If it's a transition/warm-up → intro/outro
- If it sustains a groove → mid
- If it's a hands-in-the-air moment → peak

---

### Dimension 2: Bass Weight (Low-End Character)
Captures the sub-bass presence and frequency balance.

| Label | Description | Audio Features | Examples |
|-------|-------------|----------------|----------|
| **light** | Minimal sub-bass, airy and bright | Low 20-80Hz energy, high treble | Minimal ambient, bright melodic house |
| **balanced** | Even frequency distribution | Moderate low-end, clear mids | Deep house, groovy techno |
| **heavy** | Sub-bass dominant, chest-thumping | High 20-80Hz energy, dub delays | Dub techno, bass-heavy house |

**Labeling Tips:**
- Imagine playing this on a club system: **Do you feel it in your chest?** → heavy
- Balanced low-end that doesn't dominate → balanced
- Bright, treble-focused with minimal sub → light

**Terminology Note:**
- "Heavy" bass ≠ "deep techno" genre (that's a combo of heavy + straight + hypnotic/atmospheric)
- This dimension is about **physical bass presence**, not musical style

---

### Dimension 3: Rhythm (Drum Pattern)
Captures percussion density and pattern type.

| Label | Description | Audio Features | Examples |
|-------|-------------|----------------|----------|
| **straight** | 4/4 kick pattern, steady and regular | High onset regularity, 4-on-floor | House, techno, most club tracks |
| **breaks** | Breakbeat patterns, syncopated | Irregular kick placement, shuffled hats | UK garage, breakbeat house, jungle-influenced |
| **sparse** | Minimal percussion, < 10 hits/bar | Low onset density, spacious | Ambient techno, minimal dub, downtempo |

**Labeling Tips:**
- **Straight**: Steady 4/4 kick → most house and techno
- **Breaks**: Syncopated drums, broken beat → UK garage, breakbeat
- **Sparse**: Very minimal drums or beatless → ambient, minimal dub

**Terminology Note:**
- "Sparse" ≠ "minimal techno" genre (that's a combo of dimensions)
- This dimension is about **drum density**, not subgenre

---

### Dimension 4: Vibe (Texture & Space)
Captures the textural evolution and spatial characteristics.

| Label | Description | Audio Features | Examples |
|-------|-------------|----------------|----------|
| **hypnotic** | Repetitive loops, trance-inducing, minimal evolution | High self-similarity, low spectral variance | Loop-based techno, hypnotic dub techno |
| **melodic** | Clear harmonic progressions, memorable hooks | Pitch content, chord changes | Melodic house, progressive techno |
| **atmospheric** | Spacious, reverb-heavy, textured pads | High reverb depth, spatial width, pad presence | Atmospheric dub techno, ambient techno, spacious downtempo |

**Labeling Tips:**
- **Hypnotic**: Does it loop and mesmerize? Minimal changes over time?
- **Melodic**: Are there clear melodies, chord progressions, or hooks?
- **Atmospheric**: Is it spacious, reverb-drenched, pad-heavy?

**Edge Case: Hypnotic vs. Atmospheric**
Many dub techno tracks sit at the boundary between hypnotic and atmospheric. Use this rule:
- **More repetitive loops, less space** → hypnotic (e.g., Basic Channel tight loops)
- **More reverb/delay, spacious, dubby** → atmospheric (e.g., deep dub with cavernous echoes)

When 50/50, choose **atmospheric** for tracks with strong dub character (heavy reverb, spatial depth).

---

## Audio Feature Correlations

This table shows how audio features map to taxonomy dimensions (useful for ML training):

| Dimension | Primary Features | Secondary Features |
|-----------|------------------|-------------------|
| **Energy** | BPM, RMS loudness, spectral flux | Onset rate, dynamic range |
| **Bass Weight** | 20-80Hz energy ratio, sub-bass RMS | Spectral centroid, bass variance |
| **Rhythm** | Onset regularity, percussion density | Beat strength, syncopation index |
| **Vibe** | Self-similarity, spectral variance, reverb depth | Harmonic content, pad presence |

---

## Example Mappings

### Your Collection Archetypes

| Track Type | Energy | Bass Weight | Rhythm | Vibe | Full Label |
|------------|--------|-------------|--------|------|------------|
| **Classic dub techno** (Basic Channel) | mid | heavy | straight | hypnotic | `mid \| heavy \| straight \| hypnotic` |
| **Atmospheric dub techno** (Deepchord) | mid | heavy | straight | atmospheric | `mid \| heavy \| straight \| atmospheric` |
| **Deep house** (groovy, melodic) | mid | balanced | straight | melodic | `mid \| balanced \| straight \| melodic` |
| **Hypnotic peak techno** | peak | balanced | straight | hypnotic | `peak \| balanced \| straight \| hypnotic` |
| **Atmospheric downtempo** | intro/outro | balanced | sparse | atmospheric | `intro/outro \| balanced \| sparse \| atmospheric` |
| **Ambient** (beatless pads) | intro/outro | light | sparse | atmospheric | `intro/outro \| light \| sparse \| atmospheric` |
| **Melodic house** (peak energy) | peak | balanced | straight | melodic | `peak \| balanced \| straight \| melodic` |
| **Breakbeat house** | mid | balanced | breaks | melodic | `mid \| balanced \| breaks \| melodic` |
| **Lofi house** (repetitive, analog) | mid | balanced | straight | hypnotic | `mid \| balanced \| straight \| hypnotic` |
| **Lofi house** (spacious, textured) | mid | balanced | straight | atmospheric | `mid \| balanced \| straight \| atmospheric` |

---

## Comparison with Old Taxonomy

### What Changed

| Old System | New System | Rationale |
|------------|------------|-----------|
| Energy: 5 levels (warm-up → closing) | Energy: 3 levels (intro/outro, mid, peak) | Simpler, maps to set position |
| Vibe: 10 multi-select tags | 4 single-select dimensions | Orthogonal, measurable |
| "deep" vibe (subjective genre) | "heavy" bass weight (objective) | Avoids genre confusion |
| "minimal" vibe | "sparse" rhythm | Clearer descriptor |
| "dark" vibe | **Removed** (not relevant to collection) | - |
| "atmospheric" vibe | "atmospheric" vibe (preserved!) | Critical for collection |
| "trippy" vibe | **Removed** (approximated by atmospheric) | - |
| "analog/lofi" vibe | **Inferred** from other dimensions | Production quality less critical |
| "dubby" vibe | **Inferred** (heavy + straight + atmospheric) | - |

### What's Lost

- **Subjective moods**: "dark," "trippy" (not relevant to your collection)
- **Production quality**: "analog/lofi" (can't explicitly tag)
- **Multi-vibe tagging**: Can't tag both "hypnotic" + "atmospheric" (must choose one)

### What's Gained

- **Faster labeling**: 4 dropdowns vs. 15 buttons/checkboxes
- **Better ML training**: Orthogonal dimensions = better feature correlation
- **Clearer search**: 81 combinations vs. 5,120
- **Preserved atmospheric**: Critical for dub techno/downtempo collection

---

## Labeling Workflow

### Quick Reference

1. **Listen to the track** (use speed control if needed)
2. **Ask yourself 4 questions:**
   - When in my set? → **Energy**
   - Do I feel bass in my chest? → **Bass Weight**
   - What's the drum pattern? → **Rhythm**
   - What's the vibe? → **Vibe**
3. **Select one option per dimension** (4 dropdowns)
4. **Hit spacebar** to save and move to next track

## Edge Cases & Guidelines

### Hypnotic vs. Atmospheric
**Problem:** Many dub techno tracks feel both hypnotic AND atmospheric.

**Solution:** Ask yourself:
- Is it **repetitive loops with tight space**? → hypnotic
- Is it **spacious, reverb-heavy, cavernous**? → atmospheric

When 50/50, choose **atmospheric** for dubby/spacious tracks.

**Examples:**
- Basic Channel "Phylyps Trak" (tight loops) → hypnotic
- Deepchord "Vantage Isle" (cavernous reverb) → atmospheric

### Intro/Outro vs. Mid Energy
**Problem:** Some downtempo tracks have steady grooves (like mid-energy house).

**Solution:** Focus on **BPM and intensity**:
- BPM < 100 or beatless → intro/outro
- BPM 110-125 with sustained groove → mid

**Examples:**
- 95 BPM downtempo with drums → intro/outro
- 120 BPM deep house → mid

### Breaks vs. Straight with Syncopation
**Problem:** Some straight 4/4 tracks have syncopated hats.

**Solution:** Focus on the **kick pattern**:
- 4-on-floor kick → straight (even with syncopated hats)
- Broken kick pattern → breaks

**Examples:**
- Techno with shuffled hats but 4/4 kick → straight
- UK garage with broken kick → breaks

### Melodic vs. Atmospheric
**Problem:** Some atmospheric tracks have melodic pads.

**Solution:** Which is **more prominent**?
- Clear chord progressions/hooks → melodic
- Spatial texture/reverb dominates → atmospheric

**Examples:**
- Progressive house with clear melody → melodic
- Ambient techno with evolving pads → atmospheric

---

## Success Metrics

After labeling 200-300 tracks, the ML models should achieve:
- **Energy classifier**: F1 > 0.70 (clear BPM/loudness correlation)
- **Bass Weight classifier**: F1 > 0.65 (frequency distribution)
- **Rhythm classifier**: F1 > 0.75 (onset regularity)
- **Vibe classifier**: F1 > 0.60 (most subjective, hardest)

If models underperform, we can:
1. Refine labeling guidelines
2. Add more training data
3. Adjust dimension definitions

---

## Questions?

If you encounter tracks that don't fit cleanly:
1. **Trust your DJ instinct** - how would you use this track?
2. **When 50/50, pick the more prominent characteristic**
3. **Document edge cases** - we'll refine the taxonomy as we go

The goal is **consistent, intuitive labels** that help you find the right tracks for your sets. 🎧
