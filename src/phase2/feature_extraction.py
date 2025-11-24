"""
Feature extraction using Essentia for electronic music.
Optimized for extracting features relevant to energy and vibe classification.
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import DATA_DIR, FEATURES_DIR, MANUAL_LABELS_DIR, SAMPLE_RATE

try:
    import essentia
    import essentia.standard as es

    # Set Essentia log level to ERROR to suppress warnings
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    essentia.log.errorActive = True

    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    es = None  # type: ignore
    print("Warning: Essentia not available. Using fallback feature extraction with librosa.")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class FeatureExtractor:
    """Extract audio features optimized for electronic music classification."""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def extract_features_essentia(self, audio_path):
        """
        Extract features using Essentia (preferred for electronic music).

        Returns a dictionary of audio features.
        """
        if not ESSENTIA_AVAILABLE:
            raise ImportError("Essentia is not available")

        # Load audio
        loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        audio = loader()

        features = {}

        # Rhythm features (important for energy classification)
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        rhythm_result = rhythm_extractor(audio)

        # Handle tuple unpacking - sometimes returns different number of values
        if len(rhythm_result) >= 5:
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_result[:5]
        else:
            bpm, beats, beats_confidence = (
                rhythm_result[0],
                rhythm_result[1],
                rhythm_result[2] if len(rhythm_result) > 2 else 0.5,
            )

        # Ensure scalar values - extract from tuple/array if needed
        if isinstance(bpm, (tuple, list)):
            features["bpm"] = float(bpm[0]) if len(bpm) > 0 else 120.0
        else:
            features["bpm"] = float(bpm)

        if isinstance(beats_confidence, (tuple, list)):
            features["beats_confidence"] = (
                float(beats_confidence[0]) if len(beats_confidence) > 0 else 0.5
            )
        else:
            features["beats_confidence"] = float(beats_confidence)

        features["num_beats"] = len(beats) if hasattr(beats, "__len__") else 0

        # Spectral features (important for vibe classification)
        spectrum = es.Spectrum()
        spectral_centroid = es.Centroid()
        spectral_rolloff = es.RollOff()
        spectral_flux = es.Flux()

        w = es.Windowing(type="hann")

        centroids = []
        rolloffs = []
        fluxes = []

        # Process in frames
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            spec = spectrum(w(frame))
            centroids.append(spectral_centroid(spec))
            rolloffs.append(spectral_rolloff(spec))
            fluxes.append(spectral_flux(spec))

        features["spectral_centroid_mean"] = float(np.mean(centroids))
        features["spectral_centroid_std"] = float(np.std(centroids))
        features["spectral_rolloff_mean"] = float(np.mean(rolloffs))
        features["spectral_rolloff_std"] = float(np.std(rolloffs))
        features["spectral_flux_mean"] = float(np.mean(fluxes))

        # Energy and loudness
        loudness = es.Loudness()
        dynamic_complexity = es.DynamicComplexity()

        loudness_val = loudness(audio)
        dyn_complex_val = dynamic_complexity(audio)

        # Handle potential tuple returns
        features["loudness"] = (
            float(loudness_val[0])
            if isinstance(loudness_val, (tuple, list))
            else float(loudness_val)
        )
        features["dynamic_complexity"] = (
            float(dyn_complex_val[0])
            if isinstance(dyn_complex_val, (tuple, list))
            else float(dyn_complex_val)
        )

        # Tonal features
        try:
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            # Handle potential tuple return
            features["key_strength"] = (
                float(strength[0]) if isinstance(strength, (tuple, list)) else float(strength)
            )
        except Exception:
            features["key_strength"] = 0.0

        # MFCC (timbre features)
        mfcc_extractor = es.MFCC()
        mfccs = []

        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            spec = spectrum(w(frame))
            _, mfcc_coeffs = mfcc_extractor(spec)
            mfccs.append(mfcc_coeffs)

        mfccs = np.array(mfccs)
        for i in range(min(13, mfccs.shape[1])):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[:, i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[:, i]))

        # Zero crossing rate (useful for texture)
        zcr = es.ZeroCrossingRate()
        zcr_values = []

        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            zcr_values.append(zcr(frame))

        features["zcr_mean"] = float(np.mean(zcr_values))
        features["zcr_std"] = float(np.std(zcr_values))

        # === ENHANCED FEATURES FOR ELECTRONIC MUSIC ===

        # 1. Frequency Band Energy (for bass/deep detection)
        # Using LowLevelSpectralExtractor for comprehensive spectral analysis
        try:
            llse = es.LowLevelSpectralExtractor()
            llse_results = llse(audio)

            # Extract energy bands: low (20-150Hz), mid-low (150-800Hz),
            # mid-high (800-4kHz), high (4k-20kHz)
            # These help distinguish deep/bass-heavy vs bright tracks
            if len(llse_results) >= 5:
                energyband_low = llse_results[4]  # spectral_energyband_low
                energyband_mid_low = llse_results[5]  # spectral_energyband_middle_low
                energyband_mid_high = llse_results[6]  # spectral_energyband_middle_high
                energyband_high = llse_results[7]  # spectral_energyband_high

                features["energyband_low_mean"] = float(np.mean(energyband_low))
                features["energyband_low_std"] = float(np.std(energyband_low))
                features["energyband_mid_low_mean"] = float(np.mean(energyband_mid_low))
                features["energyband_mid_high_mean"] = float(np.mean(energyband_mid_high))
                features["energyband_high_mean"] = float(np.mean(energyband_high))

                # Bass ratio: low energy / (low + mid + high)
                total_energy = (
                    energyband_low + energyband_mid_low + energyband_mid_high + energyband_high
                )
                bass_ratio = np.mean(energyband_low / (total_energy + 1e-10))
                features["bass_ratio"] = float(bass_ratio)

            # Inharmonicity (distinguishes synthetic vs organic sounds)
            if len(llse_results) >= 23:
                inharmonicity = llse_results[22]
                features["inharmonicity_mean"] = float(np.mean(inharmonicity))

            # Pitch salience (for melodic content detection)
            if len(llse_results) >= 9:
                pitch_salience = llse_results[8]
                features["pitch_salience_mean"] = float(np.mean(pitch_salience))
                features["pitch_salience_std"] = float(np.std(pitch_salience))

        except Exception:
            # Silently set defaults if extraction fails
            features["energyband_low_mean"] = 0.0
            features["energyband_low_std"] = 0.0
            features["energyband_mid_low_mean"] = 0.0
            features["energyband_mid_high_mean"] = 0.0
            features["energyband_high_mean"] = 0.0
            features["bass_ratio"] = 0.0
            features["inharmonicity_mean"] = 0.0
            features["pitch_salience_mean"] = 0.0
            features["pitch_salience_std"] = 0.0

        # 2. Loop/Repetition Detection (for hypnotic quality)
        try:
            loop_bpm_estimator = es.LoopBpmEstimator()
            loop_bpm_confidence = es.LoopBpmConfidence()

            loop_bpm = loop_bpm_estimator(audio)
            loop_confidence = loop_bpm_confidence(audio)

            features["loop_bpm"] = float(loop_bpm)
            features["loop_confidence"] = float(loop_confidence)

        except Exception:
            # Silently set defaults if loop detection fails
            features["loop_bpm"] = 0.0
            features["loop_confidence"] = 0.0

        # 3. Spectral Contrast (differentiates harmonic vs percussive content)
        try:
            # Compute spectral contrast to help identify dubby/spacious tracks
            # High contrast = peaks and valleys in spectrum (harmonic content)
            # Low contrast = flat spectrum (noise/texture)
            contrast_values = []
            spectral_contrast = es.SpectralContrast()

            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                spec = spectrum(w(frame))
                contrast, _ = spectral_contrast(spec)
                contrast_values.append(contrast)

            contrast_values = np.array(contrast_values)
            features["spectral_contrast_mean"] = float(np.mean(contrast_values))
            features["spectral_contrast_std"] = float(np.std(contrast_values))

        except Exception:
            # Silently set defaults if spectral contrast fails
            features["spectral_contrast_mean"] = 0.0
            features["spectral_contrast_std"] = 0.0

        # 4. Danceability (rhythm regularity, useful for energy classification)
        try:
            danceability = es.Danceability()
            dance_score, _ = danceability(audio)
            features["danceability"] = float(dance_score)
        except Exception:
            # Silently set defaults if danceability fails
            features["danceability"] = 0.0

        return features

    def extract_features_librosa(self, audio_path):
        """
        Fallback feature extraction using librosa.

        Returns a dictionary of audio features.
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa is not available")

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        features = {}

        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features["bpm"] = float(tempo)
        features["num_beats"] = len(beats)
        features["beats_confidence"] = 0.5  # librosa doesn't provide confidence

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)

        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
        features["spectral_flux_mean"] = float(np.mean(spectral_flux))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features["loudness"] = float(np.mean(rms))
        features["dynamic_complexity"] = float(np.std(rms))

        # Tonal features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features["key_strength"] = float(np.max(np.mean(chroma, axis=1)))

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        return features

    def extract_features(self, audio_path):
        """
        Extract features using available library (Essentia preferred, librosa fallback).
        """
        try:
            if ESSENTIA_AVAILABLE:
                return self.extract_features_essentia(audio_path)
            elif LIBROSA_AVAILABLE:
                return self.extract_features_librosa(audio_path)
            else:
                raise ImportError("Neither Essentia nor librosa is available")
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None


def extract_features_for_labeled_tracks(use_simplified=False):
    """
    Extract features for all manually labeled tracks.
    """
    # Load manual labels
    if use_simplified:
        labels_path = os.path.join(MANUAL_LABELS_DIR, "manual_labels_simplified.json")
    else:
        labels_path = os.path.join(MANUAL_LABELS_DIR, "manual_labels.json")

    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    print(f"Found {len(labels)} labeled tracks")

    # Load sample tracks to get full paths
    sample_tracks_path = os.path.join(DATA_DIR, "sample_tracks.json")

    if not os.path.exists(sample_tracks_path):
        print(f"Sample tracks file not found: {sample_tracks_path}")
        print("Please run 'make sample' first to generate the sample list")
        return

    with open(sample_tracks_path, "r", encoding="utf-8") as f:
        sample_tracks = json.load(f)

    # Create a mapping from filename to full path
    filename_to_path = {track["filename"]: track["path"] for track in sample_tracks}

    # Create features directory
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Extract features
    extractor = FeatureExtractor()
    features_data = []
    not_found = []

    for filename, label_data in tqdm(labels.items(), desc="Extracting features"):
        # Get full path from sample tracks
        audio_path = filename_to_path.get(filename)

        if not audio_path or not os.path.exists(audio_path):
            not_found.append(filename)
            continue

        features = extractor.extract_features(audio_path)

        if features:
            features_data.append(
                {
                    "track_id": filename,
                    "features": features,
                    "energy": label_data["energy"],
                    "vibes": label_data["vibes"],
                }
            )

    # Save features
    if use_simplified:
        output_path = os.path.join(FEATURES_DIR, "training_features_simplified.json")
    else:
        output_path = os.path.join(FEATURES_DIR, "training_features.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features_data, f, indent=2)

    print("\nExtraction complete!")
    print(f"Saved features for {len(features_data)} tracks to {output_path}")

    if not_found:
        print(f"\nWarning: {len(not_found)} labeled tracks not found in sample_tracks.json")
        print("This is normal if you've labeled tracks from multiple sample batches.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--simplified", action="store_true", help="Use simplified taxonomy")
    args = parser.parse_args()
    extract_features_for_labeled_tracks(use_simplified=args.simplified)
