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
    import essentia.standard as es

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
            bpm, beats, beats_confidence = rhythm_result[0], rhythm_result[1], rhythm_result[2] if len(rhythm_result) > 2 else 0.5

        # Ensure scalar values - extract from tuple/array if needed
        if isinstance(bpm, (tuple, list)):
            features["bpm"] = float(bpm[0]) if len(bpm) > 0 else 120.0
        else:
            features["bpm"] = float(bpm)

        if isinstance(beats_confidence, (tuple, list)):
            features["beats_confidence"] = float(beats_confidence[0]) if len(beats_confidence) > 0 else 0.5
        else:
            features["beats_confidence"] = float(beats_confidence)

        features["num_beats"] = len(beats) if hasattr(beats, '__len__') else 0

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
        features["loudness"] = float(loudness_val[0]) if isinstance(loudness_val, (tuple, list)) else float(loudness_val)
        features["dynamic_complexity"] = float(dyn_complex_val[0]) if isinstance(dyn_complex_val, (tuple, list)) else float(dyn_complex_val)

        # Tonal features
        try:
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            # Handle potential tuple return
            features["key_strength"] = float(strength[0]) if isinstance(strength, (tuple, list)) else float(strength)
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


def extract_features_for_labeled_tracks():
    """
    Extract features for all manually labeled tracks.
    """
    # Load manual labels
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
    output_path = os.path.join(FEATURES_DIR, "training_features.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features_data, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"Saved features for {len(features_data)} tracks to {output_path}")

    if not_found:
        print(f"\nWarning: {len(not_found)} labeled tracks not found in sample_tracks.json")
        print("This is normal if you've labeled tracks from multiple sample batches.")


if __name__ == "__main__":
    extract_features_for_labeled_tracks()
