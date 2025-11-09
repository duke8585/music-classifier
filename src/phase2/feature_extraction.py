"""
Feature extraction using Essentia for electronic music.
Optimized for extracting features relevant to energy and vibe classification.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import FEATURES_DIR, MANUAL_LABELS_DIR, SAMPLE_RATE

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
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
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        features['bpm'] = float(bpm)
        features['beats_confidence'] = float(beats_confidence)
        features['num_beats'] = len(beats)

        # Spectral features (important for vibe classification)
        spectrum = es.Spectrum()
        spectral_centroid = es.Centroid()
        spectral_rolloff = es.RollOff()
        spectral_flux = es.Flux()

        w = es.Windowing(type='hann')
        fft = es.FFT()

        centroids = []
        rolloffs = []
        fluxes = []

        # Process in frames
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            spec = spectrum(w(frame))
            centroids.append(spectral_centroid(spec))
            rolloffs.append(spectral_rolloff(spec))
            fluxes.append(spectral_flux(spec))

        features['spectral_centroid_mean'] = float(np.mean(centroids))
        features['spectral_centroid_std'] = float(np.std(centroids))
        features['spectral_rolloff_mean'] = float(np.mean(rolloffs))
        features['spectral_rolloff_std'] = float(np.std(rolloffs))
        features['spectral_flux_mean'] = float(np.mean(fluxes))

        # Energy and loudness
        loudness = es.Loudness()
        dynamic_complexity = es.DynamicComplexity()

        features['loudness'] = float(loudness(audio))
        features['dynamic_complexity'] = float(dynamic_complexity(audio))

        # Tonal features
        try:
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            features['key_strength'] = float(strength)
        except:
            features['key_strength'] = 0.0

        # MFCC (timbre features)
        mfcc_extractor = es.MFCC()
        mfccs = []

        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            spec = spectrum(w(frame))
            _, mfcc_coeffs = mfcc_extractor(spec)
            mfccs.append(mfcc_coeffs)

        mfccs = np.array(mfccs)
        for i in range(min(13, mfccs.shape[1])):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[:, i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[:, i]))

        # Zero crossing rate (useful for texture)
        zcr = es.ZeroCrossingRate()
        zcr_values = []

        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            zcr_values.append(zcr(frame))

        features['zcr_mean'] = float(np.mean(zcr_values))
        features['zcr_std'] = float(np.std(zcr_values))

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
        features['bpm'] = float(tempo)
        features['num_beats'] = len(beats)
        features['beats_confidence'] = 0.5  # librosa doesn't provide confidence

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)

        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        features['spectral_flux_mean'] = float(np.mean(spectral_flux))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['loudness'] = float(np.mean(rms))
        features['dynamic_complexity'] = float(np.std(rms))

        # Tonal features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['key_strength'] = float(np.max(np.mean(chroma, axis=1)))

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

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
    labels_path = os.path.join(MANUAL_LABELS_DIR, 'manual_labels.json')

    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    print(f"Found {len(labels)} labeled tracks")

    # Create features directory
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Extract features
    extractor = FeatureExtractor()
    features_data = []

    for label in tqdm(labels, desc="Extracting features"):
        audio_path = label.get('path', '')

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue

        features = extractor.extract_features(audio_path)

        if features:
            features_data.append({
                'track_id': label['track_id'],
                'features': features,
                'energy': label['energy'],
                'vibes': label['vibes']
            })

    # Save features
    output_path = os.path.join(FEATURES_DIR, 'training_features.json')
    with open(output_path, 'w') as f:
        json.dump(features_data, f, indent=2)

    print(f"Saved features for {len(features_data)} tracks to {output_path}")


if __name__ == '__main__':
    extract_features_for_labeled_tracks()
