#!/usr/bin/env python3
"""
Training script for Electronic Music Mood/Vibe Classifier
Trains separate models for Energy and Vibe classification
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MusicClassifierTrainer:
    """Handles training of music classifiers"""

    # Define valid labels
    ENERGY_LABELS = ['warm-up', 'building', 'peak', 'intense', 'closing']
    VIBE_LABELS = ['dark', 'melodic', 'hypnotic', 'dubby', 'atmospheric',
                   'trippy', 'analog/lofi', 'clean/digital']

    def __init__(self, data_dir: str = 'data', models_dir: str = 'models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Will be loaded from data
        self.labels_data = None
        self.features = None
        self.energy_labels = None
        self.vibe_labels = None

    def validate_environment(self) -> bool:
        """Validate that all required dependencies are available"""
        logger.info("Validating environment...")

        missing_deps = []
        try:
            import essentia
            logger.info(f"✓ Essentia version: {essentia.__version__}")
        except ImportError:
            missing_deps.append("essentia")
            logger.error("✗ Essentia not found")

        try:
            import librosa
            logger.info(f"✓ librosa version: {librosa.__version__}")
        except ImportError:
            missing_deps.append("librosa")
            logger.error("✗ librosa not found")

        try:
            import sklearn
            logger.info(f"✓ scikit-learn version: {sklearn.__version__}")
        except ImportError:
            missing_deps.append("scikit-learn")
            logger.error("✗ scikit-learn not found")

        if missing_deps:
            logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.error("Install with: pip install essentia librosa scikit-learn")
            return False

        return True

    def load_labels(self, labels_file: str = 'manual_labels.json') -> bool:
        """Load and validate manual labels"""
        labels_path = self.data_dir / labels_file

        if not labels_path.exists():
            logger.error(f"Labels file not found: {labels_path}")
            logger.error("Expected format: {\"track_id\": {\"energy\": \"peak\", \"vibe\": [\"dark\", \"melodic\"], \"audio_path\": \"...\"}}")
            return False

        try:
            with open(labels_path, 'r') as f:
                self.labels_data = json.load(f)

            logger.info(f"✓ Loaded {len(self.labels_data)} labeled tracks")

            # Validate labels
            return self._validate_labels()

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in labels file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return False

    def _validate_labels(self) -> bool:
        """Validate label format and content"""
        if not self.labels_data:
            logger.error("No labels data loaded")
            return False

        valid_count = 0
        issues = []

        for track_id, labels in self.labels_data.items():
            # Check required fields
            if 'energy' not in labels:
                issues.append(f"{track_id}: missing 'energy' field")
                continue
            if 'vibe' not in labels:
                issues.append(f"{track_id}: missing 'vibe' field")
                continue
            if 'audio_path' not in labels:
                issues.append(f"{track_id}: missing 'audio_path' field")
                continue

            # Validate energy label
            if labels['energy'] not in self.ENERGY_LABELS:
                issues.append(f"{track_id}: invalid energy label '{labels['energy']}'")
                continue

            # Validate vibe labels (can be list or single value)
            vibe_labels = labels['vibe'] if isinstance(labels['vibe'], list) else [labels['vibe']]
            invalid_vibes = [v for v in vibe_labels if v not in self.VIBE_LABELS]
            if invalid_vibes:
                issues.append(f"{track_id}: invalid vibe labels {invalid_vibes}")
                continue

            # Check if audio file exists
            audio_path = Path(labels['audio_path'])
            if not audio_path.exists():
                issues.append(f"{track_id}: audio file not found at {audio_path}")
                continue

            valid_count += 1

        # Report issues
        if issues:
            logger.warning(f"Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more")

        logger.info(f"Valid tracks: {valid_count}/{len(self.labels_data)}")

        if valid_count < 50:
            logger.error("Insufficient training data! Need at least 50 valid labeled tracks")
            logger.error("Recommended: 200-300 tracks for good performance")
            return False
        elif valid_count < 150:
            logger.warning(f"Only {valid_count} valid tracks. Recommend 200-300 for best results")

        # Check class distribution
        self._check_class_distribution()

        return valid_count > 0

    def _check_class_distribution(self):
        """Check and report class distribution"""
        energy_counts = {label: 0 for label in self.ENERGY_LABELS}
        vibe_counts = {label: 0 for label in self.VIBE_LABELS}

        for labels in self.labels_data.values():
            if 'energy' in labels and labels['energy'] in self.ENERGY_LABELS:
                energy_counts[labels['energy']] += 1

            if 'vibe' in labels:
                vibe_labels = labels['vibe'] if isinstance(labels['vibe'], list) else [labels['vibe']]
                for v in vibe_labels:
                    if v in self.VIBE_LABELS:
                        vibe_counts[v] += 1

        logger.info("\nEnergy distribution:")
        for label, count in energy_counts.items():
            logger.info(f"  {label}: {count}")

        logger.info("\nVibe distribution:")
        for label, count in vibe_counts.items():
            logger.info(f"  {label}: {count}")

        # Warn about class imbalance
        energy_min = min(energy_counts.values())
        if energy_min < 10:
            logger.warning(f"⚠ Energy class imbalance: minimum {energy_min} samples")

        vibe_min = min(vibe_counts.values())
        if vibe_min < 10:
            logger.warning(f"⚠ Vibe class imbalance: minimum {vibe_min} samples")

    def extract_features(self) -> bool:
        """Extract audio features from labeled tracks"""
        logger.info("Extracting audio features...")

        try:
            import essentia.standard as es
        except ImportError:
            logger.error("Essentia not available for feature extraction")
            return False

        features_list = []
        energy_labels = []
        vibe_labels_list = []
        failed_tracks = []

        for track_id, labels in self.labels_data.items():
            # Skip invalid entries
            if not all(k in labels for k in ['energy', 'vibe', 'audio_path']):
                continue

            audio_path = Path(labels['audio_path'])
            if not audio_path.exists():
                continue

            try:
                # Extract features using Essentia
                features = self._extract_single_track_features(str(audio_path))

                if features is not None:
                    features_list.append(features)
                    energy_labels.append(labels['energy'])

                    # Handle multiple vibe labels (multi-label)
                    # For now, take the first vibe label (can be improved)
                    vibe = labels['vibe'][0] if isinstance(labels['vibe'], list) else labels['vibe']
                    vibe_labels_list.append(vibe)
                else:
                    failed_tracks.append(track_id)

            except Exception as e:
                logger.error(f"Failed to extract features from {track_id}: {e}")
                failed_tracks.append(track_id)

        if failed_tracks:
            logger.warning(f"Failed to extract features from {len(failed_tracks)} tracks")

        if not features_list:
            logger.error("No features extracted! Check audio files and paths")
            return False

        self.features = np.array(features_list)
        self.energy_labels = np.array(energy_labels)
        self.vibe_labels = np.array(vibe_labels_list)

        logger.info(f"✓ Extracted features: shape {self.features.shape}")
        logger.info(f"  Feature dimension: {self.features.shape[1]}")
        logger.info(f"  Number of samples: {self.features.shape[0]}")

        # Check for NaN or Inf values
        if np.any(np.isnan(self.features)):
            logger.warning("⚠ NaN values detected in features")
            self.features = np.nan_to_num(self.features)

        if np.any(np.isinf(self.features)):
            logger.warning("⚠ Inf values detected in features")
            self.features = np.nan_to_num(self.features)

        return True

    def _extract_single_track_features(self, audio_path: str) -> np.ndarray:
        """Extract features from a single audio file using Essentia"""
        import essentia.standard as es

        # Load audio
        loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
        audio = loader()

        if len(audio) == 0:
            logger.warning(f"Empty audio file: {audio_path}")
            return None

        # Extract various features
        features = []

        # Spectral features
        spectrum = es.Spectrum()
        spectral_centroid = es.Centroid()
        spectral_rolloff = es.RollOff()

        # Rhythm features
        rhythm_extractor = es.RhythmExtractor2013()

        # Timbral features
        mfcc = es.MFCC()

        # Process in frames for spectral features
        frame_size = 2048
        hop_size = 1024

        centroids = []
        rolloffs = []
        mfccs = []

        for frame_start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[frame_start:frame_start + frame_size]

            # Window the frame
            windowed = frame * np.hanning(len(frame))

            spec = spectrum(windowed)
            centroids.append(spectral_centroid(spec))
            rolloffs.append(spectral_rolloff(spec))

            mfcc_bands, mfcc_coeffs = mfcc(spec)
            mfccs.append(mfcc_coeffs)

        # Aggregate frame-level features
        features.extend([
            np.mean(centroids),
            np.std(centroids),
            np.mean(rolloffs),
            np.std(rolloffs)
        ])

        # Add MFCC statistics (mean and std of each coefficient)
        mfccs = np.array(mfccs)
        features.extend(np.mean(mfccs, axis=0))
        features.extend(np.std(mfccs, axis=0))

        # Extract rhythm features
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        features.extend([bpm, beats_confidence])

        return np.array(features)

    def train_models(self) -> bool:
        """Train Energy and Vibe classification models"""
        if self.features is None or self.energy_labels is None:
            logger.error("No features available. Run extract_features first")
            return False

        logger.info("\n" + "="*60)
        logger.info("TRAINING ENERGY CLASSIFIER")
        logger.info("="*60)
        success_energy = self._train_single_model(
            self.features,
            self.energy_labels,
            'energy',
            self.ENERGY_LABELS
        )

        logger.info("\n" + "="*60)
        logger.info("TRAINING VIBE CLASSIFIER")
        logger.info("="*60)
        success_vibe = self._train_single_model(
            self.features,
            self.vibe_labels,
            'vibe',
            self.VIBE_LABELS
        )

        return success_energy and success_vibe

    def _train_single_model(self, X: np.ndarray, y: np.ndarray,
                           model_name: str, class_names: List[str]) -> bool:
        """Train a single classifier"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Train model
        logger.info(f"Training Random Forest for {model_name}...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")

        # Predictions
        y_pred = model.predict(X_test)

        # F1 score
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        logger.info(f"F1 Score (macro): {f1_macro:.3f}")
        logger.info(f"F1 Score (weighted): {f1_weighted:.3f}")

        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        logger.info(str(cm))

        # Cross-validation
        logger.info("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        logger.info(f"CV F1 scores: {cv_scores}")
        logger.info(f"CV F1 mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Save model
        model_path = self.models_dir / f'{model_name}_classifier.joblib'
        joblib.dump(model, model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Check if meets success criteria
        target_f1 = 0.70 if model_name == 'energy' else 0.60
        if f1_macro >= target_f1:
            logger.info(f"✓ SUCCESS: F1 score {f1_macro:.3f} meets target {target_f1}")
        else:
            logger.warning(f"⚠ F1 score {f1_macro:.3f} below target {target_f1}")
            logger.warning("Consider: more training data, feature engineering, or hyperparameter tuning")

        return True

    def run_full_training(self) -> bool:
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("ELECTRONIC MUSIC CLASSIFIER - TRAINING PIPELINE")
        logger.info("="*60)

        # Step 1: Validate environment
        if not self.validate_environment():
            logger.error("Environment validation failed!")
            return False

        # Step 2: Load labels
        logger.info("\n" + "="*60)
        logger.info("LOADING LABELS")
        logger.info("="*60)
        if not self.load_labels():
            logger.error("Failed to load labels!")
            return False

        # Step 3: Extract features
        logger.info("\n" + "="*60)
        logger.info("FEATURE EXTRACTION")
        logger.info("="*60)
        if not self.extract_features():
            logger.error("Feature extraction failed!")
            return False

        # Step 4: Train models
        if not self.train_models():
            logger.error("Model training failed!")
            return False

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Models saved in: {self.models_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Review the classification reports above")
        logger.info("2. If F1 scores are low, consider:")
        logger.info("   - Labeling more tracks (target: 200-300)")
        logger.info("   - Balancing class distribution")
        logger.info("   - Feature engineering improvements")
        logger.info("3. Run inference on your full library")

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train music classification models')
    parser.add_argument('--data-dir', default='data', help='Directory containing labels')
    parser.add_argument('--models-dir', default='models', help='Directory to save models')
    parser.add_argument('--labels-file', default='manual_labels.json',
                       help='Name of labels file')

    args = parser.parse_args()

    # Create trainer
    trainer = MusicClassifierTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    # Run training
    success = trainer.run_full_training()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
