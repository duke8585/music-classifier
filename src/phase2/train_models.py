"""
Train classification models for RELABEL v2.0: energy, bass_weight, rhythm, and vibe.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import (
    BASS_WEIGHT_LABELS,
    ENERGY_LABELS,
    FEATURES_DIR,
    MAX_DEPTH,
    MIN_F1_BASS_WEIGHT,
    MIN_F1_ENERGY,
    MIN_F1_RHYTHM,
    MIN_F1_VIBE,
    MODELS_DIR,
    N_ESTIMATORS,
    RANDOM_STATE,
    RHYTHM_LABELS,
    TEST_SIZE,
    VIBE_LABELS,
)

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using Random Forest only.")


class ModelTrainer:
    """Train and evaluate classification models for RELABEL v2.0."""

    def __init__(self, features_file=None):
        self.energy_model = None
        self.bass_weight_model = None
        self.rhythm_model = None
        self.vibe_model = None
        self.scaler = None
        self.energy_encoder = None
        self.bass_weight_encoder = None
        self.rhythm_encoder = None
        self.vibe_encoder = None
        self.feature_names = None
        self.features_file = features_file

    def load_training_data(self):
        """
        Load features and labels from JSON file.

        Returns:
            X: Feature matrix
            y_energy: Energy labels
            y_bass_weight: Bass weight labels
            y_rhythm: Rhythm labels
            y_vibe: Vibe labels
            track_ids: Track identifiers
        """
        if self.features_file:
            features_path = self.features_file
        else:
            features_path = os.path.join(FEATURES_DIR, "training_features.json")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Training features not found: {features_path}")

        with open(features_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            raise ValueError("No training data available")

        print(f"Loaded {len(data)} training samples")

        # Extract features and labels
        X = []
        y_energy = []
        y_bass_weight = []
        y_rhythm = []
        y_vibe = []
        track_ids = []

        for item in data:
            features = item["features"]
            feature_vector = list(features.values())

            X.append(feature_vector)
            y_energy.append(item["energy"])
            y_bass_weight.append(item["bass_weight"])
            y_rhythm.append(item["rhythm"])
            y_vibe.append(item["vibe"])
            track_ids.append(item["track_id"])

        # Store feature names
        self.feature_names = list(data[0]["features"].keys())

        X = np.array(X)
        y_energy = np.array(y_energy)
        y_bass_weight = np.array(y_bass_weight)
        y_rhythm = np.array(y_rhythm)
        y_vibe = np.array(y_vibe)

        return X, y_energy, y_bass_weight, y_rhythm, y_vibe, track_ids

    def train_energy_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train energy level classifier.

        Returns:
            model: Trained classifier
            metrics: Performance metrics
        """
        print("\n=== Training Energy Classifier ===")

        # Encode labels to ensure they're consecutive integers starting from 0
        self.energy_encoder = LabelEncoder()
        self.energy_encoder.fit(ENERGY_LABELS)  # Fit on all possible labels
        y_train_encoded = self.energy_encoder.transform(y_train)
        y_test_encoded = self.energy_encoder.transform(y_test)

        label_mapping = dict(
            zip(
                self.energy_encoder.classes_,
                self.energy_encoder.transform(self.energy_encoder.classes_),
            )
        )
        print(f"Label mapping: {label_mapping}")

        # Check label diversity in training set
        unique_train_labels = np.unique(y_train_encoded)
        n_train_classes = len(unique_train_labels)

        unique_label_names = self.energy_encoder.inverse_transform(unique_train_labels)
        print(f"Training set has {n_train_classes} unique classes: {unique_label_names}")

        # Use Random Forest for small/imbalanced datasets, XGBoost for larger ones
        use_rf = n_train_classes == 1 or len(y_train) < 20

        if use_rf:
            print("Using Random Forest Classifier (small/imbalanced dataset)")
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif XGBOOST_AVAILABLE:
            print("Using XGBoost Classifier")
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )
        else:
            print("Using Random Forest Classifier")
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )

        # Train
        model.fit(X_train, y_train_encoded)

        # Evaluate
        y_pred_encoded = model.predict(X_test)

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        # Get unique labels present in test set for proper F1 calculation
        unique_labels_encoded = np.unique(np.concatenate([y_test_encoded, y_pred_encoded]))

        f1_macro = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            labels=unique_labels_encoded,
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            labels=unique_labels_encoded,
            zero_division=0,
        )

        print("\nEnergy Classifier Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")

        print("\nClassification Report:")
        # Decode labels for display
        present_label_names = self.energy_encoder.inverse_transform(unique_labels_encoded)
        print(
            classification_report(
                y_test_encoded,
                y_pred_encoded,
                labels=unique_labels_encoded,
                target_names=present_label_names,
                zero_division=0,
            )
        )

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]

            print("\nTop 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_names[idx]}: {importance[idx]:.4f}")

        metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

        return model, metrics

    def train_bass_weight_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train bass weight classifier.

        Returns:
            model: Trained classifier
            metrics: Performance metrics
        """
        print("\n=== Training Bass Weight Classifier ===")

        # Encode labels
        self.bass_weight_encoder = LabelEncoder()
        self.bass_weight_encoder.fit(BASS_WEIGHT_LABELS)
        y_train_encoded = self.bass_weight_encoder.transform(y_train)
        y_test_encoded = self.bass_weight_encoder.transform(y_test)

        # Check class distribution
        unique_train_labels = np.unique(y_train_encoded)
        unique_label_names = self.bass_weight_encoder.inverse_transform(unique_train_labels)
        print(f"Training set has {len(unique_train_labels)} classes: {unique_label_names}")

        # Train
        use_rf = len(unique_train_labels) == 1 or len(y_train) < 20
        if use_rf:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )

        model.fit(X_train, y_train_encoded)

        # Evaluate
        y_pred_encoded = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        unique_labels_encoded = np.unique(np.concatenate([y_test_encoded, y_pred_encoded]))
        f1_macro = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            labels=unique_labels_encoded,
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            labels=unique_labels_encoded,
            zero_division=0,
        )

        print("\nBass Weight Classifier Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")

        print("\nClassification Report:")
        present_label_names = self.bass_weight_encoder.inverse_transform(unique_labels_encoded)
        print(
            classification_report(
                y_test_encoded,
                y_pred_encoded,
                labels=unique_labels_encoded,
                target_names=present_label_names,
                zero_division=0,
            )
        )

        metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
        return model, metrics

    def train_rhythm_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train rhythm classifier.

        Returns:
            model: Trained classifier
            metrics: Performance metrics
        """
        print("\n=== Training Rhythm Classifier ===")

        # Encode labels
        self.rhythm_encoder = LabelEncoder()
        self.rhythm_encoder.fit(RHYTHM_LABELS)
        y_train_encoded = self.rhythm_encoder.transform(y_train)
        y_test_encoded = self.rhythm_encoder.transform(y_test)

        # Check class distribution
        unique_train_labels = np.unique(y_train_encoded)
        unique_label_names = self.rhythm_encoder.inverse_transform(unique_train_labels)
        print(f"Training set has {len(unique_train_labels)} classes: {unique_label_names}")

        # Train
        use_rf = len(unique_train_labels) == 1 or len(y_train) < 20
        if use_rf:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )

        model.fit(X_train, y_train_encoded)

        # Evaluate
        y_pred_encoded = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        unique_labels_encoded = np.unique(np.concatenate([y_test_encoded, y_pred_encoded]))
        f1_macro = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            labels=unique_labels_encoded,
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            labels=unique_labels_encoded,
            zero_division=0,
        )

        print("\nRhythm Classifier Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")

        print("\nClassification Report:")
        present_label_names = self.rhythm_encoder.inverse_transform(unique_labels_encoded)
        print(
            classification_report(
                y_test_encoded,
                y_pred_encoded,
                labels=unique_labels_encoded,
                target_names=present_label_names,
                zero_division=0,
            )
        )

        metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
        return model, metrics

    def train_vibe_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train vibe classifier (single-label).

        Returns:
            model: Trained classifier
            metrics: Performance metrics
        """
        print("\n=== Training Vibe Classifier ===")

        # Encode labels
        self.vibe_encoder = LabelEncoder()
        self.vibe_encoder.fit(VIBE_LABELS)
        y_train_encoded = self.vibe_encoder.transform(y_train)
        y_test_encoded = self.vibe_encoder.transform(y_test)

        # Check class distribution
        unique_train_labels = np.unique(y_train_encoded)
        unique_label_names = self.vibe_encoder.inverse_transform(unique_train_labels)
        print(f"Training set has {len(unique_train_labels)} classes: {unique_label_names}")

        # Train
        use_rf = len(unique_train_labels) == 1 or len(y_train) < 20
        if use_rf:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1
            )

        model.fit(X_train, y_train_encoded)

        # Evaluate
        y_pred_encoded = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

        unique_labels_encoded = np.unique(np.concatenate([y_test_encoded, y_pred_encoded]))
        f1_macro = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            labels=unique_labels_encoded,
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            labels=unique_labels_encoded,
            zero_division=0,
        )

        print("\nVibe Classifier Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")

        print("\nClassification Report:")
        present_label_names = self.vibe_encoder.inverse_transform(unique_labels_encoded)
        print(
            classification_report(
                y_test_encoded,
                y_pred_encoded,
                labels=unique_labels_encoded,
                target_names=present_label_names,
                zero_division=0,
            )
        )

        metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
        return model, metrics

    def train(self):
        """
        Main training pipeline for RELABEL v2.0 (4 models).
        """
        # Load data
        X, y_energy, y_bass_weight, y_rhythm, y_vibe, track_ids = self.load_training_data()

        print(f"\nDataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Energy label distribution: {np.unique(y_energy, return_counts=True)}")
        print(f"Bass weight label distribution: {np.unique(y_bass_weight, return_counts=True)}")
        print(f"Rhythm label distribution: {np.unique(y_rhythm, return_counts=True)}")
        print(f"Vibe label distribution: {np.unique(y_vibe, return_counts=True)}")

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data (stratify by energy for consistency)
        (
            X_train,
            X_test,
            y_energy_train,
            y_energy_test,
            y_bass_weight_train,
            y_bass_weight_test,
            y_rhythm_train,
            y_rhythm_test,
            y_vibe_train,
            y_vibe_test,
        ) = train_test_split(
            X_scaled,
            y_energy,
            y_bass_weight,
            y_rhythm,
            y_vibe,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_energy,
        )

        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Train all 4 classifiers
        self.energy_model, energy_metrics = self.train_energy_classifier(
            X_train, y_energy_train, X_test, y_energy_test
        )

        self.bass_weight_model, bass_weight_metrics = self.train_bass_weight_classifier(
            X_train, y_bass_weight_train, X_test, y_bass_weight_test
        )

        self.rhythm_model, rhythm_metrics = self.train_rhythm_classifier(
            X_train, y_rhythm_train, X_test, y_rhythm_test
        )

        self.vibe_model, vibe_metrics = self.train_vibe_classifier(
            X_train, y_vibe_train, X_test, y_vibe_test
        )

        # Save models
        self.save_models()

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY - RELABEL v2.0")
        print("=" * 70)
        energy_f1 = energy_metrics["f1_macro"]
        bass_f1 = bass_weight_metrics["f1_macro"]
        rhythm_f1 = rhythm_metrics["f1_macro"]
        vibe_f1 = vibe_metrics["f1_macro"]
        print(f"Energy Classifier F1 (macro):      {energy_f1:.3f} (target: {MIN_F1_ENERGY:.2f})")
        print(
            f"Bass Weight Classifier F1 (macro): {bass_f1:.3f} (target: {MIN_F1_BASS_WEIGHT:.2f})"
        )
        print(f"Rhythm Classifier F1 (macro):      {rhythm_f1:.3f} (target: {MIN_F1_RHYTHM:.2f})")
        print(f"Vibe Classifier F1 (macro):        {vibe_f1:.3f} (target: {MIN_F1_VIBE:.2f})")
        print("=" * 70)

        # Check if targets are met
        targets_met = (
            energy_metrics["f1_macro"] >= MIN_F1_ENERGY
            and bass_weight_metrics["f1_macro"] >= MIN_F1_BASS_WEIGHT
            and rhythm_metrics["f1_macro"] >= MIN_F1_RHYTHM
            and vibe_metrics["f1_macro"] >= MIN_F1_VIBE
        )

        if targets_met:
            print("✓ All F1 targets met!")
        else:
            print(
                "⚠ Some F1 targets not met. Consider labeling more data or tuning hyperparameters."
            )

        return energy_metrics, bass_weight_metrics, rhythm_metrics, vibe_metrics

    def save_models(self):
        """Save trained models and preprocessing objects for RELABEL v2.0."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save all 4 models
        joblib.dump(self.energy_model, os.path.join(MODELS_DIR, "energy_model.pkl"))
        joblib.dump(self.bass_weight_model, os.path.join(MODELS_DIR, "bass_weight_model.pkl"))
        joblib.dump(self.rhythm_model, os.path.join(MODELS_DIR, "rhythm_model.pkl"))
        joblib.dump(self.vibe_model, os.path.join(MODELS_DIR, "vibe_model.pkl"))

        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(self.energy_encoder, os.path.join(MODELS_DIR, "energy_encoder.pkl"))
        joblib.dump(self.bass_weight_encoder, os.path.join(MODELS_DIR, "bass_weight_encoder.pkl"))
        joblib.dump(self.rhythm_encoder, os.path.join(MODELS_DIR, "rhythm_encoder.pkl"))
        joblib.dump(self.vibe_encoder, os.path.join(MODELS_DIR, "vibe_encoder.pkl"))

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "energy_labels": ENERGY_LABELS,
            "bass_weight_labels": BASS_WEIGHT_LABELS,
            "rhythm_labels": RHYTHM_LABELS,
            "vibe_labels": VIBE_LABELS,
            "taxonomy_version": "RELABEL v2.0",
        }

        with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModels saved to {MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RELABEL v2.0 classification models (energy, bass_weight, rhythm, vibe)"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Path to training features JSON file (default: data/features/training_features.json)",
    )

    args = parser.parse_args()

    trainer = ModelTrainer(features_file=args.features_file)
    trainer.train()


if __name__ == "__main__":
    main()
