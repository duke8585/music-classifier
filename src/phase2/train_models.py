"""
Train classification models for energy and vibe prediction.
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
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import (
    ENERGY_LABELS,
    FEATURES_DIR,
    MAX_DEPTH,
    MODELS_DIR,
    N_ESTIMATORS,
    RANDOM_STATE,
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
    """Train and evaluate classification models."""

    def __init__(self, features_file=None):
        self.energy_model = None
        self.vibe_model = None
        self.scaler = None
        self.mlb = None  # MultiLabelBinarizer for vibes
        self.energy_encoder = None  # LabelEncoder for energy labels
        self.feature_names = None
        self.features_file = features_file

    def load_training_data(self):
        """
        Load features and labels from JSON file.

        Returns:
            X: Feature matrix
            y_energy: Energy labels
            y_vibes: Vibe labels (multi-label)
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
        y_vibes = []
        track_ids = []

        for item in data:
            features = item["features"]
            feature_vector = list(features.values())

            X.append(feature_vector)
            y_energy.append(item["energy"])
            y_vibes.append(item["vibes"])
            track_ids.append(item["track_id"])

        # Store feature names
        self.feature_names = list(data[0]["features"].keys())

        X = np.array(X)
        y_energy = np.array(y_energy)

        return X, y_energy, y_vibes, track_ids

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

    def train_vibe_classifier(self, X_train, y_train, X_test, y_test):
        """
        Train vibe classifier (multi-label).

        Returns:
            model: Trained classifier
            metrics: Performance metrics
        """
        print("\n=== Training Vibe Classifier (Multi-label) ===")

        # Binarize labels
        self.mlb = MultiLabelBinarizer()
        y_train_bin = self.mlb.fit_transform(y_train)
        y_test_bin = self.mlb.transform(y_test)

        print(f"Vibe classes: {self.mlb.classes_}")

        # Check dataset size
        use_rf = len(y_train) < 20

        # Train one classifier per vibe (One-vs-Rest approach)
        from sklearn.multioutput import MultiOutputClassifier

        if use_rf:
            print("Using Random Forest Classifier (small dataset)")
            base_model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
            )
            model = MultiOutputClassifier(base_model, n_jobs=-1)
        elif XGBOOST_AVAILABLE:
            print("Using XGBoost Classifier")
            base_model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            )
            model = MultiOutputClassifier(base_model, n_jobs=-1)
        else:
            print("Using Random Forest Classifier")
            base_model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
            )
            model = MultiOutputClassifier(base_model, n_jobs=-1)

        # Train
        model.fit(X_train, y_train_bin)

        # Evaluate
        y_pred_bin = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test_bin, y_pred_bin)
        f1_macro = f1_score(y_test_bin, y_pred_bin, average="macro")
        f1_weighted = f1_score(y_test_bin, y_pred_bin, average="weighted")
        f1_samples = f1_score(y_test_bin, y_pred_bin, average="samples")

        print("\nVibe Classifier Performance:")
        print(f"Accuracy (exact match): {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")
        print(f"F1 (samples): {f1_samples:.3f}")

        # Per-class performance
        print("\nPer-Vibe Performance:")
        for i, vibe in enumerate(self.mlb.classes_):
            vibe_f1 = f1_score(y_test_bin[:, i], y_pred_bin[:, i])
            print(f"  {vibe}: F1 = {vibe_f1:.3f}")

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_samples": f1_samples,
        }

        return model, metrics

    def train(self):
        """
        Main training pipeline.
        """
        # Load data
        X, y_energy, y_vibes, track_ids = self.load_training_data()

        print(f"\nDataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Energy label distribution: {np.unique(y_energy, return_counts=True)}")

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_energy_train, y_energy_test, y_vibes_train, y_vibes_test = (
            train_test_split(
                X_scaled,
                y_energy,
                y_vibes,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_energy,
            )
        )

        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Train energy classifier
        self.energy_model, energy_metrics = self.train_energy_classifier(
            X_train, y_energy_train, X_test, y_energy_test
        )

        # Train vibe classifier
        self.vibe_model, vibe_metrics = self.train_vibe_classifier(
            X_train, y_vibes_train, X_test, y_vibes_test
        )

        # Save models
        self.save_models()

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Energy Classifier F1 (macro): {energy_metrics['f1_macro']:.3f}")
        print(f"Vibe Classifier F1 (macro): {vibe_metrics['f1_macro']:.3f}")

        return energy_metrics, vibe_metrics

    def save_models(self):
        """Save trained models and preprocessing objects."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save models
        joblib.dump(self.energy_model, os.path.join(MODELS_DIR, "energy_model.pkl"))
        joblib.dump(self.vibe_model, os.path.join(MODELS_DIR, "vibe_model.pkl"))
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(self.mlb, os.path.join(MODELS_DIR, "mlb.pkl"))
        joblib.dump(self.energy_encoder, os.path.join(MODELS_DIR, "energy_encoder.pkl"))

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "energy_labels": ENERGY_LABELS,
            "vibe_labels": VIBE_LABELS,
        }

        with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModels saved to {MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train energy and vibe classification models")
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
