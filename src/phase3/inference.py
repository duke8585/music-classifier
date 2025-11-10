"""
Batch inference system for classifying music tracks.
Processes large music libraries and outputs predictions with confidence scores.
"""

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import MODELS_DIR, PREDICTIONS_DIR
from src.phase2.feature_extraction import FeatureExtractor


class MusicClassifier:
    """Load trained models and perform inference on new tracks."""

    def __init__(self):
        self.energy_model = None
        self.vibe_model = None
        self.scaler = None
        self.mlb = None
        self.feature_names = None
        self.feature_extractor = FeatureExtractor()

    def load_models(self):
        """Load trained models and preprocessing objects."""
        print("Loading models...")

        # Load models
        energy_model_path = os.path.join(MODELS_DIR, "energy_model.pkl")
        vibe_model_path = os.path.join(MODELS_DIR, "vibe_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        mlb_path = os.path.join(MODELS_DIR, "mlb.pkl")
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")

        if not all(
            os.path.exists(p) for p in [energy_model_path, vibe_model_path, scaler_path, mlb_path]
        ):
            raise FileNotFoundError("Model files not found. Please train models first.")

        self.energy_model = joblib.load(energy_model_path)
        self.vibe_model = joblib.load(vibe_model_path)
        self.scaler = joblib.load(scaler_path)
        self.mlb = joblib.load(mlb_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            self.feature_names = metadata["feature_names"]

        print("Models loaded successfully")

    def predict_track(self, audio_path):
        """
        Predict energy and vibe labels for a single track.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with predictions and confidence scores
        """
        # Extract features
        features = self.feature_extractor.extract_features(audio_path)

        if features is None:
            return None

        # Ensure features are in correct order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)

        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Predict energy
        energy_pred = self.energy_model.predict(feature_vector_scaled)[0]

        # Get energy confidence (probability)
        if hasattr(self.energy_model, "predict_proba"):
            energy_proba = self.energy_model.predict_proba(feature_vector_scaled)[0]
            energy_confidence = float(np.max(energy_proba))

            # Get top 3 energy predictions
            energy_proba_sorted = np.argsort(energy_proba)[::-1][:3]
            energy_top3 = [
                {"label": self.energy_model.classes_[idx], "confidence": float(energy_proba[idx])}
                for idx in energy_proba_sorted
            ]
        else:
            energy_confidence = 0.5
            energy_top3 = [{"label": energy_pred, "confidence": 0.5}]

        # Predict vibes
        vibe_pred_bin = self.vibe_model.predict(feature_vector_scaled)[0]

        # Get vibe probabilities if available
        vibe_predictions = []
        if hasattr(self.vibe_model, "predict_proba"):
            # For MultiOutputClassifier, get probabilities for each vibe
            for i, vibe in enumerate(self.mlb.classes_):
                # Try to get probability from individual estimator
                try:
                    estimator = self.vibe_model.estimators_[i]
                    if hasattr(estimator, "predict_proba"):
                        prob = estimator.predict_proba(feature_vector_scaled)[0][1]
                    else:
                        prob = 0.5
                except Exception:
                    prob = 0.5

                vibe_predictions.append(
                    {"label": vibe, "predicted": bool(vibe_pred_bin[i]), "confidence": float(prob)}
                )
        else:
            for i, vibe in enumerate(self.mlb.classes_):
                vibe_predictions.append(
                    {"label": vibe, "predicted": bool(vibe_pred_bin[i]), "confidence": 0.5}
                )

        # Get predicted vibes (those with prediction = True)
        predicted_vibes = [v["label"] for v in vibe_predictions if v["predicted"]]

        result = {
            "energy": {
                "predicted": energy_pred,
                "confidence": energy_confidence,
                "top_3": energy_top3,
            },
            "vibes": {"predicted": predicted_vibes, "all_vibes": vibe_predictions},
        }

        return result

    def predict_batch(self, audio_paths, output_format="json"):
        """
        Predict labels for multiple tracks.

        Args:
            audio_paths: List of audio file paths
            output_format: 'json' or 'csv'

        Returns:
            List of predictions
        """
        predictions = []

        for audio_path in tqdm(audio_paths, desc="Processing tracks"):
            try:
                result = self.predict_track(audio_path)

                if result:
                    predictions.append(
                        {
                            "path": audio_path,
                            "filename": os.path.basename(audio_path),
                            "energy": result["energy"]["predicted"],
                            "energy_confidence": result["energy"]["confidence"],
                            "vibes": result["vibes"]["predicted"],
                            "energy_top3": result["energy"]["top_3"],
                            "vibe_details": result["vibes"]["all_vibes"],
                        }
                    )
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        return predictions

    def save_predictions(self, predictions, output_name="predictions"):
        """
        Save predictions to JSON and CSV formats.

        Args:
            predictions: List of prediction dictionaries
            output_name: Base name for output files
        """
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(PREDICTIONS_DIR, f"{output_name}.json")
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to {json_path}")

        # Save as CSV for easy review
        csv_data = []
        for pred in predictions:
            csv_data.append(
                {
                    "filename": pred["filename"],
                    "path": pred["path"],
                    "energy": pred["energy"],
                    "energy_confidence": f"{pred['energy_confidence']:.3f}",
                    "vibes": ", ".join(pred["vibes"]),
                    "num_vibes": len(pred["vibes"]),
                }
            )

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(PREDICTIONS_DIR, f"{output_name}.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved CSV to {csv_path}")


def process_music_library(music_dir, file_extensions=["*.mp3", "*.wav", "*.flac", "*.m4a"]):
    """
    Process all music files in a directory.

    Args:
        music_dir: Path to music library
        file_extensions: List of audio file extensions to process
    """
    # Find all audio files
    audio_paths = []
    for ext in file_extensions:
        pattern = os.path.join(music_dir, "**", ext)
        audio_paths.extend(glob.glob(pattern, recursive=True))

    print(f"Found {len(audio_paths)} audio files")

    if not audio_paths:
        print("No audio files found!")
        return

    # Load classifier and predict
    classifier = MusicClassifier()
    classifier.load_models()

    predictions = classifier.predict_batch(audio_paths)

    # Save results
    classifier.save_predictions(predictions)

    # Print summary
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total tracks processed: {len(predictions)}")

    # Energy distribution
    energy_counts = {}
    for pred in predictions:
        energy = pred["energy"]
        energy_counts[energy] = energy_counts.get(energy, 0) + 1

    print("\nEnergy Distribution:")
    for energy, count in sorted(energy_counts.items()):
        print(f"  {energy}: {count} ({count / len(predictions) * 100:.1f}%)")

    # Vibe distribution
    vibe_counts = {}
    for pred in predictions:
        for vibe in pred["vibes"]:
            vibe_counts[vibe] = vibe_counts.get(vibe, 0) + 1

    print("\nVibe Distribution:")
    for vibe, count in sorted(vibe_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vibe}: {count} ({count / len(predictions) * 100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify music tracks")
    parser.add_argument("music_dir", help="Path to music library")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["*.mp3", "*.wav", "*.flac", "*.m4a"],
        help="File extensions to process",
    )

    args = parser.parse_args()

    process_music_library(args.music_dir, args.extensions)
