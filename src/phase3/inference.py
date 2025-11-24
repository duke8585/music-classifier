"""
Batch inference system for classifying music tracks using RELABEL v2.0.
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
    """Load trained models and perform inference on new tracks using RELABEL v2.0."""

    def __init__(self):
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
        self.feature_extractor = FeatureExtractor()

    def load_models(self):
        """Load trained models and preprocessing objects for RELABEL v2.0."""
        print("Loading RELABEL v2.0 models...")

        # Define paths for all 4 models + scaler + encoders
        energy_model_path = os.path.join(MODELS_DIR, "energy_model.pkl")
        bass_weight_model_path = os.path.join(MODELS_DIR, "bass_weight_model.pkl")
        rhythm_model_path = os.path.join(MODELS_DIR, "rhythm_model.pkl")
        vibe_model_path = os.path.join(MODELS_DIR, "vibe_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        energy_encoder_path = os.path.join(MODELS_DIR, "energy_encoder.pkl")
        bass_weight_encoder_path = os.path.join(MODELS_DIR, "bass_weight_encoder.pkl")
        rhythm_encoder_path = os.path.join(MODELS_DIR, "rhythm_encoder.pkl")
        vibe_encoder_path = os.path.join(MODELS_DIR, "vibe_encoder.pkl")
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")

        # Check all required files exist
        required_files = [
            energy_model_path,
            bass_weight_model_path,
            rhythm_model_path,
            vibe_model_path,
            scaler_path,
            energy_encoder_path,
            bass_weight_encoder_path,
            rhythm_encoder_path,
            vibe_encoder_path,
        ]

        if not all(os.path.exists(p) for p in required_files):
            raise FileNotFoundError(
                "Model files not found. Please train models first with RELABEL v2.0."
            )

        # Load all models
        self.energy_model = joblib.load(energy_model_path)
        self.bass_weight_model = joblib.load(bass_weight_model_path)
        self.rhythm_model = joblib.load(rhythm_model_path)
        self.vibe_model = joblib.load(vibe_model_path)

        # Load scaler and encoders
        self.scaler = joblib.load(scaler_path)
        self.energy_encoder = joblib.load(energy_encoder_path)
        self.bass_weight_encoder = joblib.load(bass_weight_encoder_path)
        self.rhythm_encoder = joblib.load(rhythm_encoder_path)
        self.vibe_encoder = joblib.load(vibe_encoder_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            self.feature_names = metadata["feature_names"]

        print("Models loaded successfully (RELABEL v2.0)")

    def predict_track(self, audio_path):
        """
        Predict all 4 dimensions for a single track (RELABEL v2.0).

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with predictions and confidence scores for all dimensions
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

        # Helper function to predict single-label dimension
        def predict_dimension(model, encoder, dimension_name):
            # Predict encoded label
            pred_encoded = model.predict(feature_vector_scaled)[0]
            pred_label = encoder.inverse_transform([pred_encoded])[0]

            # Get confidence and top-3
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_vector_scaled)[0]
                confidence = float(np.max(proba))

                # Get top 3 predictions
                proba_sorted = np.argsort(proba)[::-1][:3]
                top3 = [
                    {
                        "label": encoder.inverse_transform([idx])[0],
                        "confidence": float(proba[idx]),
                    }
                    for idx in proba_sorted
                ]
            else:
                confidence = 0.5
                top3 = [{"label": pred_label, "confidence": 0.5}]

            return {"label": pred_label, "confidence": confidence, "top_3": top3}

        # Predict all 4 dimensions
        result = {
            "energy": predict_dimension(self.energy_model, self.energy_encoder, "energy"),
            "bass_weight": predict_dimension(
                self.bass_weight_model, self.bass_weight_encoder, "bass_weight"
            ),
            "rhythm": predict_dimension(self.rhythm_model, self.rhythm_encoder, "rhythm"),
            "vibe": predict_dimension(self.vibe_model, self.vibe_encoder, "vibe"),
        }

        return result

    def predict_batch(self, audio_paths, output_format="json"):
        """
        Predict labels for multiple tracks (RELABEL v2.0).

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
                            "energy": result["energy"]["label"],
                            "energy_confidence": result["energy"]["confidence"],
                            "bass_weight": result["bass_weight"]["label"],
                            "bass_weight_confidence": result["bass_weight"]["confidence"],
                            "rhythm": result["rhythm"]["label"],
                            "rhythm_confidence": result["rhythm"]["confidence"],
                            "vibe": result["vibe"]["label"],
                            "vibe_confidence": result["vibe"]["confidence"],
                            # Include full results for detailed analysis
                            "energy_top3": result["energy"]["top_3"],
                            "bass_weight_top3": result["bass_weight"]["top_3"],
                            "rhythm_top3": result["rhythm"]["top_3"],
                            "vibe_top3": result["vibe"]["top_3"],
                        }
                    )
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        return predictions

    def save_predictions(self, predictions, output_name="predictions"):
        """
        Save predictions to JSON and CSV formats (RELABEL v2.0).

        Args:
            predictions: List of prediction dictionaries
            output_name: Base name for output files
        """
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)

        # Save as JSON (includes top-3 predictions for all dimensions)
        json_path = os.path.join(PREDICTIONS_DIR, f"{output_name}.json")
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to {json_path}")

        # Save as CSV for easy review (top predictions + confidences)
        csv_data = []
        for pred in predictions:
            csv_data.append(
                {
                    "filename": pred["filename"],
                    "path": pred["path"],
                    "energy": pred["energy"],
                    "energy_conf": f"{pred['energy_confidence']:.3f}",
                    "bass_weight": pred["bass_weight"],
                    "bass_conf": f"{pred['bass_weight_confidence']:.3f}",
                    "rhythm": pred["rhythm"],
                    "rhythm_conf": f"{pred['rhythm_confidence']:.3f}",
                    "vibe": pred["vibe"],
                    "vibe_conf": f"{pred['vibe_confidence']:.3f}",
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
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY - RELABEL v2.0")
    print("=" * 70)
    print(f"Total tracks processed: {len(predictions)}")

    # Energy distribution
    energy_counts = {}
    for pred in predictions:
        energy = pred["energy"]
        energy_counts[energy] = energy_counts.get(energy, 0) + 1

    print("\nEnergy Distribution:")
    for energy, count in sorted(energy_counts.items()):
        print(f"  {energy}: {count} ({count / len(predictions) * 100:.1f}%)")

    # Bass Weight distribution
    bass_counts = {}
    for pred in predictions:
        bass = pred["bass_weight"]
        bass_counts[bass] = bass_counts.get(bass, 0) + 1

    print("\nBass Weight Distribution:")
    for bass, count in sorted(bass_counts.items()):
        print(f"  {bass}: {count} ({count / len(predictions) * 100:.1f}%)")

    # Rhythm distribution
    rhythm_counts = {}
    for pred in predictions:
        rhythm = pred["rhythm"]
        rhythm_counts[rhythm] = rhythm_counts.get(rhythm, 0) + 1

    print("\nRhythm Distribution:")
    for rhythm, count in sorted(rhythm_counts.items()):
        print(f"  {rhythm}: {count} ({count / len(predictions) * 100:.1f}%)")

    # Vibe distribution
    vibe_counts = {}
    for pred in predictions:
        vibe = pred["vibe"]
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
