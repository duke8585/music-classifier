"""
Flask web app for manual music labeling.
Usage: python src/phase1/label_app.py
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# Get project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"

# Add project root to path so we can import config
sys.path.insert(0, str(PROJECT_ROOT))
from config.config import ENERGY_LABELS, VIBE_LABELS

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Configuration
SAMPLE_TRACKS_FILE = PROJECT_ROOT / "data/sample_tracks.json"  # List of tracks to label
LABELS_FILE = PROJECT_ROOT / "data/manual_labels.json"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff"}


def load_labels():
    """Load existing labels from JSON file."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r") as f:
            data = json.load(f)
            # Convert list format to dict format if needed
            if isinstance(data, list):
                return {}
            return data
    return {}


def save_labels(labels):
    """Save labels to JSON file."""
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def get_audio_files():
    """Get list of audio files from sample tracks JSON."""
    if not SAMPLE_TRACKS_FILE.exists():
        return []

    with open(SAMPLE_TRACKS_FILE, "r") as f:
        tracks = json.load(f)

    return tracks


@app.route("/")
def index():
    """Main labeling interface."""
    audio_files = get_audio_files()
    labels = load_labels()

    # Find first unlabeled track or start from beginning
    current_index = 0
    for i, track in enumerate(audio_files):
        if track["filename"] not in labels:
            current_index = i
            break

    return render_template(
        "label.html",
        audio_files=audio_files,
        current_index=current_index,
        energy_labels=ENERGY_LABELS,
        vibe_labels=VIBE_LABELS,
        labels=labels,
        total_files=len(audio_files),
        labeled_count=len(labels),
    )


@app.route("/audio/<int:track_index>")
def serve_audio(track_index):
    """Serve audio files from their absolute paths."""
    tracks = get_audio_files()
    if track_index < 0 or track_index >= len(tracks):
        return "Track not found", 404

    track = tracks[track_index]

    # Prefer WAV if available (better browser support)
    if "wav_path" in track:
        wav_path = Path(track["wav_path"])
        if wav_path.exists():
            return send_from_directory(
                wav_path.parent, wav_path.name, mimetype="audio/wav"
            )

    # Fall back to original AIFF
    track_path = Path(track["path"])
    if not track_path.exists():
        return "File not found", 404

    # Set proper MIME type
    mimetype = "audio/aiff" if track_path.suffix.lower() == ".aiff" else None
    return send_from_directory(track_path.parent, track_path.name, mimetype=mimetype)


@app.route("/save_label", methods=["POST"])
def save_label():
    """Save a label for a track."""
    data = request.json
    filename = data.get("filename")
    energy = data.get("energy")
    vibes = data.get("vibes", [])

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    labels = load_labels()
    labels[filename] = {"energy": energy, "vibes": vibes}
    save_labels(labels)

    return jsonify({"success": True, "labeled_count": len(labels)})


@app.route("/get_label/<path:filename>")
def get_label(filename):
    """Get existing label for a track."""
    labels = load_labels()
    # Handle both old format (filename keys) and new format
    return jsonify(labels.get(filename, {}))


@app.route("/stats")
def stats():
    """Get labeling statistics."""
    audio_files = get_audio_files()
    labels = load_labels()

    return jsonify(
        {
            "total": len(audio_files),
            "labeled": len(labels),
            "remaining": len(audio_files) - len(labels),
        }
    )


if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("Music Labeling Tool")
    print(f"{'=' * 60}")
    print(f"Sample tracks file: {SAMPLE_TRACKS_FILE.absolute()}")
    print(f"Labels file: {LABELS_FILE.absolute()}")

    tracks = get_audio_files()
    print(f"Audio files loaded: {len(tracks)}")

    if len(tracks) == 0:
        print("\nNo tracks found! Run: python generate_sample_list.py")

    print(f"Already labeled: {len(load_labels())}")
    print(f"\nStarting server at http://localhost:5001")
    print(f"{'=' * 60}\n")

    app.run(debug=True, port=5001)
