#!/usr/bin/env python3
"""
Convert AIFF samples to WAV for better browser compatibility.
Creates temporary WAV files in /tmp for playback.
"""

import json
import subprocess
from pathlib import Path

SAMPLE_TRACKS_FILE = Path("data/sample_tracks.json")
WAV_DIR = Path("/tmp/music_classifier_wavs")

def convert_aiff_to_wav(aiff_path, wav_path):
    """Convert AIFF to WAV using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', str(aiff_path),
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            str(wav_path)
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {aiff_path}: {e}")
        return False
    except FileNotFoundError:
        print("ffmpeg not found. Install with: brew install ffmpeg")
        return False

def main():
    # Create WAV directory
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample tracks
    with open(SAMPLE_TRACKS_FILE, 'r') as f:
        tracks = json.load(f)

    print(f"Converting {len(tracks)} AIFF files to WAV...")

    converted = 0
    for track in tracks:
        aiff_path = Path(track['path'])
        if not aiff_path.exists():
            print(f"File not found: {aiff_path}")
            continue

        # Create WAV filename
        wav_filename = aiff_path.stem + '.wav'
        wav_path = WAV_DIR / wav_filename

        if wav_path.exists():
            print(f"Already exists: {wav_filename}")
            track['wav_path'] = str(wav_path)
            converted += 1
            continue

        print(f"Converting: {aiff_path.name}...")
        if convert_aiff_to_wav(aiff_path, wav_path):
            track['wav_path'] = str(wav_path)
            converted += 1

    # Save updated tracks
    with open(SAMPLE_TRACKS_FILE, 'w') as f:
        json.dump(tracks, f, indent=2)

    print(f"\nConverted {converted}/{len(tracks)} files")
    print(f"WAV files saved to: {WAV_DIR}")

if __name__ == '__main__':
    main()
