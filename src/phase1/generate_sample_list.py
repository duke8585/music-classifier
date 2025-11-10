#!/usr/bin/env python3
"""
Generate a random sample of AIFF files from Bandcamp exports for labeling.
"""

import json
import random
from pathlib import Path

# Source directory
MUSIC_DIR = Path("/Users/maxr/iCloudDrive/bandcamp_exports")
OUTPUT_FILE = Path("data/sample_tracks.json")

def main():
    # Find all AIFF files
    print(f"Scanning {MUSIC_DIR} for AIFF files...")
    aiff_files = list(MUSIC_DIR.glob("**/*.aiff"))

    print(f"Found {len(aiff_files)} AIFF files")

    if len(aiff_files) == 0:
        print("No AIFF files found!")
        return

    # Select random sample (10 files, or all if less than 10)
    sample_size = min(10, len(aiff_files))
    sample_files = random.sample(aiff_files, sample_size)

    # Extract track info from filenames
    tracks = []
    for file_path in sample_files:
        # Parse filename: "Artist - Title – Label - Track.aiff"
        filename = file_path.stem  # Remove .aiff extension

        tracks.append({
            'path': str(file_path),
            'filename': file_path.name,
            'title': filename,
            'month': file_path.parent.name
        })

    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(tracks, f, indent=2)

    print(f"\nSelected {len(tracks)} random tracks:")
    for i, track in enumerate(tracks, 1):
        print(f"{i}. {track['filename']}")

    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
