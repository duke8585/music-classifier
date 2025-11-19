#!/usr/bin/env python3
"""
Generate a random sample of AIFF files from Bandcamp exports for labeling.
"""

import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.config import EXCLUDE_PATTERNS, SAMPLE_BATCH_SIZE

# Source directory
MUSIC_DIR = Path("/Users/maxr/iCloudDrive/bandcamp_exports")
OUTPUT_FILE = Path("data/sample_tracks.json")


def should_exclude(file_path: Path) -> bool:
    """
    Check if a file should be excluded based on EXCLUDE_PATTERNS.
    Patterns are matched against the full file path (case-insensitive).
    """
    if not EXCLUDE_PATTERNS:
        return False

    full_path = str(file_path)
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, full_path, re.IGNORECASE):
            return True
    return False


def backup_existing_sample():
    """
    Backup existing sample_tracks.json to sample_tracks.bak.N.json.

    Creates an incremental backup (e.g., .bak.1.json, .bak.2.json, etc.).
    Finds the highest existing backup number and creates the next one.
    """
    if not OUTPUT_FILE.exists():
        return

    # Find existing backup files with pattern sample_tracks.bak.N.json
    backup_pattern = OUTPUT_FILE.parent / "sample_tracks.bak.*.json"
    existing_backups = list(OUTPUT_FILE.parent.glob("sample_tracks.bak.*.json"))

    # Extract backup numbers and find the max
    backup_numbers = []
    for backup_file in existing_backups:
        # Extract number from filename like "sample_tracks.bak.1.json"
        stem = backup_file.stem  # "sample_tracks.bak.1"
        parts = stem.split(".")
        if len(parts) >= 3 and parts[-1].isdigit():
            backup_numbers.append(int(parts[-1]))

    # Determine next backup number
    next_num = max(backup_numbers, default=0) + 1

    # Create backup with incremental number
    backup_path = OUTPUT_FILE.parent / f"sample_tracks.bak.{next_num}.json"
    shutil.copy2(OUTPUT_FILE, backup_path)
    print(f"Backed up existing sample to: {backup_path.name}")


def main():
    # Backup existing sample file if it exists
    backup_existing_sample()

    # Find all AIFF files
    print(f"Scanning {MUSIC_DIR} for AIFF files...")
    all_aiff_files = list(MUSIC_DIR.glob("**/*.aiff"))

    print(f"Found {len(all_aiff_files)} AIFF files")

    # Filter out excluded files
    if EXCLUDE_PATTERNS:
        print(f"Applying {len(EXCLUDE_PATTERNS)} exclusion pattern(s)...")
        aiff_files = [f for f in all_aiff_files if not should_exclude(f)]
        excluded_count = len(all_aiff_files) - len(aiff_files)
        print(f"Excluded {excluded_count} files matching patterns")
        print(f"Remaining: {len(aiff_files)} files")
    else:
        aiff_files = all_aiff_files

    if len(aiff_files) == 0:
        print("No AIFF files found after filtering!")
        return

    # Select random sample (SAMPLE_BATCH_SIZE files, or all if less)
    sample_size = min(SAMPLE_BATCH_SIZE, len(aiff_files))
    sample_files = random.sample(aiff_files, sample_size)

    # Extract track info from filenames
    tracks = []
    for file_path in sample_files:
        # Parse filename: "Artist - Title – Label - Track.aiff"
        filename = file_path.stem  # Remove .aiff extension

        tracks.append(
            {
                "path": str(file_path),
                "filename": file_path.name,
                "title": filename,
                "month": file_path.parent.name,
            }
        )

    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)

    print(f"\nSelected {len(tracks)} random tracks:")
    for i, track in enumerate(tracks, 1):
        print(f"{i}. {track['filename']}")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
