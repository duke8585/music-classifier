"""
Rebuild sample_tracks.json to include all labeled tracks with their current paths.
"""

import json
import os

# Paths
MANUAL_LABELS_PATH = "data/manual_labels.json"
SAMPLE_TRACKS_PATH = "data/sample_tracks.json"
MUSIC_ROOT = "/Users/maxr/iCloudDrive/bandcamp_exports"

# Load labels
with open(MANUAL_LABELS_PATH, "r") as f:
    labels = json.load(f)

print(f"Found {len(labels)} labeled tracks")

# Search for each labeled file
sample_tracks = []
found_count = 0
not_found = []

for filename in labels.keys():
    # Search for this file in the music directory
    found = False
    for root, dirs, files in os.walk(MUSIC_ROOT):
        if filename in files:
            full_path = os.path.join(root, filename)
            # Extract month from path
            rel_path = os.path.relpath(root, MUSIC_ROOT)
            month = rel_path.split("/")[0] if "/" in rel_path else rel_path

            sample_tracks.append(
                {
                    "path": full_path,
                    "filename": filename,
                    "title": os.path.splitext(filename)[0],
                    "month": month,
                }
            )
            found_count += 1
            found = True
            break

    if not found:
        not_found.append(filename)

print(f"✓ Found {found_count} tracks")
if not_found:
    print(f"✗ Not found: {len(not_found)} tracks")
    print("First 5 missing tracks:")
    for f in not_found[:5]:
        print(f"  - {f}")

# Save updated sample_tracks.json
with open(SAMPLE_TRACKS_PATH, "w") as f:
    json.dump(sample_tracks, f, indent=2)

print(f"\nSaved {len(sample_tracks)} tracks to {SAMPLE_TRACKS_PATH}")
