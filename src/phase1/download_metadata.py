"""
Download and prepare MTG-Jamendo metadata for electronic music tracks.
"""

import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import DATA_DIR, MANUAL_LABELS_DIR


def download_mtg_jamendo_metadata():
    """
    Download MTG-Jamendo dataset metadata and filter for electronic music tracks.
    """
    print("Downloading MTG-Jamendo metadata...")

    # Create data directory if it doesn't exist
    os.makedirs(MANUAL_LABELS_DIR, exist_ok=True)

    # Download the autotagging.tsv file
    url = "https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging.tsv"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save raw metadata
        metadata_path = os.path.join(DATA_DIR, 'mtg_jamendo_raw.tsv')
        with open(metadata_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded metadata to {metadata_path}")

        # Parse TSV
        df = pd.read_csv(metadata_path, sep='\t')
        print(f"Total tracks in dataset: {len(df)}")

        # Filter for electronic music genres
        electronic_genres = [
            'electronic', 'techno', 'house', 'trance', 'ambient',
            'drum and bass', 'dubstep', 'electronica', 'minimal',
            'electro', 'breakbeat', 'downtempo', 'idm'
        ]

        # Check if tags column exists
        if 'tags' in df.columns:
            # Filter tracks with electronic music tags
            electronic_df = df[df['tags'].str.lower().str.contains('|'.join(electronic_genres), na=False)]
        else:
            print("Warning: 'tags' column not found. Using all tracks.")
            electronic_df = df

        print(f"Electronic music tracks found: {len(electronic_df)}")

        # Prepare metadata for labeling
        metadata_for_labeling = []
        for idx, row in electronic_df.head(500).iterrows():  # Take first 500 electronic tracks
            track_info = {
                'track_id': str(row.get('TRACK_ID', idx)),
                'artist': row.get('ARTIST_ID', 'Unknown'),
                'title': row.get('TRACK_ID', f'Track_{idx}'),
                'tags': row.get('tags', ''),
                'path': row.get('PATH', ''),
            }
            metadata_for_labeling.append(track_info)

        # Save filtered metadata
        output_path = os.path.join(MANUAL_LABELS_DIR, 'mtg_jamendo_electronic.json')
        with open(output_path, 'w') as f:
            json.dump(metadata_for_labeling, f, indent=2)

        print(f"Saved {len(metadata_for_labeling)} electronic tracks to {output_path}")

        # Create empty manual labels file
        labels_path = os.path.join(MANUAL_LABELS_DIR, 'manual_labels.json')
        if not os.path.exists(labels_path):
            with open(labels_path, 'w') as f:
                json.dump([], f)
            print(f"Created empty labels file at {labels_path}")

        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading metadata: {e}")
        print("Creating sample metadata for development...")

        # Create sample metadata if download fails
        sample_tracks = []
        for i in range(100):
            sample_tracks.append({
                'track_id': f'sample_{i:04d}',
                'artist': f'Artist_{i % 20}',
                'title': f'Electronic Track {i}',
                'tags': 'electronic, techno, house',
                'path': f'/path/to/track_{i:04d}.mp3'
            })

        output_path = os.path.join(MANUAL_LABELS_DIR, 'mtg_jamendo_electronic.json')
        with open(output_path, 'w') as f:
            json.dump(sample_tracks, f, indent=2)

        labels_path = os.path.join(MANUAL_LABELS_DIR, 'manual_labels.json')
        with open(labels_path, 'w') as f:
            json.dump([], f)

        print(f"Created sample metadata with {len(sample_tracks)} tracks")
        return output_path


if __name__ == '__main__':
    download_mtg_jamendo_metadata()
