#!/usr/bin/env python3
"""
Main entry point for Music Classifier.
Provides a simple CLI for running different phases.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Lazy imports to avoid dependency errors when running simple commands


def main():
    parser = argparse.ArgumentParser(
        description='Music Classifier - ML-powered mood and energy classification for DJ libraries',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Phase 1: Labeling tool
    label_parser = subparsers.add_parser('label', help='Launch labeling webapp')

    # Phase 2: Extract features
    features_parser = subparsers.add_parser('extract', help='Extract features from labeled tracks')

    # Phase 2: Train models
    train_parser = subparsers.add_parser('train', help='Train classification models')

    # Phase 3: Inference
    predict_parser = subparsers.add_parser('predict', help='Predict labels for music library')
    predict_parser.add_argument('music_dir', help='Path to music library directory')
    predict_parser.add_argument('--extensions', nargs='+',
                                default=['*.mp3', '*.wav', '*.flac', '*.m4a'],
                                help='Audio file extensions to process')

    # Phase 4: Rekordbox integration
    rekordbox_parser = subparsers.add_parser('rekordbox', help='Integrate with Rekordbox XML')
    rekordbox_parser.add_argument('xml_path', help='Path to Rekordbox XML file')
    rekordbox_parser.add_argument('--predictions', help='Path to predictions JSON (optional)')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'label':
            print("Phase 1: Launching labeling webapp...")
            print("\nPlease use the Makefile instead:")
            print("  make label")
            print("\nThis will open Safari at http://localhost:5001")
            return

        elif args.command == 'extract':
            print("Phase 2: Extracting features from labeled tracks...")
            from src.phase2.feature_extraction import extract_features_for_labeled_tracks
            extract_features_for_labeled_tracks()

        elif args.command == 'train':
            print("Phase 2: Training classification models...")
            from src.phase2.train_models import main as train_models
            train_models()

        elif args.command == 'predict':
            print(f"Phase 3: Predicting labels for music library at {args.music_dir}...")
            from src.phase3.inference import process_music_library
            process_music_library(args.music_dir, args.extensions)

        elif args.command == 'rekordbox':
            print(f"Phase 4: Integrating predictions with Rekordbox XML...")
            from src.phase4.rekordbox_integration import integrate_predictions_with_rekordbox
            integrate_predictions_with_rekordbox(args.xml_path, args.predictions)

        print("\n✓ Command completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
