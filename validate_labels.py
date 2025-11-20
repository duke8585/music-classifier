#!/usr/bin/env python3
"""
Quick validation script for manual_labels.json
Run this before training to catch issues early
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Valid labels
ENERGY_LABELS = ['warm-up', 'building', 'peak', 'intense', 'closing']
VIBE_LABELS = ['dark', 'melodic', 'hypnotic', 'dubby', 'atmospheric',
               'trippy', 'analog/lofi', 'clean/digital']


def validate_labels(labels_file: str = 'data/manual_labels.json'):
    """Validate labels file and report issues"""

    print("="*60)
    print("LABELS FILE VALIDATION")
    print("="*60)

    # Check if file exists
    labels_path = Path(labels_file)
    if not labels_path.exists():
        print(f"❌ ERROR: Labels file not found at {labels_path}")
        print(f"\nExpected location: {labels_path.absolute()}")
        print(f"\nCreate the file or use: python validate_labels.py path/to/your/labels.json")
        return False

    print(f"✓ Found labels file: {labels_path}")

    # Load JSON
    try:
        with open(labels_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Could not read file: {e}")
        return False

    print(f"✓ Valid JSON format")
    print(f"✓ Total entries: {len(data)}")

    # Validate each entry
    print("\n" + "="*60)
    print("VALIDATING ENTRIES")
    print("="*60)

    valid_tracks = []
    errors = []
    warnings = []

    energy_distribution = Counter()
    vibe_distribution = Counter()

    for track_id, labels in data.items():
        track_valid = True

        # Check required fields
        if 'energy' not in labels:
            errors.append(f"{track_id}: Missing 'energy' field")
            track_valid = False
            continue

        if 'vibe' not in labels:
            errors.append(f"{track_id}: Missing 'vibe' field")
            track_valid = False
            continue

        if 'audio_path' not in labels:
            errors.append(f"{track_id}: Missing 'audio_path' field")
            track_valid = False
            continue

        # Validate energy
        energy = labels['energy']
        if energy not in ENERGY_LABELS:
            errors.append(f"{track_id}: Invalid energy '{energy}'. Valid: {ENERGY_LABELS}")
            track_valid = False
        else:
            energy_distribution[energy] += 1

        # Validate vibe
        vibes = labels['vibe'] if isinstance(labels['vibe'], list) else [labels['vibe']]
        invalid_vibes = [v for v in vibes if v not in VIBE_LABELS]
        if invalid_vibes:
            errors.append(f"{track_id}: Invalid vibe(s) {invalid_vibes}. Valid: {VIBE_LABELS}")
            track_valid = False
        else:
            for v in vibes:
                vibe_distribution[v] += 1

        # Check audio file
        audio_path = Path(labels['audio_path'])
        if not audio_path.exists():
            warnings.append(f"{track_id}: Audio file not found at {audio_path}")
            # Don't mark as invalid yet - maybe files will be added later

        if track_valid:
            valid_tracks.append(track_id)

    # Report errors
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"   {error}")
    else:
        print(f"✓ No validation errors")

    # Report warnings
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            print(f"   {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more")

    # Report valid tracks
    print(f"\n✓ Valid tracks: {len(valid_tracks)}/{len(data)}")

    # Check minimum data requirements
    print("\n" + "="*60)
    print("DATA REQUIREMENTS CHECK")
    print("="*60)

    if len(valid_tracks) < 50:
        print(f"❌ INSUFFICIENT DATA: {len(valid_tracks)} valid tracks")
        print(f"   Minimum required: 50 tracks")
        print(f"   Recommended: 200-300 tracks")
    elif len(valid_tracks) < 150:
        print(f"⚠ LIMITED DATA: {len(valid_tracks)} valid tracks")
        print(f"   Recommended: 200-300 tracks for best results")
    else:
        print(f"✓ SUFFICIENT DATA: {len(valid_tracks)} valid tracks")

    # Check class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)

    print("\nEnergy labels:")
    for label in ENERGY_LABELS:
        count = energy_distribution[label]
        status = "✓" if count >= 10 else "⚠" if count >= 5 else "❌"
        print(f"  {status} {label:15s}: {count:3d} samples")

    print("\nVibe labels:")
    for label in VIBE_LABELS:
        count = vibe_distribution[label]
        status = "✓" if count >= 10 else "⚠" if count >= 5 else "❌"
        print(f"  {status} {label:15s}: {count:3d} samples")

    # Check for imbalance
    energy_min = min(energy_distribution.values()) if energy_distribution else 0
    energy_max = max(energy_distribution.values()) if energy_distribution else 0
    vibe_min = min(vibe_distribution.values()) if vibe_distribution else 0
    vibe_max = max(vibe_distribution.values()) if vibe_distribution else 0

    print("\n" + "="*60)
    print("CLASS BALANCE CHECK")
    print("="*60)

    if energy_min < 10:
        print(f"⚠ Energy classes: Minimum {energy_min} samples (recommend 10+)")
    else:
        print(f"✓ Energy classes: Well distributed (min {energy_min}, max {energy_max})")

    if vibe_min < 10:
        print(f"⚠ Vibe classes: Minimum {vibe_min} samples (recommend 10+)")
    else:
        print(f"✓ Vibe classes: Well distributed (min {vibe_min}, max {vibe_max})")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    ready_to_train = (
        len(errors) == 0 and
        len(valid_tracks) >= 50 and
        energy_min >= 5 and
        vibe_min >= 5
    )

    if ready_to_train:
        print("✓ READY TO TRAIN!")
        print("\nRun: python train.py")
        return True
    else:
        print("❌ NOT READY TO TRAIN")
        print("\nFix the issues above before training")

        if errors:
            print("- Fix validation errors")
        if len(valid_tracks) < 50:
            print(f"- Label more tracks (need {50 - len(valid_tracks)} more)")
        if energy_min < 5:
            print("- Balance energy class distribution")
        if vibe_min < 5:
            print("- Balance vibe class distribution")

        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate labels file')
    parser.add_argument('labels_file', nargs='?', default='data/manual_labels.json',
                       help='Path to labels JSON file')

    args = parser.parse_args()

    success = validate_labels(args.labels_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
