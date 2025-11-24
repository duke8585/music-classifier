#!/usr/bin/env python3
"""
Relabel manual labels with simplified taxonomy.

New taxonomy:
Energy (4 classes):
  - intro/outro: closing + warm-up
  - low: building
  - high: peak
  - intense: intense

Vibes (6 classes):
  - deep
  - melodic
  - hypnotic: trippy + hypnotic merged
  - dubby
  - breaks
  - percussive/lofi: analog/lofi + clean/digital merged

Removed:
  - dark
  - atmospheric (too broad)
  - trippy (merged into hypnotic)
  - clean/digital (merged into percussive/lofi)
  - analog/lofi (merged into percussive/lofi)

"""
# FIXME remove later on

import json
from pathlib import Path
from typing import List


def relabel_energy(old_energy: str) -> str:
    """Map old energy labels to new simplified taxonomy."""
    energy_map = {
        "closing": "intro/outro",
        "warm-up": "intro/outro",
        "building": "low",
        "peak": "high",
        "intense": "intense",
    }
    return energy_map[old_energy]


def relabel_vibes(old_vibes: List[str]) -> List[str]:
    """Map old vibe labels to new simplified taxonomy."""
    new_vibes = set()

    for vibe in old_vibes:
        if vibe == "dark":
            # Remove dark entirely
            continue
        elif vibe == "atmospheric":
            # Remove atmospheric entirely
            continue
        elif vibe == "trippy":
            # Merge trippy into hypnotic
            new_vibes.add("hypnotic")
        elif vibe == "hypnotic":
            new_vibes.add("hypnotic")
        elif vibe == "analog/lofi":
            # Merge into percussive/lofi
            new_vibes.add("percussive/lofi")
        elif vibe == "clean/digital":
            # Merge into percussive/lofi
            new_vibes.add("percussive/lofi")
        elif vibe in ["deep", "melodic", "dubby", "breaks"]:
            # Keep as-is
            new_vibes.add(vibe)
        else:
            print(f"WARNING: Unknown vibe '{vibe}'")

    return sorted(list(new_vibes))


def relabel_dataset(input_path: Path, output_path: Path) -> None:
    """Relabel the entire dataset with simplified taxonomy."""
    print(f"Loading labels from: {input_path}")
    with open(input_path) as f:
        old_labels = json.load(f)

    new_labels = {}
    energy_counts = {"intro/outro": 0, "low": 0, "high": 0, "intense": 0}
    vibe_counts = {
        "deep": 0,
        "melodic": 0,
        "hypnotic": 0,
        "dubby": 0,
        "breaks": 0,
        "percussive/lofi": 0,
    }
    vibes_per_track = []

    for filename, data in old_labels.items():
        old_energy = data["energy"]
        old_vibes = data["vibes"]

        new_energy = relabel_energy(old_energy)
        new_vibes = relabel_vibes(old_vibes)

        # Handle tracks with no vibes after relabeling
        if len(new_vibes) == 0:
            # Most intro/outro tracks that were only "atmospheric" are likely "dubby"
            # since atmospheric+dubby was a common pairing
            if new_energy == "intro/outro":
                new_vibes = ["dubby"]
                print(
                    f"INFO: intro/outro track with no vibes, defaulting to 'dubby': {filename[:60]}"
                )
            else:
                print(f"WARNING: Track has no vibes after relabeling: {filename}")
                print(f"  Old vibes: {old_vibes}, Energy: {new_energy}")
                # For non-intro/outro tracks, keep empty but warn
                pass

        new_labels[filename] = {"energy": new_energy, "vibes": new_vibes}

        # Count
        energy_counts[new_energy] += 1
        for vibe in new_vibes:
            vibe_counts[vibe] += 1
        vibes_per_track.append(len(new_vibes))

    # Statistics
    print("\n" + "=" * 80)
    print("RELABELING STATISTICS")
    print("=" * 80)
    print(f"\nTotal tracks: {len(new_labels)}")
    print(f"Original tracks: {len(old_labels)}")

    print("\n--- ENERGY DISTRIBUTION ---")
    for energy, count in sorted(energy_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(new_labels) * 100
        print(f"  {energy:12s}: {count:3d} tracks ({pct:5.1f}%)")

    print("\n--- VIBE DISTRIBUTION ---")
    for vibe, count in sorted(vibe_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(new_labels) * 100
        print(f"  {vibe:16s}: {count:3d} tracks ({pct:5.1f}%)")

    print("\n--- VIBES PER TRACK ---")
    avg_vibes = sum(vibes_per_track) / len(vibes_per_track)
    print(f"  Average: {avg_vibes:.2f}")
    print(f"  Min: {min(vibes_per_track)}")
    print(f"  Max: {max(vibes_per_track)}")

    # Count tracks with 0 vibes
    zero_vibe_tracks = sum(1 for v in vibes_per_track if v == 0)
    if zero_vibe_tracks > 0:
        print(f"\n  WARNING: {zero_vibe_tracks} tracks have NO vibes after relabeling!")

    # Save
    print(f"\nSaving relabeled data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_labels, f, indent=2, ensure_ascii=False)

    print("\n✓ Relabeling complete!")
    print("=" * 80)


def main():
    input_path = Path("data/manual_labels/manual_labels.json")
    output_path = Path("data/manual_labels/manual_labels_simplified.json")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    relabel_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
