#!/usr/bin/env python3
"""
Analyze manual labels to identify patterns, overlaps, and inconsistencies.
"""
# FIXME remove later on

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


def load_labels(labels_path: Path) -> Dict:
    """Load manual labels from JSON file."""
    with open(labels_path) as f:
        return json.load(f)


def analyze_energy_distribution(labels: Dict) -> pd.DataFrame:
    """Analyze energy label distribution."""
    energy_counts = Counter(track["energy"] for track in labels.values())
    df = pd.DataFrame.from_dict(energy_counts, orient="index", columns=["count"])
    df["percentage"] = (df["count"] / df["count"].sum() * 100).round(2)
    return df.sort_values("count", ascending=False)


def analyze_vibe_distribution(labels: Dict) -> pd.DataFrame:
    """Analyze vibe label distribution."""
    all_vibes = []
    for track in labels.values():
        all_vibes.extend(track["vibes"])
    vibe_counts = Counter(all_vibes)
    df = pd.DataFrame.from_dict(vibe_counts, orient="index", columns=["count"])
    df["percentage"] = (df["count"] / len(labels) * 100).round(2)
    return df.sort_values("count", ascending=False)


def analyze_vibe_combinations(labels: Dict) -> pd.DataFrame:
    """Analyze which vibes commonly appear together."""
    vibe_pairs = defaultdict(int)

    for track in labels.values():
        vibes = sorted(track["vibes"])
        # Count all pairs
        for i, vibe1 in enumerate(vibes):
            for vibe2 in vibes[i + 1 :]:
                pair = (vibe1, vibe2)
                vibe_pairs[pair] += 1

    # Convert to DataFrame
    data = [
        {"vibe1": v1, "vibe2": v2, "co_occurrence": count} for (v1, v2), count in vibe_pairs.items()
    ]
    df = pd.DataFrame(data).sort_values("co_occurrence", ascending=False)
    return df


def analyze_vibes_per_track(labels: Dict) -> Dict:
    """Analyze how many vibes are typically assigned per track."""
    vibe_counts = [len(track["vibes"]) for track in labels.values()]
    return {
        "mean": sum(vibe_counts) / len(vibe_counts),
        "min": min(vibe_counts),
        "max": max(vibe_counts),
        "distribution": Counter(vibe_counts),
    }


def analyze_energy_vibe_patterns(labels: Dict) -> pd.DataFrame:
    """Analyze which vibes are most common for each energy level."""
    energy_vibe_matrix = defaultdict(lambda: defaultdict(int))

    for track in labels.values():
        energy = track["energy"]
        for vibe in track["vibes"]:
            energy_vibe_matrix[energy][vibe] += 1

    # Convert to DataFrame
    df = pd.DataFrame(energy_vibe_matrix).T.fillna(0).astype(int)
    return df


def find_label_overlaps(
    labels: Dict,
) -> Dict[str, List[Tuple[str, str, Set[str]]]]:
    """Identify tracks that might be causing confusion."""
    # Group by energy level
    energy_groups = defaultdict(list)
    for filename, data in labels.items():
        energy_groups[data["energy"]].append((filename, set(data["vibes"])))

    overlaps = {
        "trippy_hypnotic": [],
        "dark_vibes": [],
        "atmospheric_dubby": [],
        "melodic_tracks": [],
    }

    for filename, data in labels.items():
        vibes = set(data["vibes"])

        # Check for your noted overlaps
        if "trippy" in vibes and "hypnotic" in vibes:
            overlaps["trippy_hypnotic"].append((filename, data["energy"], vibes))

        if "dark" in vibes:
            overlaps["dark_vibes"].append((filename, data["energy"], vibes))

        if "atmospheric" in vibes and "dubby" in vibes:
            overlaps["atmospheric_dubby"].append((filename, data["energy"], vibes))

        if "melodic" in vibes:
            overlaps["melodic_tracks"].append((filename, data["energy"], vibes))

    return overlaps


def main():
    labels_path = Path("data/manual_labels/manual_labels.json")
    labels = load_labels(labels_path)

    print("=" * 80)
    print("LABEL ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal tracks labeled: {len(labels)}\n")

    # Energy distribution
    print("=" * 80)
    print("ENERGY LABEL DISTRIBUTION")
    print("=" * 80)
    energy_dist = analyze_energy_distribution(labels)
    print(energy_dist)
    print()

    # Vibe distribution
    print("=" * 80)
    print("VIBE LABEL DISTRIBUTION")
    print("=" * 80)
    vibe_dist = analyze_vibe_distribution(labels)
    print(vibe_dist)
    print()

    # Vibes per track
    print("=" * 80)
    print("VIBES PER TRACK STATISTICS")
    print("=" * 80)
    vibe_stats = analyze_vibes_per_track(labels)
    print(f"Mean vibes per track: {vibe_stats['mean']:.2f}")
    print(f"Min vibes per track: {vibe_stats['min']}")
    print(f"Max vibes per track: {vibe_stats['max']}")
    print("\nDistribution:")
    for num_vibes, count in sorted(vibe_stats["distribution"].items()):
        print(f"  {num_vibes} vibes: {count} tracks")
    print()

    # Top vibe combinations
    print("=" * 80)
    print("TOP 20 VIBE COMBINATIONS (co-occurrence)")
    print("=" * 80)
    vibe_combos = analyze_vibe_combinations(labels)
    print(vibe_combos.head(20).to_string(index=False))
    print()

    # Energy-Vibe patterns
    print("=" * 80)
    print("ENERGY x VIBE MATRIX (how often each vibe appears with each energy)")
    print("=" * 80)
    energy_vibe = analyze_energy_vibe_patterns(labels)
    print(energy_vibe)
    print()

    # Overlaps you mentioned
    print("=" * 80)
    print("IDENTIFIED OVERLAPS & AMBIGUITIES")
    print("=" * 80)
    overlaps = find_label_overlaps(labels)

    print(f"\nTracks with BOTH 'trippy' AND 'hypnotic': {len(overlaps['trippy_hypnotic'])}")
    print(f"Tracks with 'dark' label: {len(overlaps['dark_vibes'])}")
    print(f"Tracks with BOTH 'atmospheric' AND 'dubby': {len(overlaps['atmospheric_dubby'])}")
    print(f"Tracks with 'melodic' label: {len(overlaps['melodic_tracks'])}")
    print()

    # Sample some trippy/hypnotic overlaps
    print("Sample tracks with BOTH 'trippy' AND 'hypnotic':")
    for filename, energy, vibes in overlaps["trippy_hypnotic"][:5]:
        short_name = filename[:60] + "..." if len(filename) > 60 else filename
        print(f"  [{energy}] {vibes}")
        print(f"    {short_name}")
    print()

    # Analyze "dark" ambiguity
    print("Analyzing 'dark' label across different energies:")
    dark_by_energy = defaultdict(list)
    for filename, energy, vibes in overlaps["dark_vibes"]:
        dark_by_energy[energy].append(vibes)

    for energy in ["warm-up", "building", "peak", "intense", "closing"]:
        if energy in dark_by_energy:
            print(f"\n  {energy} ({len(dark_by_energy[energy])} tracks):")
            # Show most common vibe combinations with dark
            vibe_patterns = Counter(tuple(sorted(vibes)) for vibes in dark_by_energy[energy])
            for pattern, count in vibe_patterns.most_common(3):
                print(f"    {count}x: {set(pattern)}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
