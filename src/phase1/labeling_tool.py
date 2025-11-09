"""
GUI-based labeling tool for manually tagging music tracks with energy and vibe labels.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import ENERGY_LABELS, VIBE_LABELS, MANUAL_LABELS_DIR


class MusicLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Electronic Music Labeling Tool")
        self.root.geometry("900x700")

        # Data
        self.tracks = []
        self.labels = []
        self.current_index = 0

        # Load data
        self.load_tracks()
        self.load_labels()

        # Setup UI
        self.setup_ui()
        self.display_current_track()

    def load_tracks(self):
        """Load track metadata from JSON file."""
        metadata_path = os.path.join(MANUAL_LABELS_DIR, 'mtg_jamendo_electronic.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.tracks = json.load(f)
            print(f"Loaded {len(self.tracks)} tracks")
        else:
            messagebox.showerror("Error", f"Metadata file not found: {metadata_path}")
            self.tracks = []

    def load_labels(self):
        """Load existing labels from JSON file."""
        labels_path = os.path.join(MANUAL_LABELS_DIR, 'manual_labels.json')

        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} existing labels")
        else:
            self.labels = []

    def save_labels(self):
        """Save labels to JSON file."""
        labels_path = os.path.join(MANUAL_LABELS_DIR, 'manual_labels.json')

        with open(labels_path, 'w') as f:
            json.dump(self.labels, f, indent=2)

        print(f"Saved {len(self.labels)} labels")

    def setup_ui(self):
        """Setup the user interface."""
        # Top frame - Track info
        info_frame = ttk.LabelFrame(self.root, text="Track Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.track_info_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.track_info_label.pack()

        self.tags_label = ttk.Label(info_frame, text="", font=('Arial', 9), foreground='gray')
        self.tags_label.pack()

        # Progress
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Progress: 0/0", font=('Arial', 10, 'bold'))
        self.progress_label.pack()

        # Energy selection
        energy_frame = ttk.LabelFrame(self.root, text="Energy Level", padding=10)
        energy_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.energy_var = tk.StringVar()

        for label in ENERGY_LABELS:
            ttk.Radiobutton(
                energy_frame,
                text=label.capitalize(),
                variable=self.energy_var,
                value=label
            ).pack(anchor=tk.W, padx=20, pady=5)

        # Vibe selection
        vibe_frame = ttk.LabelFrame(self.root, text="Vibe (Select Multiple)", padding=10)
        vibe_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.vibe_vars = {}
        for label in VIBE_LABELS:
            var = tk.BooleanVar()
            self.vibe_vars[label] = var
            ttk.Checkbutton(
                vibe_frame,
                text=label.capitalize(),
                variable=var
            ).pack(anchor=tk.W, padx=20, pady=3)

        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame,
            text="← Previous",
            command=self.previous_track
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Skip",
            command=self.skip_track
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Save & Next →",
            command=self.save_and_next,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Jump to Track...",
            command=self.jump_to_track
        ).pack(side=tk.RIGHT, padx=5)

        # Statistics
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="", font=('Arial', 9))
        self.stats_label.pack()

    def display_current_track(self):
        """Display the current track information."""
        if not self.tracks:
            self.track_info_label.config(text="No tracks loaded")
            return

        if self.current_index >= len(self.tracks):
            messagebox.showinfo("Complete", "You've labeled all available tracks!")
            return

        track = self.tracks[self.current_index]

        # Display track info
        info_text = f"Track: {track.get('title', 'Unknown')} | Artist: {track.get('artist', 'Unknown')}"
        self.track_info_label.config(text=info_text)

        tags_text = f"Tags: {track.get('tags', 'N/A')}"
        self.tags_label.config(text=tags_text)

        # Update progress
        labeled_count = len(self.labels)
        self.progress_label.config(text=f"Progress: {labeled_count}/{len(self.tracks)} labeled | Current: {self.current_index + 1}/{len(self.tracks)}")

        # Load existing label if any
        existing_label = self.get_existing_label(track['track_id'])
        if existing_label:
            self.energy_var.set(existing_label.get('energy', ''))
            for vibe in VIBE_LABELS:
                self.vibe_vars[vibe].set(vibe in existing_label.get('vibes', []))
        else:
            self.energy_var.set('')
            for vibe in VIBE_LABELS:
                self.vibe_vars[vibe].set(False)

        # Update statistics
        self.update_statistics()

    def get_existing_label(self, track_id):
        """Get existing label for a track if it exists."""
        for label in self.labels:
            if label['track_id'] == track_id:
                return label
        return None

    def save_and_next(self):
        """Save current labels and move to next track."""
        if not self.tracks:
            return

        track = self.tracks[self.current_index]

        # Validate input
        energy = self.energy_var.get()
        if not energy:
            messagebox.showwarning("Validation", "Please select an energy level")
            return

        vibes = [vibe for vibe, var in self.vibe_vars.items() if var.get()]
        if not vibes:
            messagebox.showwarning("Validation", "Please select at least one vibe")
            return

        # Create label
        label = {
            'track_id': track['track_id'],
            'artist': track.get('artist', 'Unknown'),
            'title': track.get('title', 'Unknown'),
            'energy': energy,
            'vibes': vibes,
            'path': track.get('path', '')
        }

        # Update or append label
        existing_index = None
        for i, l in enumerate(self.labels):
            if l['track_id'] == track['track_id']:
                existing_index = i
                break

        if existing_index is not None:
            self.labels[existing_index] = label
        else:
            self.labels.append(label)

        # Save to file
        self.save_labels()

        # Move to next track
        self.current_index += 1
        self.display_current_track()

    def skip_track(self):
        """Skip current track without saving."""
        self.current_index += 1
        self.display_current_track()

    def previous_track(self):
        """Go to previous track."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_track()

    def jump_to_track(self):
        """Jump to a specific track number."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Jump to Track")
        dialog.geometry("300x100")

        ttk.Label(dialog, text="Enter track number:").pack(pady=10)

        entry = ttk.Entry(dialog)
        entry.pack(pady=5)

        def jump():
            try:
                track_num = int(entry.get())
                if 1 <= track_num <= len(self.tracks):
                    self.current_index = track_num - 1
                    self.display_current_track()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Track number must be between 1 and {len(self.tracks)}")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")

        ttk.Button(dialog, text="Jump", command=jump).pack(pady=5)

    def update_statistics(self):
        """Update statistics display."""
        if not self.labels:
            self.stats_label.config(text="No labels yet")
            return

        energy_counts = {}
        vibe_counts = {}

        for label in self.labels:
            energy = label.get('energy', '')
            if energy:
                energy_counts[energy] = energy_counts.get(energy, 0) + 1

            for vibe in label.get('vibes', []):
                vibe_counts[vibe] = vibe_counts.get(vibe, 0) + 1

        stats_text = f"Labeled: {len(self.labels)} | "
        stats_text += f"Energy: {', '.join([f'{k}: {v}' for k, v in energy_counts.items()][:3])}"

        self.stats_label.config(text=stats_text)


def main():
    root = tk.Tk()
    app = MusicLabelingTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
