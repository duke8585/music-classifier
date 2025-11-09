"""
Rekordbox XML integration for writing ML predictions to MyTag fields.
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import PREDICTIONS_DIR


class RekordboxIntegration:
    """Handle Rekordbox XML export/import and tag writing."""

    def __init__(self, xml_path):
        """
        Initialize with path to Rekordbox XML export.

        Args:
            xml_path: Path to rekordbox.xml file
        """
        self.xml_path = xml_path
        self.tree = None
        self.root = None
        self.tracks = {}

    def parse_xml(self):
        """Parse Rekordbox XML file."""
        print(f"Parsing {self.xml_path}...")

        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()

            # Find COLLECTION element
            collection = self.root.find('.//COLLECTION')

            if collection is None:
                raise ValueError("COLLECTION element not found in XML")

            # Build track index
            for track in collection.findall('TRACK'):
                track_id = track.get('TrackID')
                location = track.get('Location', '')

                # Decode file path
                if location.startswith('file://localhost/'):
                    location = location.replace('file://localhost/', '/')

                # URL decode
                from urllib.parse import unquote
                location = unquote(location)

                self.tracks[track_id] = {
                    'element': track,
                    'location': location,
                    'name': track.get('Name', ''),
                    'artist': track.get('Artist', '')
                }

            print(f"Found {len(self.tracks)} tracks in Rekordbox library")

        except Exception as e:
            raise Exception(f"Error parsing XML: {e}")

    def match_predictions_to_tracks(self, predictions):
        """
        Match prediction file paths to Rekordbox tracks.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Dictionary mapping track_id to prediction
        """
        print("Matching predictions to Rekordbox tracks...")

        matches = {}
        unmatched_predictions = []

        for pred in predictions:
            pred_path = os.path.normpath(pred['path'])
            pred_filename = pred['filename']

            matched = False

            for track_id, track_info in self.tracks.items():
                rb_path = os.path.normpath(track_info['location'])

                # Try exact path match first
                if pred_path == rb_path:
                    matches[track_id] = pred
                    matched = True
                    break

                # Try filename match
                if pred_filename == os.path.basename(rb_path):
                    matches[track_id] = pred
                    matched = True
                    break

            if not matched:
                unmatched_predictions.append(pred_filename)

        print(f"Matched {len(matches)} tracks")
        if unmatched_predictions:
            print(f"Unmatched predictions: {len(unmatched_predictions)}")

        return matches

    def write_tags(self, predictions, my_tag_format='energy_vibes'):
        """
        Write predictions to Rekordbox MyTag fields.

        Args:
            predictions: List of prediction dictionaries
            my_tag_format: Format for MyTag field
                - 'energy_vibes': "peak | dark, hypnotic"
                - 'energy_only': "peak"
                - 'vibes_only': "dark, hypnotic"
                - 'detailed': "E:peak(0.85) | V:dark,hypnotic"
        """
        print("Writing tags to Rekordbox XML...")

        # Match predictions to tracks
        matches = self.match_predictions_to_tracks(predictions)

        updated_count = 0

        for track_id, pred in matches.items():
            track_element = self.tracks[track_id]['element']

            # Generate tag text based on format
            if my_tag_format == 'energy_vibes':
                tag_text = f"{pred['energy']} | {', '.join(pred['vibes'])}"
            elif my_tag_format == 'energy_only':
                tag_text = pred['energy']
            elif my_tag_format == 'vibes_only':
                tag_text = ', '.join(pred['vibes'])
            elif my_tag_format == 'detailed':
                energy_conf = pred['energy_confidence']
                tag_text = f"E:{pred['energy']}({energy_conf:.2f}) | V:{','.join(pred['vibes'])}"
            else:
                tag_text = f"{pred['energy']} | {', '.join(pred['vibes'])}"

            # Set MyTag field (Comments field in some Rekordbox versions)
            # Rekordbox uses different fields depending on version
            # Try multiple possible tag fields

            # Option 1: Use Comments field
            track_element.set('Comments', tag_text)

            # Option 2: Create TEMPO element if it doesn't exist
            # This is where some Rekordbox versions store custom tags
            tempo_elem = track_element.find('TEMPO')
            if tempo_elem is not None:
                tempo_elem.set('Inizio', tag_text)

            updated_count += 1

        print(f"Updated {updated_count} tracks")

        return updated_count

    def save_xml(self, output_path=None, create_backup=True):
        """
        Save modified XML.

        Args:
            output_path: Path to save modified XML (default: overwrite original)
            create_backup: Whether to create backup of original file
        """
        if output_path is None:
            output_path = self.xml_path

        # Create backup
        if create_backup and os.path.exists(self.xml_path):
            backup_path = f"{self.xml_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.xml_path, backup_path)
            print(f"Created backup: {backup_path}")

        # Save XML
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Saved modified XML to {output_path}")

    def generate_import_instructions(self):
        """Generate instructions for importing back to Rekordbox."""
        instructions = """
═══════════════════════════════════════════════════════
REKORDBOX IMPORT INSTRUCTIONS
═══════════════════════════════════════════════════════

1. BACKUP YOUR REKORDBOX LIBRARY
   - Open Rekordbox
   - File > Library > Backup Library
   - Save backup to safe location

2. IMPORT MODIFIED XML
   - Close Rekordbox completely
   - Locate your rekordbox.xml file
   - Replace it with the modified version (backup created automatically)
   - Restart Rekordbox

3. VERIFY TAGS
   - Check a few tracks to ensure tags appear correctly
   - Tags should appear in the Comments field
   - You can search using these tags in Rekordbox

4. ALTERNATIVE: Manual Import
   - File > Library > Import Collection
   - Select the modified XML file
   - Rekordbox will import the tags

5. TROUBLESHOOTING
   - If tags don't appear, check File > Preferences > View
   - Enable 'Comments' column in track display
   - You may need to reimport your library

═══════════════════════════════════════════════════════
USING TAGS FOR DJ SETS
═══════════════════════════════════════════════════════

Search examples in Rekordbox:
- "peak" - Find all peak-time tracks
- "dark, hypnotic" - Find dark hypnotic tracks
- "closing" - Find closing tracks

You can combine these with other Rekordbox filters for powerful
track selection during live sets!

═══════════════════════════════════════════════════════
"""
        return instructions


def integrate_predictions_with_rekordbox(rekordbox_xml_path, predictions_json_path=None):
    """
    Main function to integrate predictions with Rekordbox.

    Args:
        rekordbox_xml_path: Path to Rekordbox XML export
        predictions_json_path: Path to predictions JSON (default: latest in predictions dir)
    """
    # Load predictions
    if predictions_json_path is None:
        # Find latest predictions file
        predictions_files = [
            os.path.join(PREDICTIONS_DIR, f)
            for f in os.listdir(PREDICTIONS_DIR)
            if f.endswith('.json')
        ]

        if not predictions_files:
            raise FileNotFoundError("No predictions files found")

        predictions_json_path = max(predictions_files, key=os.path.getmtime)
        print(f"Using predictions file: {predictions_json_path}")

    with open(predictions_json_path, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Initialize Rekordbox integration
    rb = RekordboxIntegration(rekordbox_xml_path)
    rb.parse_xml()

    # Write tags
    rb.write_tags(predictions, my_tag_format='energy_vibes')

    # Save modified XML
    output_path = rekordbox_xml_path.replace('.xml', '_tagged.xml')
    rb.save_xml(output_path=output_path, create_backup=True)

    # Print instructions
    print(rb.generate_import_instructions())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Integrate predictions with Rekordbox')
    parser.add_argument('rekordbox_xml', help='Path to Rekordbox XML file')
    parser.add_argument('--predictions', help='Path to predictions JSON (optional)')
    parser.add_argument('--format', default='energy_vibes',
                        choices=['energy_vibes', 'energy_only', 'vibes_only', 'detailed'],
                        help='MyTag format')

    args = parser.parse_args()

    integrate_predictions_with_rekordbox(args.rekordbox_xml, args.predictions)
