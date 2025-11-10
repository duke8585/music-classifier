.PHONY: help sample label clean

help:
	@echo "Music Classifier - Available commands:"
	@echo ""
	@echo "  make sample    - Generate random sample of 10 AIFF files for labeling"
	@echo "  make label     - Start the labeling web app"
	@echo "  make clean     - Clean up generated files"
	@echo ""

sample:
	@echo "Generating random sample of tracks..."
	@python src/phase1/generate_sample_list.py

label:
	@echo "Starting labeling webapp..."
	@python src/phase1/label_app.py & echo $$! > /tmp/label_app.pid
	@sleep 1
	@echo "Opening http://localhost:5001 in Safari..."
	@open -a Safari http://localhost:5001 || open http://localhost:5001
	@wait

clean:
	@echo "Cleaning up..."
	@rm -f data/sample_tracks.json
	@rm -f data/manual_labels.json
	@echo "Done"
