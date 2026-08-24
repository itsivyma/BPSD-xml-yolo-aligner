# Development guide

## Supported entry points

Use `bpsd-aligner` rather than invoking root scripts directly. Daily workflows:

- `web`: normal multi-page user workflow;
- `align` / `batch-align`: direct terminal alignment;
- `dry-run`: resumable dataset processing;
- `regression-smoke`: fixed real-page verification;
- `xml-export` / `combine`: research exports;
- Review commands: queue, sheets, apply, dataset and calibration;
- `job-admin`: deployment inspection and cleanup.

Several root-level `slur_*`, `fingering_*`, `fermata_*`, `tuplet_*`, and audit
scripts are historical research/validation tools. They remain available for
reproducibility but are not the place to add normal website behavior.

## Add or change a YOLO class

1. Confirm the exact class name and ID in the dataset's `notes.json`. Never
   hard-code a new numeric ID into generic matching logic.
2. Read `docs/XML_YOLO_CLASS_COMPARISON.md` and identify actual MusicXML
   evidence. Decide whether the value is known, ambiguous, or unavailable.
3. Update `bpsd_aligner/class_registry.py` only for score-independent facts:
   threshold family and BPS-OMR `musical_time`. Unknown roles must stay `None`.
4. Add matching behavior to the appropriate engine strategy. Until the large
   engine is decomposed, search `bps_xml_alignment.py` for the closest existing
   class family and keep the change isolated.
5. Preserve uncertainty: incomplete evidence must use Review status and blank
   strict output values rather than a guessed note or time.
6. Add unit tests for positive, ambiguous and missing-evidence cases.
7. Add or select a representative real page, then run the fixed regression.
8. Update class comparison and website documentation.

## Confidence policy

`confidence` is a heuristic score. Defaults live in `class_registry.py` and an
operator may override them through `BPSD_ALIGNER_THRESHOLDS`. Do not lower a
threshold from a few examples. Build `review_training_rows.csv`, collect at
least 200 reviewed rows across representative scores, and run
`calibrate-thresholds` before proposing a change.

## Tests

```bash
# Focused test while editing
python -m pytest -q tests/test_class_registry.py

# Required before handoff
python -m pytest -q

# Required after matching, repeat, schema or Review changes
bpsd-aligner regression-smoke \
  --manifest regression/representative_pages.json \
  --dataset-root "/path/to/中研院" \
  --output-dir output/regression-smoke \
  --baseline regression/representative_baseline.json \
  --resume
```

The semantic baseline detects endpoint, note-ID, staff, class, time, written
measure and cross-page changes even when aggregate row counts are unchanged.

## Refactoring rules

- One commit should either move code or change behavior, not both.
- New geometry behavior belongs in `bpsd_aligner/geometry.py`; review drawing
  belongs in `bpsd_aligner/overlay.py`; generic one-to-one pairing belongs in
  `bpsd_aligner/candidate_scoring.py`. `bps_xml_alignment.py` temporarily
  re-exports compatibility names for older scripts.
- Preserve public CLI commands and compatibility imports during a move.
- Use typed records at module boundaries instead of adding more unstructured
  dictionary keys.
- Keep UI state management out of alignment algorithms.
- Keep generated outputs under ignored directories; never commit private score
  data, review images or annotation files.
- Long-running work must report progress and write checkpoints after each page.

## Release checklist

1. `git diff --check` passes.
2. Full unit tests pass.
3. Fixed real-page regression passes without unexplained baseline changes.
4. Package installs in a fresh virtual environment.
5. `bpsd-aligner --help`, `--version`, and `web` start successfully.
6. README, START_HERE and relevant schema/class documentation are current.
7. Bump the package version only when preparing a release commit.
