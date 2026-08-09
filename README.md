# BPSD XML–YOLO Aligner

Installable command-line tools and a Streamlit website for aligning YOLO
symbol boxes on scanned BPSD score pages with MusicXML events and BPSD note
annotations. The same Python pipeline powers both interfaces.

The project creates candidate semantic links and visual QA material. It
does not assume that a YOLO box and a MusicXML event share an ID, and it
does not treat geometric proximity as proof. Direct MusicXML matches and
geometry-derived estimates have different statuses, confidence, and review
requirements.

## Current capabilities

- Read a scanned score page, YOLO TXT annotations, and `notes.json`.
- Detect piano systems, staves, and approximate measure boundaries from
  the target scan.
- Parse MusicXML notes, measures, divisions, time signatures, dynamics,
  slurs, ties, voices, and staves.
- Convert MusicXML event positions to pickup-aware BPSD musical time.
- Attach BPSD note IDs when a corresponding note annotation is
  available, including notes that fall within a BPSD tied span.
- Match `dynamicF`, `dynamicP`, and `dynamicS` to MusicXML dynamic
  events in page reading order.
- Match MusicXML staccato, fermata, slur, tie, ornament, and tuplet evidence;
  retain lower-confidence assignments as review candidates.
- Give every YOLO class a start/end time candidate from direct MusicXML
  evidence or the nearest score anchor, while explicitly marking estimates.
- Generate slur endpoint candidates and visual QA sheets, including
  scan-only and cross-system cases.
- Keep confirmed, candidate, unresolved, and scan-only results
  distinguishable during review.
- Upload one raw score page through the website and run the same alignment
  pipeline without preparing intermediate CSV files first.
- Export a single `all_information.csv` containing every YOLO row, every
  extracted MusicXML event, and every flattened source XML node.
- Draw all YOLO boxes and alignment labels back onto full-page review images.

## Evidence and review rules

The input sources contribute different information:

| Source | Information used |
| --- | --- |
| YOLO TXT | Class ID and normalized bounding-box geometry |
| `notes.json` | Class ID to class-name mapping |
| Scan image | Staff, system, barline, and glyph geometry |
| MusicXML | Musical structure, timing, pitch, voice, staff, slur, and tie events |
| BPSD note annotations | BPSD note IDs and note timing |

There is no universal ID shared by YOLO and MusicXML. The tools therefore
produce alignments by combining score structure and geometry, then
expose uncertain cases for review.

The default policy is conservative:

- MusicXML-supported values may be written when the correspondence is
  established.
- Candidate values remain labeled as candidates.
- Unknown semantic links stay blank rather than being presented as facts;
  geometry-derived time estimates are populated and marked `review`.
- `--infer-fingerings` is optional and non-authoritative because the
  current source MusicXML contains no fingering elements.
- Repeat mapping must be checked before extending the workflow across
  a whole sonata.

## Alignment inputs

The main alignment command uses external copies of:

- a scanned score page;
- the matching YOLO `.txt` file;
- the matching `notes.json`;
- the corresponding BPSD MusicXML file; and
- the corresponding BPSD note annotation CSV.

A clean rendered score page is also used by the cross-system slur QA
tool.

Dataset files are not included in this repository.

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install .
```

Confirm the installation:

```bash
bpsd-aligner --version
bpsd-aligner --help
```

For development and tests, use `python -m pip install ".[dev]"`.

## Terminal and website

Every processing stage is available through one terminal command:

```bash
bpsd-aligner align --help
bpsd-aligner dry-run --help
bpsd-aligner xml-export --help
bpsd-aligner combine --help
```

Start the website locally with:

```bash
bpsd-aligner web
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

The **Run alignment** tab accepts one score page per run:

1. score image (`.jpg`, `.jpeg`, or `.png`);
2. matching YOLO annotation (`.txt`);
3. BPSD written/repetition MusicXML (`.xml` or `.musicxml`);
4. BPSD `ann_score_note.csv`;
5. YOLO `notes.json` class map; and
6. optionally, unfolded MusicXML for a score containing repeats.

Set **MusicXML page** to the page represented by the uploaded image, then click
**Align all information**. The page reports each stage as it runs and keeps a
completed job in the current Streamlit session, so an ordinary UI rerun with
the same files reuses the in-memory result.

The website returns:

- `all_information.csv`: every YOLO row, MusicXML event, and raw XML node;
- `combined_master.csv`: BPS-OMR-oriented YOLO and MusicXML event rows;
- `alignment_detailed.csv`: box-level matching evidence and confidence;
- full-page dynamics, fingering, and all-symbol review overlays;
- validation JSON and a ZIP containing all generated outputs.

Use `source_record_type` in `all_information.csv` to distinguish `yolo`,
`xml_event`, and `xml_node` rows. XML-only events receive readable class names.
Rows are ordered by `start_meas` and `end_meas`; raw XML nodes without a musical
time follow the timed symbol rows. Machine-generated fingering and geometric
fallback links remain candidates and must be checked in the overlay.

Dataset-wide raw alignment remains available through the resumable CLI because
source score collections can be too large for ordinary browser uploads.

## Run the alignment

```bash
bpsd-aligner align \
  --image /path/to/page.jpeg \
  --yolo /path/to/page.txt \
  --notes-json /path/to/notes.json \
  --xml /path/to/score.xml \
  --bps-notes /path/to/ann_score_note.csv \
  --output-dir /path/to/output \
  --all-symbols
```

Run the following command for all available options:

```bash
python bps_xml_alignment.py --help
```

The alignment command writes a CSV, QA overlays, and a JSON report to the
selected output directory. With `--all-symbols`, direct MusicXML matches are
preferred and every remaining class receives a reviewable geometry-derived
time candidate when a page anchor is available.

## Slur QA tools

- `slur_endpoint_check.py`: inspect the endpoint notes of one MusicXML
  slur.
- `scan_only_slur_check.py`: inspect a slur visible in the scan but not
  matched to MusicXML.
- `cross_system_slur_check.py`: inspect the two visible segments of a
  slur crossing a system break.
- `slur_batch_candidates.py`: rank endpoint candidates and combine
  earlier human-review decisions.
- `slur_batch_endpoint_sheet.py`: generate batch endpoint review sheets.

Batch results use explicit review states:

- `locked_xml_match`: confirmed MusicXML match.
- `locked_scan_only`: confirmed scan-only slur.
- `high_confidence_candidate`: promising candidate, not yet confirmed.
- `needs_review`: insufficient or conflicting evidence.
- `possible_scan_only`: no sufficiently supported MusicXML match yet.

Only the two `locked_*` states represent previously confirmed review
decisions.

## Resumable dataset stages

The dataset-wide commands write progress with `flush=True`, use atomic
output replacement, and can reuse completed work. Run Python unbuffered
so progress is visible immediately in command runners:

```bash
python -u dataset_dry_run.py \
  --manifest /path/to/page_manifest.csv \
  --scope /path/to/system_scope_manifest.csv \
  --notes-json /path/to/notes.json \
  --repeat-mapping-dir /path/to/repeat_mapping \
  --review-dir /path/to/human_reviews \
  --output-dir /path/to/dry_run \
  --resume
```

The alignment stage checkpoints every successful page under
`OUTPUT/checkpoints/`. Missing, stale, corrupt, or previously failed pages
are rerun; valid page checkpoints are loaded directly. Sonata/master
aggregation and validation run only after all page results are available.

To adopt a previously completed and passing output directory without
executing alignment or rewriting its CSV files, replace `--resume` with:

```text
--checkpoint-existing-only
```

The visual stages resume at page/sheet granularity:

```bash
python -u render_yolo_overlays.py \
  --xia-dir /path/to/xia \
  --output-dir /path/to/yolo_overlays \
  --resume

python -u alignment_review_sheets.py \
  --master-csv /path/to/Xia_BPSD_alignment_master.csv \
  --output-dir /path/to/review_sheets \
  --resume
```

Overlay PNG pairs and review PNG/CSV pairs are decoded and checked before
reuse. Index CSV/HTML files are rebuilt atomically after the reusable and
new outputs have been collected.

## YOLO format

Each YOLO annotation row contains:

```text
class_id x_center y_center width height
```

Example:

```text
18 0.201429 0.228247 0.022286 0.017574
```

The four bounding-box coordinates are normalized values between `0`
and `1`.

## Run tests

```bash
python -m pytest -v
```

## Project structure

```text
.
├── bpsd_aligner
│   ├── cli.py
│   ├── web.py
│   ├── web_pipeline.py
│   └── web_utils.py
├── bps_xml_alignment.py
├── combine_yolo_xml.py
├── dataset_dry_run.py
├── dataset_inventory.py
├── xml_export.py
├── cross_system_slur_check.py
├── pyproject.toml
├── Dockerfile
├── requirements.txt
└── tests/
```

## Data and privacy

This repository does not include score images, MusicXML files, YOLO
annotations, BPSD annotations, human-review CSV files, generated QA
images, exported spreadsheets, or other dataset files.

Machine-specific prototypes and input paths are intentionally excluded
from version control.

## Current scope

The lossless combined CSV preserves every YOLO bbox and every MusicXML event,
adds readable XML-only class names, and is sorted by musical start/end time. It
does not claim that every YOLO/XML pair is correct: candidate, ambiguous,
unresolved, XML-only, and YOLO-only states remain explicit until reviewed.

## License

No software license has been selected yet. Until a license is added,
copyright law reserves reuse, modification, and redistribution rights
to the copyright holder.

## Disclaimer

This is an independent annotation-alignment and QA utility. Dataset
files must be obtained and used according to their original licenses
and terms.
