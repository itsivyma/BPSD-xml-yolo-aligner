# BPSD XML–YOLO Aligner

> New user or maintainer? Start with the concise
> [15-minute guide](START_HERE.md), then read the
> [architecture](docs/ARCHITECTURE.md),
> [development guide](docs/DEVELOPMENT.md), and
> [handover notes](docs/HANDOVER.md).

[![CI](https://github.com/itsivyma/BPSD-xml-yolo-aligner/actions/workflows/ci.yml/badge.svg)](https://github.com/itsivyma/BPSD-xml-yolo-aligner/actions/workflows/ci.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Installable command-line tools and a Streamlit website for aligning YOLO
symbol boxes on scanned BPSD score pages with MusicXML events and BPSD note
annotations. The same Python pipeline powers both interfaces.

The project creates candidate semantic links and visual QA material. It
does not assume that a YOLO box and a MusicXML event share an ID, and it
does not treat geometric proximity as proof. Direct MusicXML matches and
geometry-derived estimates have different statuses, confidence, and review
requirements.

Current package version: **0.6.0**. The package is installed from this GitHub
repository; it is not currently published on PyPI.

- [Quick installation](#installation)
- [Website workflow](docs/WEB_USAGE.md)
- [XML and YOLO class comparison](docs/XML_YOLO_CLASS_COMPARISON.md)
- [CSV schema notes](docs/CSV_SCHEMA_PROPOSAL.md)
- [Report a problem](https://github.com/itsivyma/BPSD-xml-yolo-aligner/issues)

## Workflow at a glance

```text
score images + YOLO TXT + notes.json
                  +
repetition MusicXML + BPSD note CSV
                  +
optional unfolded MusicXML + clean repetition PDF
                  |
                  v
       XML–YOLO alignment pipeline
                  |
       +----------+-----------+
       |          |           |
       v          v           v
 YOLO CSV     XML CSVs    review images
       \          |           /
        \         |          /
         +--- human review --+
                  |
                  v
      corrected BPS-OMR final CSV
```

Use the Streamlit website for an image-first workflow. Use the CLI for scripts,
large datasets, automation, and resumable batch processing.

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
- Build performance order directly from repetition MusicXML `repeat` and
  `ending` structures; unfolded MusicXML is optional validation evidence.
- Give every YOLO class a start/end time candidate from direct MusicXML
  evidence or the nearest score anchor, while explicitly marking estimates.
- Generate slur endpoint candidates and visual QA sheets, including
  scan-only and cross-system cases.
- Keep confirmed, candidate, unresolved, and scan-only results
  distinguishable during review.
- Upload multiple raw score pages through the website and run the same
  alignment pipeline without preparing intermediate CSV files first.
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

For the Beethoven Piano Sonata Dataset v2 layout used during development, use
these source versions:

| Website field | Dataset path or file |
| --- | --- |
| Score images | `finished/Xia/images/*.{jpg,jpeg,png}` |
| YOLO TXT files | `finished/Xia/labels/*.txt` |
| YOLO class map | `finished/Xia/notes.json` |
| Repetition MusicXML | `0_RawData/score_xml_repetitions/*.xml` |
| Unfolded MusicXML (optional validation) | `0_RawData/score_xml_unfolded/*.xml` |
| Clean repetition PDF | `0_RawData/score_pdf_repetitions/*.pdf` |
| BPSD note annotations | `2_Annotations/ann_score_note/*.csv` |

Do not upload Sibelius `.sib` files. Do not use `score_pdf_unfolded` as the
clean page reference because its expanded measure order does not match the
written scan page-for-page. Every image and YOLO TXT must describe the same
page and have the same filename stem, for example `score-04.jpeg` and
`score-04.txt`.

## Installation

Requirements:

- Git;
- Python 3.11, 3.12, 3.13, or 3.14; and
- sufficient local storage for uploaded scores, checkpoints, review images,
  and exported ZIP files.

Clone the public repository:

```bash
git clone https://github.com/itsivyma/BPSD-xml-yolo-aligner.git
cd BPSD-xml-yolo-aligner
```

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
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

To update an existing checkout:

```bash
git pull --ff-only
python -m pip install --upgrade .
```

## Terminal and website

Every processing stage is available through one terminal command:

```bash
bpsd-aligner align --help
bpsd-aligner batch-align --help
bpsd-aligner inventory --help
bpsd-aligner dry-run --help
bpsd-aligner regression-smoke --help
bpsd-aligner xml-export --help
bpsd-aligner combine --help
bpsd-aligner review-queue --help
bpsd-aligner reduce-review --help
bpsd-aligner review-sheets --help
bpsd-aligner render-overlays --help
bpsd-aligner apply-review --help
bpsd-aligner review-eval --help
bpsd-aligner review-dataset --help
bpsd-aligner calibrate-thresholds --help
bpsd-aligner job-admin --help
bpsd-aligner worker --help
bpsd-aligner web --help
```

Start the website locally with:

```bash
bpsd-aligner web
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.
This address is local to the computer running the command. Other people can
clone the repository and run their own local copy; a shared internet URL
requires a separate deployment.

For the complete upload, pairing, review, correction, checkpoint, and download
instructions, read [Website workflow](docs/WEB_USAGE.md).

### Run locally with Docker

Docker is optional. It provides the same CLI and website without installing the
Python dependencies directly on the host:

```bash
docker build -t bpsd-xml-yolo-aligner .
docker run --rm -p 8501:8501 bpsd-xml-yolo-aligner
```

Then open `http://localhost:8501`. For persistent background jobs and
checkpoints, mount a host directory:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/bpsd-jobs:/var/lib/bpsd-aligner" \
  bpsd-xml-yolo-aligner
```

### Website inputs and results

The **Run alignment** tab accepts one or more score pages per run:

1. score images (`.jpg`, `.jpeg`, or `.png`);
2. matching YOLO annotations (`.txt`) with the same filename stems;
3. BPSD written/repetition MusicXML (`.xml` or `.musicxml`);
4. BPSD `ann_score_note.csv`;
5. YOLO `notes.json` class map; and
6. optionally, unfolded MusicXML to validate the XML-derived repeat traversal;
   and
7. preferably, the matching whole-score `score_pdf_repetitions` PDF.

Page numbers can be inferred from filename suffixes such as `-04`, or assigned
consecutively from **First MusicXML page**, then corrected directly in the
pairing table. Click **Align all uploaded pages**. Whole-score MusicXML/repeat
data is prepared once, while each successful page is checkpointed to a
fingerprinted job directory. Exact retries resume from disk, and a portable
checkpoint ZIP can be downloaded and restored with the same input files.
The server also persists `job_status.json` after each page so operators can
inspect the last durable stage after a browser or process interruption. Large
ZIP downloads are opt-in in the UI and are not loaded on every rerun.
Website alignments run in an independent worker by default, continue after the
browser closes, wait for a bounded worker slot, and support cooperative
cancellation between pages. The original synchronous mode remains available
as a fallback checkbox option.

The final website CSV is `bps_omr_final.csv`, with exactly one row per YOLO
bounding box and exactly these columns:

- `class_id`, `x`, `y`, `w`, `h`, `class`, `musical_time`;
- `start_meas`, `end_meas`, `start_note`, `end_note`, `connected_note`,
  `stem_dir`;
- `human_corrected` (`1` only for an applied human correction, otherwise `0`);
- `is_repeated_measure` (`1` when the written measure is played more than once,
  otherwise `0`).

Uncertain, unavailable, and not-applicable semantic values are blank. Machine
review candidates remain visible in the review interface and overlays but are
not written as confirmed values in the final CSV. The website also returns:

`musical_time` is assigned by an explicit class policy rather than a blanket
default. Note/rest/accidental/articulation/dynamic/fingering/stem/beam/slur/tie/
tuplet and related performance glyphs use `0`; clef/key/time-signature/tempo/
term and score-direction labels use `1`. Context-dependent numeral or unknown
classes remain blank and are listed in validation warnings.

- one pre-rendered full-page needs-review overview per score page. Selecting a
  YOLO class generates its uncluttered full-page overlay on demand and caches
  it for later viewing or full-resolution download. Labels retain stable
  `Y{line}`, written measure range, BPSD start/end time, and status. The CLI
  continues to generate dynamics, fingering, all-symbol, review, and per-class
  overlays by default;
- validation JSON and one optional diagnostics ZIP containing review images,
  XML/YOLO source-preserving tables, spans, links, and performance-expanded
  repeat data. These intermediate CSVs remain available for research and
  debugging but no longer appear as separate primary downloads.

The completed-job panel provides an image-first Review workspace plus an
advanced editable table. The workspace shows one symbol at a time with a
full-page highlight and native-resolution context crop, supports page/class/
status queues, and checkpoints confirm, correction, scan-only, wrong-class,
bad-bbox, false-positive, uncertain, and skipped decisions. Checkpoints can be
downloaded and restored as JSON. Each checkpoint is bound to the exact uploaded
files, alignment settings, score ID, and pipeline version by a local SHA-256
fingerprint, so decisions cannot be restored onto a different input batch. The
Applying review validates time order, note IDs, staff, connected notes, and
class mappings. It regenerates every corrected CSV, corrected check images, a
lossless corrections JSON, one corrected-output ZIP, and
overall/per-class/per-field accuracy computed before applying corrections.
Every explicit save is persisted atomically in the fingerprinted job directory.
After a browser or server restart, decisions are restored only when the inputs,
pipeline version, score, and authenticated owner all match. The corrected ZIP's
training manifest also points each reviewed row to its full and cropped PNG.

Small grace-note `notehead*Small`, `accidental*Small`, `stemSmall`,
`beamSmall`, and `flag*Small` boxes use the corresponding MusicXML `<grace>`, `<type>`,
`<accidental>`, `<stem>`, and `<beam>` evidence. Stem/chord and beam rows keep
all BPSD note IDs in `connected_note`; stem direction comes from MusicXML and
duration ends come from the BPSD note annotations. `ottavaBracket` uses paired
`<octave-shift>` directions, while `keyboardPed`, `keyboardPedalPed`, and
`keyboardPedalUp` use MusicXML pedal endpoints. Automatic confirmation requires
a mutual-best geometry assignment, a clear candidate margin, complete required
note IDs, and a conservative class threshold. Missing, cross-system, or
ambiguous evidence stays review-only and is blank in the strict final CSV.

Printed `tempo*`, `term*`, selected special-text classes, and point-like
`dynamicCrescendo`/`dynamicDiminuendo` now match MusicXML `<words>` positions.
Hairpin classes use paired `<wedge>` endpoints, and
`ornamentWiggleTrill` uses paired `<wavy-line>` note endpoints. Extended
`*Long` word boxes remain review-only when MusicXML provides fragmented words
without a trustworthy stop endpoint. XML-free `fingeringSubstitution` and
standalone numerals likewise remain review-only.

Slurs use segment-aware matching. Same-system slurs are confirmed only when the
scan/XML geometry is mutual-best with a clear top-two margin. Every printed
segment of a cross-system slur may share the same complete XML endpoints, but
cross-system and scan/XML-disagreement cases remain review-only.
The shared-score checkpoint also builds one whole-score `xml_spans.csv` before
page alignment. Cross-page slur, tie, wavy-line, wedge, octave-shift, and pedal
segments can therefore show both pages' XML times and note IDs in Review.
Because one page cannot verify the remote endpoint geometrically, these
cross-page candidates require an explicit human confirmation before their
semantic values enter `bps_omr_final.csv`.
When both endpoint pages were uploaded in the same job, Review shows the
start-page and end-page crops side by side. Each crop is independently
clickable, so selecting the left note changes only the start endpoint and
selecting the right note changes only the end endpoint.
The website pre-renders only the per-page needs-review overview. Per-class
inspection images are generated when that class is opened and then cached,
which keeps large multi-page jobs substantially smaller. The CLI retains its
full-overlay default for offline batch inspection.
When the clean repetition PDF is supplied, XML noteheads are localized on the
clean page and transferred measure-locally to the scan before slur/tie scoring.
Structural mismatches fall back safely and are reported as warnings.

Dataset-wide raw alignment remains available through the resumable CLI because
source score collections can be too large for ordinary browser uploads.

### Fixed real-page regression suite

Two fixed manifests deliberately cover different data profiles:

- `regression/representative_pages.json` is the existing `Xia/` 113-class
  test/legacy suite and has the committed semantic baseline;
- `regression/finished_xia_pages.json` is the production `finished/Xia/`
  162-class suite. Create its baseline only after human review.

Both select ten varied real pages covering fingerings, slurs, ties, tuplets,
hairpins, ottava, pedal, wavy-line, repeat, cross-system, and cross-page cases.
The runner validates `notes.json` against the declared class count, preventing
accidental cross-profile tests. Dataset paths remain outside this repository
and are resolved relative to `--dataset-root`.
The baseline includes row-level semantic digests of geometry, class, musical
time, endpoint note IDs, staff, written measures, and cross-page assignments;
therefore an endpoint regression cannot pass merely because aggregate row
counts stayed unchanged. Per-class digests identify the affected symbol class.

Run against the reviewed baseline committed with this repository:

```bash
bpsd-aligner regression-smoke \
  --manifest regression/representative_pages.json \
  --dataset-root "/path/to/中研院" \
  --output-dir output/regression-smoke \
  --baseline regression/representative_baseline.json \
  --resume
```

To evaluate actual correctness rather than only stability, export reviewed
decisions as normalized ground truth and add
`--ground-truth /path/to/evaluation_ground_truth.csv`. The runner writes
`ground_truth_accuracy.json`; `regression/ground_truth_template.csv` is the
portable schema. Private score annotations remain outside Git.

Only after reviewing an intentional result change, update that baseline:

```bash
bpsd-aligner regression-smoke \
  --manifest regression/representative_pages.json \
  --dataset-root "/path/to/中研院" \
  --output-dir output/regression-smoke \
  --baseline regression/representative_baseline.json \
  --resume --update-baseline
```

Each passing page gets its own input-and-code-bound checkpoint. Website upload
fingerprints, background requests, job manifests, page checkpoints, and
portable checkpoint ZIPs use the same code-signature rule. Interrupted,
missing, stale, failed, or code-outdated pages are rerun; unchanged passing
pages are resumed. The
runner prints shared-score and page-stage progress, skips QA image generation,
and writes `regression_summary.csv` plus `regression_report.json`. The CSV keeps
full per-class diagnostics; the compact reviewed baseline tracks page metrics
and only each page's required representative classes. A tracked change makes
the command fail until it is reviewed and the baseline is explicitly updated.

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
bpsd-aligner align --help
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
python -m pip install ".[dev]"
python -m pytest -v
```

The CI workflow tests Python 3.11–3.14 and verifies both the package entry point
and Docker website health check.

## Troubleshooting

### `bpsd-aligner: command not found`

Activate the same virtual environment in which the package was installed, then
reinstall and verify it:

```bash
source .venv/bin/activate
python -m pip install --upgrade .
python -m bpsd_aligner.cli --help
```

On Windows, replace the first command with
`.venv\Scripts\Activate.ps1`.

### `ModuleNotFoundError: No module named 'bpsd_aligner'` on macOS

Remove a broken editable installation and use a normal installation. This also
avoids Python startup problems caused by a hidden editable-install `.pth` file
on some macOS/Python configurations:

```bash
python -m pip uninstall -y bpsd-xml-yolo-aligner
python -m pip install .
python -c "import bpsd_aligner; print(bpsd_aligner.__version__)"
bpsd-aligner --help
```

### The website address does not open

Keep the terminal running after `bpsd-aligner web`. The local website exists
only while that process is active. Check `http://localhost:8501`, not another
computer's `localhost`. A shared internet URL requires a separate deployment.

### Upload or alignment problems

- Confirm every image/TXT pair has the same filename stem.
- Use repetition MusicXML for written-score geometry and unfolded MusicXML for
  performance order.
- Use the clean repetition PDF, not the unfolded PDF.
- Check the page-pairing table and printed system-start measures before running.
- Retain `validation.json`, `job_status.json`, and the resumable checkpoint ZIP
  when reporting a failed or interrupted job.

## Project structure

```text
.
├── bpsd_aligner/
│   ├── cli.py                 # installed command router
│   ├── web.py                 # Streamlit interface
│   ├── web_pipeline.py        # multi-page website pipeline
│   ├── web_worker.py          # independent background worker
│   ├── job_store.py           # persistent job/checkpoint state
│   ├── review_corrections.py  # review validation and regeneration
│   ├── review_dataset.py      # normalized human-review learning rows
│   ├── calibrate_thresholds.py # precision/recall threshold suggestions
│   ├── regression_smoke.py    # fixed real-page resumable regression
│   ├── job_admin.py           # privacy-safe deployment status/cleanup
│   ├── pdf_utils.py            # clean-PDF rendering helpers
│   ├── bps_omr_schema.py       # final CSV class semantics
│   ├── class_registry.py        # central class family/timeline policy
│   ├── geometry.py              # staff/system/barline geometry
│   ├── overlay.py               # review overlay rendering
│   ├── thresholds.py           # per-class confidence policy
│   └── web_utils.py            # upload and UI helpers
├── docs/
│   ├── WEB_USAGE.md
│   ├── XML_YOLO_CLASS_COMPARISON.md
│   └── CSV_SCHEMA_PROPOSAL.md
├── tests/
├── bps_xml_alignment.py          # one-page alignment core
├── dataset_dry_run.py            # resumable dataset workflow
├── combine_yolo_xml.py           # lossless combined exports
├── xml_export.py                 # MusicXML event/node exports
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## Data and privacy

This repository does not include score images, MusicXML files, YOLO
annotations, BPSD annotations, human-review CSV files, generated QA
images, exported spreadsheets, or other dataset files.

Machine-specific prototypes and input paths are intentionally excluded
from version control.

For deployment, mount `BPSD_ALIGNER_JOB_DIR` on persistent storage. Browser
jobs enforce per-file, total-byte, file-count, decoded-image, and checkpoint
expansion limits. MusicXML accepts the standard Recordare `DOCTYPE`, while
entity expansion and external resource resolution remain disabled.
Per-class auto-accept thresholds can be overridden with
`BPSD_ALIGNER_THRESHOLDS=/path/to/thresholds.json`.

Every applied website review also exports `review_training_rows.csv`. Combine
reviewed jobs and calibrate only after collecting enough human evidence:

```bash
bpsd-aligner calibrate-thresholds \
  --review-dataset /path/to/job-1/review_training_rows.csv \
  --review-dataset /path/to/job-2/review_training_rows.csv \
  --output-dir output/threshold-calibration
```

Normalized legacy ground truth can enter the same workflow without hand-made
JSON. The page ID is read from either `page_id` or
`review_candidate_set_id=page:Yline` in the detailed CSV:

```bash
bpsd-aligner review-dataset \
  --predictions /path/to/page_alignment_detailed.csv \
  --ground-truth /path/to/evaluation_ground_truth.csv \
  --output-dir output/review-dataset
```

Rows not explicitly marked `confirmed`, and expected fields left blank as
unknown, are excluded from threshold labels.

The default guard requires 200 reviewed rows overall, 20 per class family, and
at least 10 accepted examples whose 95% Wilson precision lower bound reaches
0.98. Observed precision alone can no longer mark a threshold ready.
Insufficient groups produce no override. Review `thresholds.recommended.json`, set it with
`BPSD_ALIGNER_THRESHOLDS`, and rerun the fixed real-page regression before
deployment. This calibrates a decision threshold; it does not turn the
heuristic `confidence` value into a probability.

`confidence` is retained for compatibility, but it is a heuristic alignment
score rather than a calibrated probability. Detailed exports repeat it as
`match_score`, set `confidence_calibrated=false`, and expose the available
evidence through `geometry_score`, `candidate_margin`, `count_agreement`, and
`xml_time_confirmed`. Default auto-accept thresholds are intentionally
conservative: fingering `0.95`, dynamics and slurs `0.85`, and ties,
articulations, fermatas, ornaments, and tuplets `0.80`.

Before publication, every final row is checked for normalized YOLO geometry,
the class-specific `musical_time` flag, paired and ordered start/end times,
integer note IDs, valid JSON `connected_note`, first/last-note consistency,
stem-only `stem_dir`, binary review/repeat flags, and equality between final
row count and YOLO box count. Violations make validation fail instead of being
silently exported.

Detailed alignment CSVs store only a small `review_candidate_set_id`. Unique
clickable XML notes and set membership are written to
`review_note_candidates.csv` and `review_candidate_sets.csv`; the website
hydrates them only in memory. This avoids repeating full-page note JSON in
every YOLO row while preserving Review behavior.
In addition, a machine row must have both `alignment_status=matched` and
`xml_time_confirmed=true`; a high geometry score alone cannot populate final
time or note fields.

Set `BPSD_ALIGNER_ACCESS_TOKEN` to a long random secret for a single shared
gate. For separate user job namespaces, prefer `BPSD_ALIGNER_USERS_FILE` with
an external JSON mapping of usernames to distinct tokens. Set
`BPSD_ALIGNER_JOB_RETENTION_HOURS=168` to remove inactive unlocked jobs after
seven days; cleanup is disabled when the variable is unset. See
`.env.example` for the supported deployment variables. Public deployments
still need HTTPS and reverse-proxy rate limiting.

Use `BPSD_ALIGNER_MAX_CONCURRENT_JOBS` to cap simultaneous alignment workers
(default `2`) and `BPSD_ALIGNER_MAX_PAGES` to cap pages per browser job
(default `200`). Oversized jobs are rejected before alignment begins.
Every state transition is appended to each job's `job_events.jsonl`. Operators
can inspect state counts without exposing uploaded filenames or full user IDs:

```bash
bpsd-aligner job-admin status
bpsd-aligner job-admin prune --retention-hours 168
```

## Support and contributions

Use [GitHub Issues](https://github.com/itsivyma/BPSD-xml-yolo-aligner/issues)
for reproducible bugs, installation problems, and feature requests. Include:

- operating system and Python version;
- package version from `bpsd-aligner --version`;
- the command or website stage that failed;
- the error traceback; and
- `validation.json` or `job_status.json` when available.

Do not attach copyrighted score pages or dataset files unless their license
allows redistribution. Small synthetic examples are preferred. Pull requests
should include focused tests and pass `python -m pytest -v`.

## Versioning and citation

Until formal GitHub Releases are published, record both the package version and
Git commit SHA when using the software in an experiment:

```bash
bpsd-aligner --version
git rev-parse HEAD
```

The BPSD dataset is a separate work. Cite the dataset using the citation and
license supplied with the dataset distribution; citing this software does not
replace the dataset citation. A software citation may identify the repository,
package version, commit SHA, and access date.

## Current scope

The lossless combined CSV preserves every YOLO bbox and every MusicXML event,
adds readable XML-only class names, and is sorted by musical start/end time. It
does not claim that every YOLO/XML pair is correct: candidate, ambiguous,
unresolved, XML-only, and YOLO-only states remain explicit until reviewed. If
the structural repeat traversal disagrees with the optional unfolded XML, the
affected rows are forced into review and uncertain semantic values remain blank
in the strict final CSV.

## License

The software in this repository is available under the MIT License. See
[`LICENSE`](LICENSE). This license applies only to the software; it does not
grant rights to the BPSD dataset, Beethoven score editions, scans, MusicXML,
YOLO annotations, or other uploaded material.

## Disclaimer

This is an independent annotation-alignment and QA utility. Dataset
files must be obtained and used according to their original licenses
and terms.
