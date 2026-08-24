# Website workflow

The website aligns one or more score pages from the same piece and returns one
strict BPS-OMR CSV plus full-page review images.

## Start the website

From an activated virtual environment in the repository:

```bash
python -m pip install .
bpsd-aligner web
```

Open the Streamlit address printed in the terminal, normally
`http://localhost:8501`.

## Prepare one piece

Every image and YOLO TXT must describe the same page and share the exact filename
stem, for example `score-04.jpeg` and `score-04.txt`. Upload these required
files in **Run alignment**:

| Upload | Purpose |
| --- | --- |
| Score images | Detect systems/staves and render full-page review overlays |
| YOLO TXT files | Preserve every class ID and normalized bounding box |
| Written/repetition MusicXML | Supply measures, notes, pitch, staff, voice, dynamics, ties, slurs, and other score structure |
| Clean repetition PDF (recommended) | Supply clean system, barline, and notehead geometry for slur/tie endpoint alignment |
| `ann_score_note.csv` | Supply the official BPSD note IDs and musical timeline |
| `notes.json` | Translate every YOLO class ID into its class name |

The system reads standard forward/backward repeats and first/second endings
directly from repetition MusicXML. Unfolded MusicXML is optional validation
evidence: it can reveal a disagreement, but it never changes or fills the
structural traversal.

Use these exact dataset versions in the website:

| Website field | Dataset source |
| --- | --- |
| Score images | `finished/Xia/images` |
| YOLO TXT files | `finished/Xia/labels` |
| Repetition MusicXML | `0_RawData/score_xml_repetitions/*.xml` |
| Unfolded MusicXML | `0_RawData/score_xml_unfolded/*.xml` |
| Clean repetition PDF | `0_RawData/score_pdf_repetitions/*.pdf` |
| BPSD note annotations | `2_Annotations/ann_score_note/*.csv` |
| YOLO class map | `finished/Xia/notes.json` |

The website does not parse `.sib` files. Do not upload files from
`score_sibelius_repetitions` or `score_sibelius_unfolded`. Upload the single
whole-score PDF from `score_pdf_repetitions`; the website validates its page
count and renders only the requested pages. Do not upload `score_pdf_unfolded`,
whose expanded form cannot be aligned page-for-page with the repetition scan.

Enter a stable **Score ID**. The website can infer each MusicXML page from the
final number in a filename, or assign consecutive pages starting from **First
MusicXML page**. The pairing table must be correct before alignment begins.

Some scanned editions place page or system breaks differently from the
repetition MusicXML. In that case, enter the printed first measure of every
scanned system in **掃描譜每行起始小節**, from top to bottom, separated by
commas. For example, Op. 90 scan page 6 uses
`198, 202, 206, 211, 223, 235`. Optionally enter the printed final measure in
**掃描譜本頁最後小節**; when the following consecutive page also has anchors,
the website derives this value automatically. Leave both fields blank when the
layouts agree. These anchors change only page geometry: pitch, staff, musical
time, and note ID still come from the same-numbered MusicXML measure.

## Run and review

Click **Align all uploaded pages**. The whole-score repeat mapping, XML nodes,
and XML events are prepared once. Each page then reports six alignment stages:

1. validate uploads;
2. reuse the shared written/unfolded mapping;
3. align YOLO boxes;
4. reuse the shared MusicXML event and node export;
5. save the page output and checkpoint;
6. validate and package outputs.

After completion, first choose one score page under **Review images**. The page
label includes its number of rows needing review. **Overview** shows only that
page's `review_overlay`. Under **One class at a time**, choose one YOLO class to
see every symbol of that class on the selected full page, including both direct
matches and review cases. Each class-image label shows its stable YOLO ID,
written measure range, BPSD start/end time, and status.
The website pre-renders only the overview. It generates the selected class
image on demand and caches it, so multi-page jobs do not store an image for
every class before those images are actually needed.

Each completed page is checkpointed atomically under a fingerprinted job
directory. Retrying the exact same batch resumes completed pages even after a
server restart when `BPSD_ALIGNER_JOB_DIR` is persistent. Download the
**resumable alignment checkpoint ZIP** for portable recovery; select the same
inputs and upload that ZIP in a later session. Original uploads are deliberately
excluded from the checkpoint archive.

`job_status.json` is updated atomically after upload preparation, shared-score
preprocessing, and every completed or resumed page. It records the current
stage, completed/total pages, timestamps, and a failure message when relevant.
This makes server-side diagnosis possible even when the browser disconnects.

Multi-page uploads run in an independent background worker by default. Closing
the browser does not terminate that worker. While the page is open, status
refreshes every two seconds and completed outputs load automatically after the
durable result has been finalized. **Refresh background status** remains
available as a manual fallback. Jobs wait for a configured worker slot instead
of failing immediately. Cancellation is cooperative: **Request cancellation**
stops the job before the next page, without deleting completed page
checkpoints.

### Human corrections

Use **Review workspace** after alignment. It presents one symbol at a time with
a full-page location and an unresampled context crop. Filter by page, class, or
queue; use Previous/Next or jump directly to an item. Low-confidence unresolved
items are placed before completed decisions. The advanced table remains below
for bulk editing.

On desktop, review uses a two-column layout: the enlarged score crop stays in
a sticky left panel while note selection, manual fields, and save actions stay
in the right panel. Selecting a candidate rerenders the green `C` preview on
the left without requiring the reviewer to scroll back up. On narrow screens,
the layout stacks and sticky positioning is disabled to avoid covering inputs.

In the focus images, the red rectangle marks the YOLO symbol. A hollow cyan
circle marks its linked notehead (`N` for one note or `S` for a range start),
and a hollow purple circle marks a range end (`E`). The crop automatically
includes both the symbol and linked notehead without resampling. If endpoint
pixel coordinates are unavailable, the workspace shows an explicit warning
instead of inventing a marker.

For a point symbol linked to the wrong note, orange numbers mark the eight
nearest XML note candidates. Under step 2, choose the correct orange number by
its measure, staff, pitch, and BPSD time; confirm the green `C` marker is on the
intended notehead, then click **儲存所選音頭更正**. This single action fills
start/end time, start/end note ID, connected note IDs, and staff, saves the
decision, and advances to the next item. The original cyan machine choice
remains visible for comparison. Span symbols such as slurs and ties continue
to use separate start/end correction fields under the advanced expander.

The workspace supports these decisions:

- `confirm`: retain the machine values; `human_corrected` remains `0`;
- `correct`: edit time, note IDs, connected notes, staff, or comment, then apply;
- `scan_only`: keep the YOLO symbol but leave unavailable XML semantics blank;
- `wrong_class`: supply both corrected class ID and class name;
- `bad_bbox`: remove the invalid box from corrected CSV and retain it in the
  corrections JSON for reannotation;
- `not_a_symbol`: remove the false positive from corrected CSV;
- `uncertain` or `skipped`: preserve the conservative machine output without
  using the row as accuracy ground truth;
- `pending`: make no decision.

Unknown values must remain blank. Applying decisions validates time order,
staff, note-ID existence, connected-note consistency, and corrected class
ID/name consistency. It then rebuilds the corrected BPS-OMR CSV, YOLO CSV,
combined master, alignment links, timelines, all-information CSV, corrected
review images, and one corrected-output ZIP. Accuracy is measured against the
original machine values before corrections and is reported overall, per class,
and per field. Blank ground-truth fields are not scored.
It also exports `review_training_rows.csv`, joining each reviewed decision with
the machine score, geometry evidence, candidate margin, XML confirmation flag,
original values, and corrected values. Rows such as `fingeringSubstitution`,
wrong-class detections, and XML-unconfirmed assignments are explicitly marked
as candidates for continued human review or a future learned relation model.
Website-generated rows also contain `review_full_image` and
`review_crop_image` paths to matching high-resolution PNGs inside the
corrected ZIP, so a future model can consume the exact reviewed crop.

Every explicit Review save is written atomically to the persistent job store.
When the same authenticated user reopens the exact alignment, the website
validates its input fingerprint, pipeline version, score ID, and owner before
restoring decisions automatically. Download `review_checkpoint.json` for a
portable backup, or restore it under **Resume from review checkpoint** for the
same alignment. The alignment checkpoint ZIP also carries server-side Review
state but never the original uploaded files.
Checkpoint schema 2.0 stores the alignment fingerprint, score ID, and pipeline
version. Restore is rejected if any uploaded image, YOLO TXT, MusicXML, BPSD
CSV, `notes.json`, clean PDF, alignment option, or pipeline version differs.
The fingerprint is calculated locally from file contents and settings; source
files are not transmitted elsewhere for validation. Older checkpoints without
an alignment fingerprint cannot be restored automatically.
New alignment runs retain original uploaded page bytes in the session so the
workspace crop comes from the scan itself. An older session without those bytes
falls back to an overlay and asks for one current-version rerun.

## Downloads

- **Final BPS-OMR CSV** is the only primary CSV download. It contains one row
  per YOLO box and the 13 fields defined by `BPS-OMR annotations.pdf`, plus
  `human_corrected` and `is_repeated_measure`.
- Uncertain, unavailable, and not-applicable semantic values are blank. Review
  candidates are not silently promoted into final timing or note fields.
- `musical_time` comes from an explicit class policy. Known note/performance
  glyph families use `0`; clef/key/time-signature/tempo/term and direction
  labels use `1`. Context-dependent numerals and unknown future classes remain
  blank and appear in validation warnings.
- Final validation checks bbox ranges, class/timeline consistency, time order,
  note-ID pairs, `connected_note` JSON and endpoints, stem direction, binary
  added flags, and exact equality between final rows and YOLO boxes.
- **Optional diagnostics ZIP** contains full-page review images, validation,
  YOLO alignment details, XML events/nodes, the combined timeline, XML spans,
  links, and the performance-expanded repeat timeline. These are retained for
  reproducibility without crowding the main download area.
- **Resumable alignment checkpoint ZIP** separately contains page checkpoints
  and derived outputs needed to continue the same fingerprinted job.

Large final and checkpoint ZIPs are loaded only after selecting **Prepare …**
beside the corresponding download. This prevents every ordinary Streamlit
rerun from reading both archives into memory.

Existing class-specific review CSVs can be normalized into one evaluation set:

```bash
bpsd-aligner review-eval \
  --review-dir /path/to/alignment/training \
  --predictions /path/to/page_alignment_detailed.csv \
  --output-dir /path/to/evaluation-output
```

Omit `--predictions` to build only `evaluation_ground_truth.csv`.

To rebuild the normalized learning rows later from downloaded artifacts, pair
each detailed prediction CSV with its corrections JSON:

```bash
bpsd-aligner review-dataset \
  --predictions /path/to/page_alignment_detailed.csv \
  --corrections /path/to/human_corrections.json \
  --output-dir /path/to/review-dataset
```

After accumulating at least 200 reviewed rows across representative scores:

```bash
bpsd-aligner calibrate-thresholds \
  --review-dataset /path/to/review-dataset/review_training_rows.csv \
  --output-dir /path/to/threshold-calibration
```

Do not deploy the suggested override blindly. Inspect false accepts, the Wilson
lower bound, per-family sample sizes, and the fixed ten-page regression first.

The selected needs-review overview and each on-demand class image can also be
downloaded directly as a full-resolution PNG. The CLI still generates the
all-symbol and other complete overlay sets for offline audits.

## Deployment controls

- `BPSD_ALIGNER_ACCESS_TOKEN` enables a password-style access gate for the
  whole Streamlit app. Leave it unset only for local/private use. Use a long,
  random secret supplied by the hosting platform; never commit the value.
- `BPSD_ALIGNER_USERS_FILE` points to an external JSON object such as
  `{"reviewer-a":"long-random-token-a"}`. When set, username and token are
  both required and the username hash becomes part of the job fingerprint, so
  different users cannot reuse each other's server-side jobs. It takes
  precedence over the single shared token.
  Keep the file outside the repository (local `.bpsd-users.json` is ignored).
- `BPSD_ALIGNER_JOB_DIR` selects persistent job/checkpoint storage. Mount this
  path on a persistent volume in production.
- `BPSD_ALIGNER_JOB_RETENTION_HOURS` opts into automatic removal of inactive,
  unlocked fingerprinted jobs older than the configured number of hours. It
  is disabled when unset. `168` retains jobs for seven days.
- `BPSD_ALIGNER_MAX_CONCURRENT_JOBS` limits simultaneous alignment workers
  (default `2`), and `BPSD_ALIGNER_MAX_PAGES` limits one browser job (default
  `200`). Jobs above the limit fail before alignment begins.
- `BPSD_ALIGNER_WORKER_QUEUE_TIMEOUT` controls how long an independent worker
  waits for a free slot before marking the job failed (default `3600` seconds).
- Each file is limited to 200 MB, each job to 500 files and 1 GB total upload,
  each decoded score image to 100 million pixels, and each restored checkpoint
  to 2 GB uncompressed.
- Uploaded MusicXML accepts the standard Recordare `DOCTYPE` declaration used
  by MusicXML 3.x, but entity expansion and external resource resolution remain
  disabled. The parser never downloads the referenced MusicXML DTD.
- Restored checkpoint ZIPs must match both the exact input fingerprint and the
  current pipeline version. Shared XML checkpoints are also bound to hashes of
  repetition XML, unfolded XML, and BPSD note annotations.
- `BPSD_ALIGNER_THRESHOLDS=/path/to/thresholds.json` overrides per-class
  auto-accept thresholds. Values must be between 0 and 1; unspecified classes
  retain conservative defaults.
- Each job records structured state transitions in `job_events.jsonl`.
  `bpsd-aligner job-admin status` reports operational state without uploaded
  filenames or full owner identifiers; `job-admin prune --retention-hours N`
  performs explicit cleanup.

The access token is a basic deployment gate, not a replacement for HTTPS,
identity-aware authentication, or reverse-proxy rate limiting on a public
internet service. The supplied Docker image runs as a non-root user, persists
jobs under `/var/lib/bpsd-aligner`, and exposes a Streamlit healthcheck.

## Accuracy boundary

The website preserves all source records, but preservation is different from a
confirmed match. Dynamics, staccato, fermata, slur, tie, selected ornaments,
tuplets, grace-note small noteheads/accidentals/stems/beams/flags, ottava brackets,
and pedal endpoints use direct MusicXML evidence plus geometry. Fingering links are optional geometric
candidates because the source MusicXML does not contain fingering elements.
Every remaining YOLO class receives a geometry-derived start/end time candidate
when a MusicXML note or rest anchor exists. These estimates use `status=review`,
stay visible in the review table and orange overlay, and are never presented as
confirmed XML matches.

The displayed `confidence` value is a heuristic match score, not a calibrated
probability. Detailed CSVs also provide `match_score`,
`confidence_calibrated=false`, `geometry_score`, `candidate_margin`,
`count_agreement`, and `xml_time_confirmed` so a reviewer can see why a match
was or was not accepted. Count equality by itself cannot create a perfect
dynamic match. The default thresholds are fingering `0.95`, small
noteheads/accidentals/stems/flags `0.92`, small beams/ottava/pedal and
tempo/term words `0.90`,
dynamics/slurs `0.85`, and ties, articulations, fermatas, ornaments, and
tuplets `0.80`.

Grace-note `notehead*Small`, `accidental*Small`, `stemSmall`, `beamSmall`, and
`flag*Small`
classes use explicit MusicXML grace/type/accidental/stem/beam evidence.
`ottavaBracket` uses paired `<octave-shift>` endpoints and pedal glyphs use
`<pedal>` endpoints. Automatic confirmation additionally requires mutual-best
geometry, a clear candidate margin, and every required BPSD note ID. Ambiguous,
incomplete, cross-system, or absent XML evidence stays in Review and its strict
final-CSV semantic cells remain blank.

Tempo, term, and supported special-text boxes use normalized MusicXML
`<words>` evidence. Point crescendo/diminuendo words use their words position;
hairpins use paired `<wedge>` endpoints; trill wiggles use paired
`<wavy-line>` note endpoints. A visually extended `*Long` word stays in Review
when MusicXML has no reliable stop endpoint. Fingering substitutions and
standalone numerals also stay in Review because the source XML cannot identify
their intended role.

Slur matching compares each YOLO curve with a same-system XML slur segment,
including separate start/end segments for cross-system slurs. Automatic
confirmation requires a high geometry score, a clear margin over the next
candidate, and a mutual-best assignment. Cross-system segments and scan-only
or scan/XML-disagreement slurs stay review-only, so their candidate endpoints
do not populate the strict final CSV.

Whole-score preprocessing writes `xml_spans.csv` into the resumable shared
checkpoint. This preserves remote endpoints for cross-page slurs, ties,
wavy-lines, wedges, octave shifts, and pedal spans. The Review workspace can
display those complete XML times and note IDs, but the row remains review-only
until a human confirms it because the other page is not visible to the
single-page geometry matcher.
If both endpoint pages are part of the uploaded batch, the workspace displays
their native-resolution endpoint crops side by side. Click the start-page crop
to replace the start note, or the end-page crop to replace the end note. If an
endpoint page was not uploaded, its XML values remain available but the UI
asks for manual input instead of inventing image geometry.

When a clean repetition PDF is supplied, the aligner uses its detected systems
and barlines as page geometry, snaps XML endpoints to clean noteheads, transfers
those locations measure-by-measure to the scan, and snaps again on the scan.
The reference is used only when its systems and measures agree with the scan
and repetition MusicXML. Otherwise the page falls back to MusicXML-width
geometry and records a warning; uncertain endpoints still remain blank in the
strict final CSV.

The system may use richer XML evidence internally, but only the bounding-box
schema above is exported as the final CSV.

## Repetition and unfolded MusicXML

- **Repetition MusicXML** represents the written/printed score. Repeated
  passages normally appear once. The aligner directly traverses standard
  `<repeat direction="forward|backward">` and `<ending number="...">`
  structures, including an omitted forward marker that means repeat from the
  beginning. Its page and system layout is used when it agrees with the scan.
  Printed-measure anchors override edition-specific page and system breaks
  without changing the corresponding MusicXML note data.
- **Unfolded MusicXML** represents performance order. Repeated passages are
  expanded into separate occurrences so its timeline follows the BPSD note
  annotations from beginning to end.

The aligner uses repetition XML for scan geometry, written measure identity,
and the primary performance traversal. It assigns every repeated written event
one occurrence per traversal visit and then matches those occurrences to the
BPSD timeline. Unfolded XML, when supplied, checks the resulting sequence but
does not construct it. D.C., D.S., Coda, Fine, malformed repeat nesting, or an
unfolded disagreement is reported for review instead of being silently guessed.
Rows with `repeat_mapping_status=structural_unfolded_disagreement` cannot be
automatically accepted; their unconfirmed time/note semantics remain blank in
the strict final CSV until a human correction is saved.
