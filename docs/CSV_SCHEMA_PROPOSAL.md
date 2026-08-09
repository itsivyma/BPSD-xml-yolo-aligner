# BPS-OMR Alignment CSV Schema Proposal

This proposal preserves the official BPS-OMR columns first, then adds
traceability, richer MusicXML semantics, quality status, and human review data.
One row always represents one YOLO bounding box.

## 1. Official BPS-OMR columns

| Column | Type | Meaning |
| --- | --- | --- |
| `class_id` | integer | YOLO class ID |
| `x`, `y`, `w`, `h` | float | Normalized YOLO bounding box |
| `class` | string | Human-readable class name |
| `musical_time` | `0`, `1`, blank | `0`: on musical timeline; `1`: outside musical timeline; blank: not classified by the specification |
| `start_meas`, `end_meas` | decimal or blank | BPSD measure-position interval; equal for point events |
| `start_note`, `end_note` | note ID, `NA`, or blank | First/last connected BPSD note; `NA` when note connection does not apply |
| `connected_note` | JSON list, `NA`, or blank | Every connected BPSD note ID |
| `stem_dir` | `0`, `1`, `NA`, or blank | Stem only: `0` down, `1` up |

Value policy:

- `NA` means the field does not apply to that symbol.
- Blank means applicable but unknown, unresolved, or not yet reviewed.
- BPSD note IDs use the zero-based data-row index in `ann_score_note/*.csv`.
- Lists are serialized as valid JSON, for example `[486, 490, 493, 494]`.

## 2. Stable identity and source columns

| Column | Meaning |
| --- | --- |
| `dataset_id` | Dataset/version identifier |
| `score_id` | Sonata movement ID, e.g. `Beethoven_Op090-01` |
| `page_id` | Image stem, e.g. `Beethoven_Op090-01-01` |
| `scan_page` | Numeric scan page |
| `yolo_line` | Original 1-based TXT line number |
| `bbox_id` | Stable ID: `{page_id}:Y{yolo_line}` |
| `image_path`, `yolo_path`, `xml_path`, `unfolded_xml_path`, `sibelius_path`, `bps_notes_path` | Source provenance; Sibelius is retained as an archival/fallback source |
| `image_sha256`, `yolo_sha256` | Input checksums for reproducibility |

## 3. MusicXML/BPSD semantic extension

| Column | Meaning |
| --- | --- |
| `xml_measure` | Printed/repetition MusicXML measure number |
| `beat_position` | Beat within measure, starting at 1 |
| `duration_measures` | `end_meas - start_meas` when defined |
| `staff`, `voice` | MusicXML staff and voice |
| `target_type` | `note`, `chord`, `rest`, `note_group`, `measure_position`, `span`, or `scan_only` |
| `note_ids` | JSON list of all connected note IDs; explicit alias of `connected_note` for analysis |
| `pitches` | JSON list of spelled pitches, e.g. `["F#4", "A4"]` |
| `midi_pitches` | JSON list of MIDI pitches |
| `xml_element` | MusicXML element used as semantic evidence |
| `xml_symbol` | MusicXML symbol/value, when available |
| `placement`, `orientation` | Above/below or start/stop orientation metadata |
| `time_signature` | Active BPSD/MusicXML time signature |
| `articulation` | MusicXML articulation value |
| `grace` | Grace-note flag from BPSD |

## 3a. Repeat-aware measure identity

The printed score and BPSD performance timeline use different measure
identities when a passage repeats. Both identities must be retained.

| Column | Meaning |
| --- | --- |
| `written_measure` | Measure number printed in the repetition-preserving score |
| `written_measure_index` | Stable zero-based measure index in `score_xml_repetitions` |
| `unfolded_measure` | Corresponding measure in `score_xml_unfolded` |
| `repeat_status` | `not_repeated`, `repeat_start`, `repeat_body`, `repeat_end`, `volta`, or `unknown` |
| `is_repeated_measure` | Boolean; true when the printed measure occurs more than once on the unfolded timeline |
| `repeat_occurrence_count` | Number of unfolded performance occurrences |
| `repeat_occurrences_json` | Lossless list of occurrence, unfolded interval, and occurrence-specific note IDs |
| `repeat_group_id` | Stable ID shared by all measures in the same repeat region |
| `volta_numbers` | JSON list of applicable endings, e.g. `[1]` or `[2]` |
| `repeat_source` | Evidence used: repetition XML, unfolded XML, Sibelius export, or human review |

`start_meas` and `end_meas` remain the official unfolded BPSD timeline and must
not be overwritten by `written_measure`. In the canonical one-row-per-bbox CSV,
they represent the first performance occurrence; `repeat_occurrences_json`
preserves every occurrence without duplicating the bounding box. A separate
performance-expanded CSV duplicates the row per occurrence and provides scalar
`repeat_occurrence`, `start_meas`, `end_meas`, and occurrence-specific note IDs.

For chord-wide marks such as staccato or fermata, all notes in the same
MusicXML `<chord/>` group are written to `connected_note`, `note_ids`, and
`pitches`. Simultaneous notes from a different voice are not merged.

## 4. Alignment and uncertainty columns

| Column | Meaning |
| --- | --- |
| `page_mapping_status` | `direct`, `system_aligned`, `needs_review`, or `unmapped` |
| `movement_scope_status` | `in_bpsd_scope`, `outside_bpsd_scope`, `mixed_page`, or `unknown` |
| `page_segment_id` | Stable segment ID when one scan page spans multiple movements |
| `scan_system_index`, `xml_system_index`, `staff_index` | Scan/XML geometry assignment |
| `match_source` | Evidence/method used for the match |
| `confidence` | Calibrated score in `[0,1]` |
| `candidate_rank` | Candidate rank; confirmed match is normally 1 |
| `alternatives_json` | Other candidates and scores |
| `alignment_status` | `matched`, `candidate`, `ambiguous`, `scan_only`, `xml_missing`, or `unresolved` |
| `geometry_status` | Barline/system/notehead geometry diagnostic |
| `error_code`, `error_message` | Machine-readable and readable failure information |
| `pipeline_version` | Code/schema version that created the row |

## 5. Human-review columns

| Column | Meaning |
| --- | --- |
| `review_status` | `not_required`, `needs_review`, `confirmed`, `corrected`, `rejected`, or `deferred` |
| `human_approved` | Boolean |
| `reviewer` | Reviewer ID/name |
| `reviewed_at` | ISO 8601 timestamp with timezone |
| `original_candidate_json` | Machine result before correction |
| `corrected_value_json` | Human correction |
| `review_source` | Review sheet/UI/import source |
| `comment` | Free-text explanation |

Machine output is never marked `human_approved=true`. Existing first-page
manual decisions are imported with provenance and protected from overwrite.

Rows outside the movement covered by BPSD retain their YOLO geometry and class,
use blank values for unknown BPS-OMR semantic fields, and set
`movement_scope_status=outside_bpsd_scope` plus
`alignment_status=xml_missing`. They are not silently discarded.

## 6. Output files

The pipeline should produce:

1. One detailed CSV per scan page.
2. One official-plus-extension CSV per sonata, as required by the PDF.
3. One all-dataset master CSV.
4. One review CSV containing only uncertain rows.
5. JSON reports for page mapping, validation, errors, and pipeline versions.
6. One performance-expanded CSV with one row per bbox occurrence, for repeated
   passages and playback-timeline analysis.

## 7. Human visual-review images

The primary review artifact is an alignment check sheet like the first-page
manual workflow, not merely a raw YOLO overlay. Each review panel must show:

- stable `Y{yolo_line}` ID and class;
- blue original YOLO box;
- green matched note, chord notes, span endpoints, or other semantic target;
- connector lines between the YOLO symbol and its proposed target(s);
- printed/XML measure, pitch(es), BPSD time and beat, note ID(s), staff, and
  confidence/status;
- repeat occurrence information when the written measure is repeated;
- human-review state (`needs_review`, `confirmed`, `corrected`, etc.).

Review renderers are class-aware:

1. point-to-note panels for fingering, staccato, fermata, and similar marks;
2. multi-note/chord panels that display every connected chord note;
3. start/end panels for slur, tie, hairpin, ottava, and other spans;
4. cross-system panels when endpoints occur on different systems/pages;
5. measure-position or scan-only panels when no connected note applies.

Panels are grouped into batch sheets so a reviewer can approve several Y IDs
at once. Every sheet has a companion CSV containing the displayed candidates
and editable review fields. A local HTML gallery indexes pages, classes, review
status, check sheets, and their CSVs.

Raw ID-only and ID-plus-class full-page overlays remain secondary provenance
artifacts. They are never presented as alignment results.
