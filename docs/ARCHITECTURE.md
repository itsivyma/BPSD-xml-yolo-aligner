# Architecture

## System boundary

The project aligns detected score glyphs with symbolic score evidence. It does
not train YOLO and does not infer missing facts when neither MusicXML nor BPSD
annotations provide evidence.

```text
images + YOLO TXT + notes.json
                │
                ├── page/system geometry ──────────────┐
                │                                      │
repetition MusicXML ── written measures/XML events ────┤
unfolded MusicXML ──── repeat-expanded order ──────────┤
BPSD note CSV ──────── official time and note IDs ─────┤
                                                       ▼
                                                alignment engine
                                                       │
                         ┌─────────────────────────────┼──────────────┐
                         ▼                             ▼              ▼
                   strict BPS-OMR CSV          diagnostic CSVs   review images
                                                       │              │
                                                       └── human review
                                                              │
                                                              ▼
                                                     corrected outputs
```

## Source responsibilities

- Scanned images and YOLO describe visible glyph class, bounding box, page and
  approximate geometry.
- Repetition MusicXML describes the printed score: written measure number,
  page/system placement, staff, voice, pitch and notation relationships.
- Unfolded MusicXML describes performance order after repeats. It supplements
  rather than replaces repetition MusicXML.
- BPSD note annotations are authoritative for official note IDs and the
  `start_meas`/`end_meas` timeline.
- Human Review is authoritative when it explicitly confirms or corrects a row.

## Main modules

| Module | Responsibility |
|---|---|
| `bps_xml_alignment.py` | Current single-page matching engine and compatibility facade |
| `bpsd_aligner/candidate_scoring.py` | Generic one-to-one pairing, mutual-best checks and candidate margins |
| `bpsd_aligner/geometry.py` | Staff/system detection, barlines and box-to-system assignment |
| `bpsd_aligner/overlay.py` | Pure in-memory and file review-overlay rendering |
| `repeat_mapping.py` | Written-to-performance repeat occurrence mapping |
| `xml_export.py` | Lossless MusicXML nodes/events export |
| `combine_yolo_xml.py` | Lossless XML + YOLO research tables |
| `bpsd_aligner/web_pipeline.py` | Shared-score preprocessing and multi-page orchestration |
| `bpsd_aligner/web.py` | Streamlit presentation and Review workspace |
| `bpsd_aligner/job_store.py` | Atomic jobs, progress, ownership and checkpoints |
| `bpsd_aligner/review_corrections.py` | Review validation and corrected output rebuilding |
| `bpsd_aligner/class_registry.py` | Stable class family, timeline and threshold defaults |
| `bpsd_aligner/bps_omr_schema.py` | Compatibility entry point for final CSV semantics |
| `bpsd_aligner/thresholds.py` | Environment overrides over registry defaults |
| `bpsd_aligner/regression_smoke.py` | Fixed real-page semantic regression |

## Important invariants

- Every YOLO input line keeps a stable page/line identity.
- Source XML nodes and events are never discarded from diagnostic exports.
- Strict final CSV follows BPS-OMR annotations and leaves unsupported values
  blank; diagnostic evidence belongs in separate outputs.
- Repeated written measures keep occurrence identity. A printed measure number
  alone is not a unique performance time.
- A machine candidate may be visible in Review without being accepted in the
  strict final CSV.
- Review data is bound to input fingerprint, pipeline version, score and owner.
- Completed page checkpoints are atomic and reusable only when inputs and code
  signatures still match.

## Job lifecycle

```text
uploaded → queued → shared preprocessing → page 1..N checkpoints
         → complete exports → outputs_ready → Review/corrections
```

The website worker writes structured status events. A browser restart does not
restart completed pages, and saved Review decisions are restored from the
persistent job directory after identity validation.

## Safe future decomposition

`bps_xml_alignment.py` and `web.py` remain the largest modules. Split them by
moving behavior without changing it:

Geometry, overlay rendering and generic candidate scoring have already been
extracted behind compatibility imports. Continue with:

1. point-symbol matchers;
2. span/endpoint matchers;
3. upload/job controllers;
4. Review UI components.

After each move, run unit tests and the fixed regression suite. Do not combine a
module move with a new matching rule in the same commit.
