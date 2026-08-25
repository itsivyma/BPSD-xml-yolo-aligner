# Final BPS-OMR CSV schema

The executable source of truth is
[`bpsd_aligner/schema.py`](../bpsd_aligner/schema.py). This document explains
the fields; it does not redefine them.

| Field | Meaning |
|---|---|
| `class_id` | YOLO category ID |
| `x`, `y`, `w`, `h` | Normalized YOLO bounding box |
| `class` | Class name from the uploaded `notes.json` |
| `musical_time` | BPS-OMR timing interpretation for the class |
| `start_meas`, `end_meas` | BPSD performance-time coordinates |
| `start_note`, `end_note` | Endpoint BPSD note IDs |
| `connected_note` | JSON/list representation of all connected note IDs |
| `stem_dir` | Stem direction only for stem classes |
| `human_corrected` | `1` only when a human changed machine values |
| `is_repeated_measure` | `1` when the written measure has multiple occurrences |

Rules:

- There is exactly one final row for every input YOLO box.
- Unknown, unavailable and not-applicable values are empty strings, never `NA`
  and never fabricated estimates.
- Unconfirmed machine candidates stay visible in review data but their uncertain
  final semantic fields are blank.
- Rows are ordered by musical start/end time. A hidden machine candidate may be
  used only as the sort key for a blank final row; it is not written into that
  row.
- For chord endpoints, `connected_note` includes every note in the endpoint
  chord; `start_note` and `end_note` remain stable representative IDs.
- Research exports are separate CLI products and are not part of the final
  BPS-OMR contract.
