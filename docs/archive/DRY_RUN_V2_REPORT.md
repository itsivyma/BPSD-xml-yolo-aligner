# Dataset Alignment Dry Run v2

## Valid run

- Pipeline: `0.2.0-dry-run`
- Canonical rows: 9,828
- Unique bbox IDs: 9,828
- Pages: 49
- Scores: 6
- Failed pages: 0
- Validation errors: 0
- Human-approved rows imported and protected: 148
- Outside-BPSD rows retained: 1,947

Alignment states:

| Status | Rows |
| --- | ---: |
| `matched` | 799 |
| `candidate` | 2,636 |
| `ambiguous` | 628 |
| `unresolved` | 3,818 |
| `xml_missing` | 1,947 |

The 799 matched rows include 148 human-approved rows. All remaining machine
matches/candidates still have `human_approved=false` and require review.

## Review artifacts

- Reviewable, not-yet-approved candidate panels: 3,915
- Batch sheets: 637
- Every batch sheet has a companion review CSV.
- The HTML gallery links all sheets and CSVs.

Blue boxes are original YOLO annotations. Green circles/lines are proposed
semantic targets. Repeated written notes display all unfolded BPSD note IDs and
their occurrence count.

## Safety findings

The first dry run exposed a legacy hard-coded class-ID mapping. In the current
Xia `notes.json`, IDs 25-29 are dynamics rather than fingerings. The core now
selects semantic classes by the authoritative class name loaded from
`notes.json`; class IDs are retained only as source data. Human reviews are
also checked against the current class name.

`output/dry_run_v1` is invalid and retained only for debugging/audit. It must
not be used for review or export. `output/dry_run_v2` is the current valid run.

## Remaining matcher work

The unresolved queue primarily contains classes whose class-aware matchers have
not yet been integrated into the dataset runner, including staccato, fermata,
slur, tie, tuplets, hairpins, text/tempo, ornaments, and other MusicXML marks.
No unresolved row has been silently guessed.
