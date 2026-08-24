# Project handover

## Current state

The system supports multi-page image/TXT upload, repetition and unfolded
MusicXML, BPSD note annotations, YOLO class maps, optional clean repetition
PDF, background jobs, progress, checkpoint/resume, strict BPS-OMR CSV, complete
diagnostic exports, overlays and an image-first human Review workspace.

Review can correct classes, point symbols and span endpoints by typed values or
clicking score noteheads. Cross-page spans display both endpoint pages when
available. Every explicit save is atomic and restored only for the same input
fingerprint, pipeline version, score and authenticated owner.

The repository intentionally does not contain Beethoven source data or local
outputs. `output/` and `validation_smoke/` may be large on a developer machine,
but they are ignored and are not downloaded by the next person cloning GitHub.

## Accuracy boundary

- YOLO detection is an external trained model.
- Alignment is currently deterministic MusicXML/BPSD logic plus geometric
  heuristics, not a learned alignment model.
- Confidence is not a calibrated probability.
- XML-unconfirmed, ambiguous, cross-system and some text/numeral relationships
  remain Review-only.
- `fingeringSubstitution` and wrong-class/bad-bbox cases need human labels or a
  future learned relation model.

Do not describe all Review rows as errors. Many are deliberately conservative
candidates that the system refuses to invent values for.

## Validation assets

- Unit tests cover parsers, repeat mapping, matching, exports, web helpers,
  Review corrections, persistence and security controls.
- `regression/representative_pages.json` selects ten varied real pages.
- `regression/representative_baseline.json` stores reviewed aggregate and
  row-level semantic fingerprints.

Never update the baseline until a human has verified the changed assignments.

## Recommended next work

1. Continue decomposing `bps_xml_alignment.py`. Geometry and rendering are now
   isolated; candidate scoring, point and span matchers remain.
2. Decompose `web.py` into upload, job status, results and Review components.
3. Replace cross-module row dictionaries with typed domain records at stable
   boundaries.
4. Collect 200–500 representative human decisions and calibrate thresholds by
   class family.
5. If enough labels accumulate, train an optional relation/ranking model that
   predicts the XML note/span candidate; retain rule evidence and human Review.
6. Add CI for unit tests, package installation and a public-data smoke fixture.

## Production handoff

Before exposing a shared website:

- configure per-user tokens;
- mount `BPSD_ALIGNER_JOB_DIR` on persistent storage;
- set upload/page/storage limits;
- schedule `bpsd-aligner job-admin prune`;
- protect logs and outputs as annotation data;
- monitor failed jobs and disk usage;
- document backup and retention policy.

See `.env.example` and `docs/WEB_USAGE.md` for the exact environment variables.

## Files a new maintainer should read first

1. `START_HERE.md`
2. `docs/ARCHITECTURE.md`
3. `bpsd_aligner/class_registry.py`
4. `bpsd_aligner/web_pipeline.py`
5. focused sections of `bps_xml_alignment.py`
6. the test matching the feature being changed

Root-level research scripts should be consulted only when reproducing an older
class-specific experiment.
