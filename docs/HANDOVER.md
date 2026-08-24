# Project handover

## Current state

The system supports multi-page image/TXT upload, repetition and unfolded
MusicXML, BPSD note annotations, YOLO class maps, optional clean repetition
PDF, background jobs, progress, signed checkpoint/resume, strict BPS-OMR CSV,
on-demand class overlays and an image-first human Review workspace.

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
- `regression/representative_pages.json` selects ten varied pages from the
  legacy/test `Xia/` 113-class profile.
- `regression/finished_xia_pages.json` selects the same coverage from the
  production `finished/Xia/` 162-class profile.
- `regression/representative_baseline.json` stores reviewed aggregate and
  row-level semantic fingerprints.
- `regression/finished_xia_baseline.json` stores the human-accepted production
  semantic fingerprints, including the reviewed tuplet endpoints and complete
  slur endpoint chords.
- `regression/review_evidence/` preserves the original pre-website human
  fingering decisions. Read its README before normalization; batch-level spot
  checks are not per-row approvals. Run `bpsd-aligner validate-evidence
  regression/review_evidence` before using the archive.

Never update the baseline until a human has verified the changed assignments.

## Recommended next work

1. Continue decomposing `bps_xml_alignment.py`. Geometry, rendering and generic
   candidate scoring are isolated; point and span matchers remain behind the
   compatibility facade.
2. Continue decomposing `web.py` into upload, job status and result renderers.
   Framework-independent Review selection/input logic is already in
   `review_workspace.py`, and the UI no longer imports the matcher directly.
3. Replace cross-module row dictionaries with typed domain records at stable
   boundaries.
4. Collect 200–500 representative human decisions and calibrate thresholds by
   class family.
5. If enough labels accumulate, train an optional relation/ranking model that
   predicts the XML note/span candidate; retain rule evidence and human Review.
6. Replace the local token gate with organization SSO/OIDC when the deployment
   becomes a public multi-user service.

On Python 3.14, use the documented regular `pip install ".[dev]"`. Some
setuptools releases generate a hidden editable `.pth` that Python 3.14 skips;
this does not affect the regular wheel installation used by end users.

## Production handoff

The repository now enforces production authentication, private job permissions,
signed checkpoints, upload/PDF/ZIP limits, storage quota and default retention.
Before exposing a shared website, additionally:

- place the service behind HTTPS and preferably SSO/OIDC;
- mount `BPSD_ALIGNER_JOB_DIR` on persistent storage;
- set upload/page/storage limits;
- verify the configured automatic retention and backup policy;
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
