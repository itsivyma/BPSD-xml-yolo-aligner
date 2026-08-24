# Security policy

Please report vulnerabilities privately to the repository owner rather than
opening a public issue with exploit details.

The current release protects MusicXML parsing from external entities, limits
image/PDF/ZIP expansion, validates normalized YOLO geometry, stores jobs with
private permissions, binds review data to authenticated owners, and signs
portable checkpoints with per-member hashes and HMAC.

For shared deployments:

- use HTTPS and an external identity provider or reverse proxy;
- set `BPSD_ALIGNER_DEPLOYMENT_MODE=production`;
- persist and back up `BPSD_ALIGNER_JOB_DIR` and signing secrets separately;
- use a high-entropy `BPSD_ALIGNER_CHECKPOINT_SECRET` shared only by trusted
  instances;
- monitor storage, failed jobs and retention with `bpsd-aligner job-admin`;
- never commit user-token files, score data, annotation data or job outputs.

Checkpoint ZIPs from releases before the signed archive schema are intentionally
rejected. Re-run alignment with the current release to create a trusted archive.
