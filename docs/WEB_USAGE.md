# Website usage and operations

## Local use

```bash
python -m pip install .
bpsd-aligner web
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Four-step workflow

1. **Upload score files**: choose all score images and all matching YOLO TXT
   files, then one repetition MusicXML, BPSD note CSV and `notes.json`.
2. **Align**: verify the page-pairing table and press
   **Align all uploaded pages**. The durable background worker writes progress
   and one checkpoint per completed page.
3. **Review**: inspect the overview or one class at a time. The main workspace
   supports previous/next navigation, typed endpoints, direct notehead clicks,
   cross-page spans and automatic save of explicit decisions.
4. **Download**: use the final CSV button at the top of the result. After
   applying review decisions, it becomes the reviewed final CSV.

Images and TXT files pair by exact filename stem. Multiple files can be
selected in one upload. If scan systems do not start at the same written
measures as MusicXML systems, edit the optional scan-system anchors in the
pairing table; otherwise leave them blank.

## Correct dataset versions

| Field | Version |
|---|---|
| Score images | Scanned pages, normally `finished/Xia/images` |
| YOLO TXT | Matching `finished/Xia/labels` |
| YOLO class map | Matching `finished/Xia/notes.json` |
| Repetition MusicXML | `score_xml_repetitions/*.xml` |
| Unfolded MusicXML | `score_xml_unfolded/*.xml`, optional validation |
| Clean PDF | `score_pdf_repetitions/*.pdf` |
| BPSD note annotations | `2_Annotations/ann_score_note/*.csv` |

Do not upload `.sib` or unfolded PDF files.

## Review rules

- Green/blue machine links are candidates; only confirmed evidence enters the
  final semantic fields.
- Unknown values remain blank. Never invent a time or note ID to finish a row.
- `Matched spot check` deterministically samples up to three rows per class.
- Saving a correction sets `human_corrected=1`; confirming an unchanged machine
  result does not.
- Review state is bound to owner, score, input fingerprint, pipeline version and
  code signature.

## Resume checkpoint ZIP

The optional ZIP resumes the same input batch; it is not a general data import.
It contains derived page outputs and review state, not original uploads. The ZIP
is HMAC-signed, member-hashed and size-limited. To move it between deployments,
both deployments must share `BPSD_ALIGNER_CHECKPOINT_SECRET`.

## Production configuration

Copy [`.env.example`](../.env.example). Important settings:

| Variable | Purpose |
|---|---|
| `BPSD_ALIGNER_DEPLOYMENT_MODE=production` | Fail closed without authentication |
| `BPSD_ALIGNER_USERS_FILE` | Per-user plaintext or `sha256:` token map |
| `BPSD_ALIGNER_ACCESS_TOKEN` | Simpler shared-token fallback |
| `BPSD_ALIGNER_CHECKPOINT_SECRET` | HMAC key; at least 32 bytes |
| `BPSD_ALIGNER_JOB_DIR` | Persistent private job volume |
| `BPSD_ALIGNER_JOB_RETENTION_HOURS` | Inactive-job retention; production default 168 |
| `BPSD_ALIGNER_MAX_STORAGE_BYTES` | Whole job-store quota |
| `BPSD_ALIGNER_MAX_PAGES` | Pages per browser job |
| `BPSD_ALIGNER_MAX_PDF_PAGES` | Clean-PDF page cap |
| `BPSD_ALIGNER_MAX_PDF_RENDER_PIXELS` | Rendered page pixel cap |

Run behind HTTPS and preferably organization SSO/OIDC. The built-in token gate
is a small-deployment fallback, not a replacement for an identity provider.

```bash
bpsd-aligner job-admin status
bpsd-aligner job-admin prune --retention-hours 168
bpsd-aligner dispatcher --job-dir /var/lib/bpsd-aligner
```

The default dispatcher is serial (`BPSD_ALIGNER_MAX_CONCURRENT_JOBS=1`) to keep
memory use predictable. Increase it only after measuring representative jobs.

## Default artifacts

The website stores only final/review essentials, validation JSON, overview
images and a signed recovery checkpoint. Full XML-node, combined and timeline
research tables remain available through `bpsd-aligner xml-export` and
`bpsd-aligner combine`; they are intentionally not generated for every web job.
