# Archived human review evidence

These files preserve human decisions produced by the pre-website fingering
audit workflow on 2026-08-09. They are retained because human review cannot be
reconstructed from source MusicXML, YOLO labels, or generated outputs.

## Files

- `human_audit_feedback_v1.json`: six audit-trail entries, including one
  correction explicitly withdrawn as a typo.
- `human_audit_feedback_round2.json`: twenty individually resolved review
  entries.
- `human_audit_feedback_round3.json`: one batch-level sampled acceptance. It
  explicitly does not mean that every row was individually approved.

## Usage rules

- Treat these JSON files as immutable source evidence.
- Do not load them automatically as current alignment truth.
- Honor `apply_to_audit: false` and other withdrawal/status fields.
- Do not convert the round-3 batch acceptance into per-row human approvals.
- Before using an entry for regression scoring or confidence calibration,
  normalize it to the current ground-truth schema and verify that its
  `bbox_id`, note IDs, score version, and measure-numbering domain still match
  the selected dataset profile.
- New website reviews belong in the current review export/ground-truth workflow,
  not in these archived files.

The one-off round-2/round-3 migration scripts and generated audit outputs are
deliberately not archived here. The current `bpsd-aligner apply-review` and
regression tools remain the supported workflow.

Validate the archive without applying any decision:

```bash
bpsd-aligner validate-evidence regression/review_evidence
```

The validator also excludes the earlier decision sharing an identity with a
later withdrawal. The current archive contains 26 historical row entries, of
which 24 remain active; its one sampled batch acceptance promotes zero rows to
individual human approval.
