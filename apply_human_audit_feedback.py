"""Apply resolved human audit feedback without overwriting source CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.1.0-human-audit-feedback"
REVIEW_EXTRA_FIELDS = [
    "feedback_resolution_status",
    "corrected_xml_event_ids_json",
    "corrected_pitch",
    "corrected_staff",
    "corrected_measure_reference",
    "feedback_source",
    "feedback_evidence",
]
CORRECTION_FIELDS = [
    "sample_id", "bbox_id", "score_id", "page_id", "class",
    "original_start_meas", "original_note_ids", "original_pitches",
    "original_xml_event_ids_json", "corrected_start_meas",
    "corrected_end_meas", "corrected_note_ids",
    "corrected_xml_event_ids_json", "corrected_pitch", "corrected_staff",
    "corrected_measure_reference", "audit_decision", "reviewer",
    "reviewed_at", "feedback_evidence",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _active_feedback(payload: dict) -> tuple[list[dict], list[dict]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("feedback JSON must contain an entries list")
    active, withdrawn = [], []
    for entry in entries:
        if entry.get("apply_to_audit") is False:
            withdrawn.append(entry)
        elif str(entry.get("resolution_status", "")).startswith("resolved"):
            active.append(entry)
    return active, withdrawn


def apply_feedback_rows(
    audit_rows: list[dict], feedback_payload: dict
) -> tuple[list[dict], list[dict], list[str]]:
    """Return reviewed audit rows, normalized corrections, and validation errors."""

    active, _withdrawn = _active_feedback(feedback_payload)
    errors: list[str] = []
    sample_rows = {row.get("sample_id", ""): row for row in audit_rows}
    active_by_sample: dict[str, dict] = {}
    for entry in active:
        sample_id = entry.get("sample_id", "")
        if not sample_id:
            errors.append("active feedback entry is missing sample_id")
            continue
        if sample_id in active_by_sample:
            errors.append(f"duplicate active feedback for {sample_id}")
            continue
        active_by_sample[sample_id] = entry

    reviewed = [dict(row) for row in audit_rows]
    corrections = []
    reviewed_by_sample = {row.get("sample_id", ""): row for row in reviewed}
    reviewer = str(feedback_payload.get("reviewer") or "User")
    reviewed_at = str(feedback_payload.get("recorded_date") or "")
    source = str(feedback_payload.get("source") or "human_review")

    for index, (sample_id, entry) in enumerate(sorted(active_by_sample.items()), start=1):
        source_row = sample_rows.get(sample_id)
        row = reviewed_by_sample.get(sample_id)
        if source_row is None or row is None:
            errors.append(f"feedback sample_id not found in audit CSV: {sample_id}")
            continue
        if entry.get("bbox_id") != source_row.get("bbox_id"):
            errors.append(
                f"bbox_id mismatch for {sample_id}: "
                f"feedback={entry.get('bbox_id')} audit={source_row.get('bbox_id')}"
            )
            continue
        note_ids = entry.get("corrected_note_ids")
        event_ids = entry.get("corrected_xml_event_ids")
        if not isinstance(note_ids, list) or not note_ids:
            errors.append(f"resolved feedback has no corrected_note_ids: {sample_id}")
            continue
        if not isinstance(event_ids, list) or not event_ids:
            errors.append(f"resolved feedback has no corrected_xml_event_ids: {sample_id}")
            continue
        correction = entry.get("user_correction") or {}
        corrected_start = entry.get("corrected_start_meas")
        corrected_pitch = entry.get("canonical_pitch") or correction.get("pitch", "")
        measure_reference = {
            "measure": correction.get("measure"),
            "domain": entry.get("measure_numbering_domain", "bps_start_meas"),
        }
        row.update(
            {
                "proposal_status": "audit_rejected_original_corrected",
                "proposed_combined_status": "human_corrected",
                "audit_decision": "incorrect_original_match_corrected",
                "audited_by": reviewer,
                "audited_at": reviewed_at,
                "audit_comment": entry.get("evidence", ""),
                "current_review_status": "reviewed",
                "human_approved": "true",
                "decision": "corrected",
                "corrected_start_meas": str(corrected_start),
                "corrected_end_meas": str(corrected_start),
                "corrected_note_ids": _json(note_ids),
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "comment": entry.get("evidence", ""),
                "feedback_resolution_status": entry.get("resolution_status", ""),
                "corrected_xml_event_ids_json": _json(event_ids),
                "corrected_pitch": corrected_pitch,
                "corrected_staff": correction.get("staff", "") if correction.get("staff") is not None else "",
                "corrected_measure_reference": _json(measure_reference),
                "feedback_source": source,
                "feedback_evidence": entry.get("evidence", ""),
            }
        )
        corrections.append(
            {
                "sample_id": sample_id,
                "bbox_id": row.get("bbox_id", ""),
                "score_id": row.get("score_id", ""),
                "page_id": row.get("page_id", ""),
                "class": row.get("class", ""),
                "original_start_meas": source_row.get("start_meas", ""),
                "original_note_ids": source_row.get("note_ids", ""),
                "original_pitches": source_row.get("pitches", ""),
                "original_xml_event_ids_json": source_row.get("xml_event_ids_json", ""),
                "corrected_start_meas": str(corrected_start),
                "corrected_end_meas": str(corrected_start),
                "corrected_note_ids": _json(note_ids),
                "corrected_xml_event_ids_json": _json(event_ids),
                "corrected_pitch": corrected_pitch,
                "corrected_staff": row["corrected_staff"],
                "corrected_measure_reference": row["corrected_measure_reference"],
                "audit_decision": row["audit_decision"],
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "feedback_evidence": entry.get("evidence", ""),
            }
        )
        emit_progress("human-audit-feedback", index, len(active_by_sample), sample_id)
    return reviewed, corrections, errors


def apply_human_audit_feedback(
    audit_csv: Path,
    feedback_json: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    output_csv = output_dir / "auto_accept_audit_reviewed.csv"
    corrections_csv = output_dir / "applied_corrections.csv"
    report_path = output_dir / "validation_report.json"
    checkpoint_path = output_dir / "checkpoint.json"
    fingerprint = stable_digest(
        {
            "pipeline_version": PIPELINE_VERSION,
            "audit_csv": path_signature(audit_csv),
            "feedback_json": path_signature(feedback_json),
        }
    )
    if resume and all(path.is_file() for path in [output_csv, corrections_csv, report_path, checkpoint_path]):
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if checkpoint.get("fingerprint") == fingerprint and report.get("passed") is True:
                emit_progress("human-audit-feedback", 1, 1, "resumed completed output")
                return {**report, "resumed": True}
        except (OSError, json.JSONDecodeError):
            pass

    audit_fields, audit_rows = _read_csv(audit_csv)
    feedback_payload = json.loads(feedback_json.read_text(encoding="utf-8"))
    active, withdrawn = _active_feedback(feedback_payload)
    emit_progress("human-audit-feedback", 0, len(active), f"starting (resume={resume})")
    reviewed, corrections, errors = apply_feedback_rows(audit_rows, feedback_payload)
    if len(corrections) != len(active):
        errors.append(
            f"applied correction count {len(corrections)} differs from active feedback count {len(active)}"
        )
    reviewed_ids = [row.get("sample_id", "") for row in reviewed]
    if len(reviewed_ids) != len(set(reviewed_ids)):
        errors.append("duplicate sample_id in reviewed audit output")
    reviewed_fields = audit_fields + [field for field in REVIEW_EXTRA_FIELDS if field not in audit_fields]
    atomic_write_csv(output_csv, reviewed_fields, reviewed)
    atomic_write_csv(corrections_csv, CORRECTION_FIELDS, corrections)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "source_audit_rows": len(audit_rows),
        "reviewed_audit_rows": len(reviewed),
        "feedback_entries": len(feedback_payload.get("entries", [])),
        "active_feedback_entries": len(active),
        "withdrawn_feedback_entries": len(withdrawn),
        "applied_corrections": len(corrections),
        "unreviewed_audit_rows": len(reviewed) - len(corrections),
        "source_audit_modified": False,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "reviewed_audit_csv": str(output_csv),
            "applied_corrections_csv": str(corrections_csv),
        },
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        checkpoint_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "fingerprint": fingerprint,
            "passed": report["passed"],
            "applied_corrections": len(corrections),
        },
    )
    emit_progress("human-audit-feedback-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return {**report, "resumed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = apply_human_audit_feedback(
        args.audit_csv,
        args.feedback_json,
        args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
