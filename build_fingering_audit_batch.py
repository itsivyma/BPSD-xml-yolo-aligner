"""Build a deterministic, resumable fingering audit batch from focused output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.1.0-fingering-audit-batch"
FIELDS = [
    "sample_id", "selection_reason", "bbox_id", "score_id", "page_id",
    "yolo_line", "class", "confidence", "alignment_status", "start_meas",
    "note_ids", "pitches", "staff", "xml_measure", "target_x_px",
    "target_y_px", "image_path", "review_decision", "corrected_measure",
    "corrected_staff", "corrected_pitch", "corrected_note_ids", "comment",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _stratified_take(
    rows: list[dict], count: int, *, confidence_descending: bool
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["page_id"], row["class"])].append(row)
    for members in groups.values():
        members.sort(
            key=(
                (lambda row: (-float(row.get("confidence") or 0), row["bbox_id"]))
                if confidence_descending
                else (lambda row: row["bbox_id"])
            )
        )
    selected = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def select_audit_rows(
    master_rows: list[dict], excluded_bbox_ids: set[str],
    *, unresolved_count: int = 10, risk_count: int = 10,
) -> list[dict]:
    fingering = [
        row for row in master_rows
        if row.get("class", "").startswith("fingering")
        and row.get("bbox_id") not in excluded_bbox_ids
        and row.get("movement_scope_status") == "in_bpsd_scope"
        and not row.get("error_code")
    ]
    unresolved = _stratified_take(
        [row for row in fingering if row.get("alignment_status") == "unresolved"],
        unresolved_count,
        confidence_descending=False,
    )
    used = {row["bbox_id"] for row in unresolved}
    risk = _stratified_take(
        [
            row for row in fingering
            if row.get("alignment_status") in {"candidate", "ambiguous"}
            and row["bbox_id"] not in used
        ],
        risk_count,
        confidence_descending=True,
    )
    output = []
    for reason, rows in [("unresolved", unresolved), ("high_risk_link", risk)]:
        for row in rows:
            output.append(
                {
                    **{field: row.get(field, "") for field in FIELDS},
                    "sample_id": "",
                    "selection_reason": reason,
                }
            )
    for index, row in enumerate(output, start=1):
        row["sample_id"] = f"R{index:03d}"
    return output


def build_batch(
    master_csv: Path, feedback_json: Path, output_dir: Path,
    *, unresolved_count: int = 10, risk_count: int = 10,
    resume: bool = False,
) -> dict:
    output_csv = output_dir / "fingering_audit_batch.csv"
    checkpoint_path = output_dir / "checkpoint.json"
    report_path = output_dir / "validation_report.json"
    fingerprint = stable_digest(
        {
            "version": PIPELINE_VERSION,
            "master": path_signature(master_csv),
            "feedback": path_signature(feedback_json),
            "unresolved_count": unresolved_count,
            "risk_count": risk_count,
        }
    )
    if resume and output_csv.is_file() and checkpoint_path.is_file() and report_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") == fingerprint and report.get("passed"):
            emit_progress("fingering-audit-sample", 1, 1, "resumed completed output")
            return {**report, "resumed": True}

    feedback = json.loads(feedback_json.read_text(encoding="utf-8"))
    excluded = {
        entry["bbox_id"] for entry in feedback.get("entries", [])
        if entry.get("apply_to_audit") is not False
        and str(entry.get("resolution_status", "")).startswith("resolved")
    }
    emit_progress("fingering-audit-sample", 0, 2, "selecting unresolved")
    selected = select_audit_rows(
        _read_csv(master_csv), excluded,
        unresolved_count=unresolved_count, risk_count=risk_count,
    )
    emit_progress("fingering-audit-sample", 1, 2, "selecting high-risk links")
    atomic_write_csv(output_csv, FIELDS, selected)
    reason_counts = {
        reason: sum(row["selection_reason"] == reason for row in selected)
        for reason in ["unresolved", "high_risk_link"]
    }
    errors = []
    expected = unresolved_count + risk_count
    if len(selected) != expected:
        errors.append(f"expected {expected} samples, selected {len(selected)}")
    if len({row["bbox_id"] for row in selected}) != len(selected):
        errors.append("duplicate bbox_id in audit batch")
    if {row["bbox_id"] for row in selected} & excluded:
        errors.append("previously reviewed bbox included in audit batch")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "sample_rows": len(selected),
        "reason_counts": reason_counts,
        "excluded_reviewed_bbox_ids": len(excluded),
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {"audit_csv": str(output_csv)},
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        checkpoint_path,
        {"pipeline_version": PIPELINE_VERSION, "fingerprint": fingerprint},
    )
    emit_progress(
        "fingering-audit-sample", 2, 2,
        f"passed={report['passed']} samples={len(selected)}",
    )
    return {**report, "resumed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unresolved-count", type=int, default=10)
    parser.add_argument("--risk-count", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build_batch(
        args.master_csv, args.feedback_json, args.output_dir,
        unresolved_count=args.unresolved_count,
        risk_count=args.risk_count,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
