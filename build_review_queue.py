"""Build a resumable review queue from existing combined rows and review sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.2.0-review-queue"
QUEUE_FIELDS = [
    "queue_id", "priority_rank", "priority_tier", "batch_id", "batch_rank",
    "score_id", "page_id", "yolo_line", "bbox_id", "class_id", "class",
    "source_status", "confidence", "start_meas", "end_meas", "start_note",
    "end_note", "connected_note", "note_ids", "pitches",
    "xml_event_ids_json", "xml_event_count", "link_methods_json",
    "review_png", "review_csv", "asset_status", "current_review_status",
    "human_approved", "decision", "corrected_start_meas",
    "corrected_end_meas", "corrected_note_ids", "reviewer", "reviewed_at",
    "comment", "pipeline_version",
]
BATCH_FIELDS = [
    "batch_rank", "batch_id", "score_id", "page_id", "class", "part",
    "queue_rows", "ambiguous_rows", "candidate_rows", "min_confidence",
    "max_confidence", "review_png", "review_csv", "asset_status",
    "batch_review_status",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_review_assets(review_dir: Path) -> tuple[dict[str, dict], list[dict], str]:
    _fields, index_rows = _read_csv(review_dir / "review_sheet_index.csv")
    by_bbox: dict[str, dict] = {}
    signature_items = [path_signature(review_dir / "review_sheet_index.csv")]
    normalized_sheets = []
    for index_row in index_rows:
        csv_path = review_dir / index_row["csv"]
        png_path = review_dir / index_row["png"]
        csv_exists = csv_path.is_file()
        png_exists = png_path.is_file()
        if csv_exists:
            signature_items.append(path_signature(csv_path))
        signature_items.append(
            {
                "png_path": str(png_path.resolve()),
                "png_exists": png_exists,
                "png_signature": path_signature(png_path) if png_exists else None,
            }
        )
        batch_id = (
            f"{index_row['page_id']}:{index_row['class']}:"
            f"part{int(index_row['part']):02d}"
        )
        sheet = {
            **index_row,
            "batch_id": batch_id,
            "review_png": str(png_path.resolve()),
            "review_csv": str(csv_path.resolve()),
            "asset_status": "ready" if csv_exists and png_exists else "missing",
        }
        normalized_sheets.append(sheet)
        if not csv_exists:
            continue
        _candidate_fields, candidates = _read_csv(csv_path)
        for candidate in candidates:
            bbox_id = candidate.get("bbox_id", "")
            if not bbox_id:
                continue
            if bbox_id in by_bbox:
                by_bbox[bbox_id] = {**sheet, "asset_status": "duplicate"}
            else:
                by_bbox[bbox_id] = sheet
    return by_bbox, normalized_sheets, stable_digest(signature_items)


def _priority_key(row: dict) -> tuple:
    status_rank = 0 if row["combined_status"] == "ambiguous" else 1
    try:
        confidence = float(row["confidence"])
    except (TypeError, ValueError):
        confidence = -1.0
    try:
        yolo_line = int(row["yolo_line"])
    except (TypeError, ValueError):
        yolo_line = 0
    return (
        status_rank,
        confidence,
        row["class"],
        row["score_id"],
        row["page_id"],
        yolo_line,
    )


def build_score_queue(
    rows: list[dict],
    assets_by_bbox: dict[str, dict],
    link_methods_by_bbox: dict[str, list[str]],
) -> list[dict]:
    queue = []
    for row in rows:
        if row.get("row_origin") != "yolo":
            continue
        if row.get("combined_status") not in {"candidate", "ambiguous"}:
            continue
        asset = assets_by_bbox.get(row["bbox_id"])
        queue.append(
            {
                "priority_tier": "P1_ambiguous" if row["combined_status"] == "ambiguous" else "P2_candidate",
                "batch_id": asset["batch_id"] if asset else "",
                "score_id": row["score_id"],
                "page_id": row["page_id"],
                "yolo_line": row["yolo_line"],
                "bbox_id": row["bbox_id"],
                "class_id": row["class_id"],
                "class": row["class"],
                "source_status": row["combined_status"],
                "confidence": row["confidence"],
                "start_meas": row["start_meas"],
                "end_meas": row["end_meas"],
                "start_note": row["start_note"],
                "end_note": row["end_note"],
                "connected_note": row["connected_note"],
                "note_ids": row.get("note_ids", ""),
                "pitches": row.get("pitches", ""),
                "xml_event_ids_json": row["xml_event_ids_json"],
                "xml_event_count": row["xml_event_count"],
                "link_methods_json": _json(link_methods_by_bbox.get(row["bbox_id"], [])),
                "review_png": asset["review_png"] if asset else "",
                "review_csv": asset["review_csv"] if asset else "",
                "asset_status": asset["asset_status"] if asset else "missing",
                "current_review_status": row.get("review_status", ""),
                "human_approved": row.get("human_approved", "false"),
                "decision": "",
                "corrected_start_meas": "",
                "corrected_end_meas": "",
                "corrected_note_ids": "",
                "reviewer": "",
                "reviewed_at": "",
                "comment": "",
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    return sorted(queue, key=lambda row: _priority_key({
        **row,
        "combined_status": row["source_status"],
    }))


def build_batches(queue: list[dict], sheets_by_batch: dict[str, dict]) -> list[dict]:
    rows_by_batch: dict[str, list[dict]] = defaultdict(list)
    for row in queue:
        if row["batch_id"]:
            rows_by_batch[row["batch_id"]].append(row)
    ordered = sorted(
        rows_by_batch.items(),
        key=lambda item: min(int(row["priority_rank"]) for row in item[1]),
    )
    batches = []
    for rank, (batch_id, rows) in enumerate(ordered, start=1):
        sheet = sheets_by_batch[batch_id]
        confidences = [float(row["confidence"]) for row in rows if row["confidence"]]
        batch = {
            "batch_rank": rank,
            "batch_id": batch_id,
            "score_id": rows[0]["score_id"],
            "page_id": sheet["page_id"],
            "class": sheet["class"],
            "part": sheet["part"],
            "queue_rows": len(rows),
            "ambiguous_rows": sum(row["source_status"] == "ambiguous" for row in rows),
            "candidate_rows": sum(row["source_status"] == "candidate" for row in rows),
            "min_confidence": f"{min(confidences):.3f}" if confidences else "",
            "max_confidence": f"{max(confidences):.3f}" if confidences else "",
            "review_png": sheet["review_png"],
            "review_csv": sheet["review_csv"],
            "asset_status": sheet["asset_status"],
            "batch_review_status": "not_started",
        }
        batches.append(batch)
        for row in rows:
            row["batch_rank"] = rank
    return batches


def build_review_queue(
    combined_master_path: Path,
    alignment_links_path: Path,
    review_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    _combined_fields, combined = _read_csv(combined_master_path)
    _link_fields, links = _read_csv(alignment_links_path)
    assets_by_bbox, sheets, assets_signature = load_review_assets(review_dir)
    sheets_by_batch = {sheet["batch_id"]: sheet for sheet in sheets}
    link_methods_by_bbox: dict[str, list[str]] = defaultdict(list)
    for link in links:
        method = link["link_method"]
        if method not in link_methods_by_bbox[link["bbox_id"]]:
            link_methods_by_bbox[link["bbox_id"]].append(method)
    by_score: dict[str, list[dict]] = defaultdict(list)
    for row in combined:
        if row.get("row_origin") == "yolo":
            by_score[row["score_id"]].append(row)
    score_ids = sorted(by_score)
    all_queue = []
    score_reports = []
    emit_progress("review-queue", 0, len(score_ids), f"starting (resume={resume})")
    for index, score_id in enumerate(score_ids, start=1):
        queue_path = output_dir / "per_scores" / f"{score_id}_review_queue.csv"
        checkpoint_path = output_dir / "checkpoints" / f"{score_id}.json"
        fingerprint = stable_digest(
            {
                "version": PIPELINE_VERSION,
                "combined_master": path_signature(combined_master_path),
                "alignment_links": path_signature(alignment_links_path),
                "review_assets": assets_signature,
                "score_id": score_id,
            }
        )
        reused = False
        if resume and queue_path.is_file() and checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                _saved_fields, queue = _read_csv(queue_path)
                reused = (
                    checkpoint.get("fingerprint") == fingerprint
                    and checkpoint.get("queue_rows") == len(queue)
                )
            except (OSError, csv.Error, json.JSONDecodeError):
                reused = False
        if not reused:
            queue = build_score_queue(
                by_score[score_id], assets_by_bbox, link_methods_by_bbox
            )
            atomic_write_csv(queue_path, QUEUE_FIELDS, queue)
            atomic_write_json(
                checkpoint_path,
                {
                    "score_id": score_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "fingerprint": fingerprint,
                    "queue_rows": len(queue),
                },
            )
        all_queue.extend(queue)
        score_reports.append(
            {"score_id": score_id, "queue_rows": len(queue), "resumed": reused}
        )
        emit_progress(
            "review-queue", index, len(score_ids),
            f"{score_id} {'resumed' if reused else 'built'} rows={len(queue)}",
        )

    all_queue.sort(key=lambda row: _priority_key({
        **row,
        "combined_status": row["source_status"],
    }))
    for rank, row in enumerate(all_queue, start=1):
        row["priority_rank"] = rank
        row["queue_id"] = f"Q{rank:05d}"
    batches = build_batches(all_queue, sheets_by_batch)
    final_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in all_queue:
        final_by_score[row["score_id"]].append(row)
    for score_id, rows in final_by_score.items():
        atomic_write_csv(
            output_dir / "per_scores" / f"{score_id}_review_queue.csv",
            QUEUE_FIELDS,
            rows,
        )
    atomic_write_csv(output_dir / "review_queue.csv", QUEUE_FIELDS, all_queue)
    atomic_write_csv(output_dir / "review_batches.csv", BATCH_FIELDS, batches)

    errors = []
    source_review_ids = {
        row["bbox_id"] for row in combined
        if row.get("row_origin") == "yolo"
        and row.get("combined_status") in {"candidate", "ambiguous"}
    }
    queue_ids = [row["queue_id"] for row in all_queue]
    queue_bbox_ids = [row["bbox_id"] for row in all_queue]
    if len(queue_ids) != len(set(queue_ids)):
        errors.append("duplicate queue_id")
    if len(queue_bbox_ids) != len(set(queue_bbox_ids)):
        errors.append("duplicate bbox_id in review queue")
    if source_review_ids != set(queue_bbox_ids):
        errors.append("review queue does not exactly cover candidate/ambiguous bbox IDs")
    if any(row["asset_status"] != "ready" for row in all_queue):
        errors.append("review queue contains missing or duplicate assets")
    if any(not Path(row["review_png"]).is_file() or not Path(row["review_csv"]).is_file() for row in all_queue):
        errors.append("review queue references missing asset paths")
    status_ranks = [0 if row["source_status"] == "ambiguous" else 1 for row in all_queue]
    if status_ranks != sorted(status_ranks):
        errors.append("review queue priority status order is invalid")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "queue_rows": len(all_queue),
        "unique_bbox_ids": len(set(queue_bbox_ids)),
        "batch_rows": len(batches),
        "status_counts": dict(Counter(row["source_status"] for row in all_queue)),
        "class_counts": dict(Counter(row["class"] for row in all_queue)),
        "asset_status_counts": dict(Counter(row["asset_status"] for row in all_queue)),
        "score_reports": score_reports,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "review_queue": str(output_dir / "review_queue.csv"),
            "review_batches": str(output_dir / "review_batches.csv"),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("review-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-master", type=Path, required=True)
    parser.add_argument("--alignment-links", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build_review_queue(
        args.combined_master,
        args.alignment_links,
        args.review_dir,
        args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
