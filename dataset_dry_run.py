"""Run safe dataset-wide candidates and export a traceable canonical CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from bps_xml_alignment import (
    assign_system,
    detect_systems,
    load_categories,
    load_bps_notes,
    load_yolo,
    run_alignment,
)
from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.3.0"
OFFICIAL_FIELDS = [
    "class_id", "x", "y", "w", "h", "class", "musical_time",
    "start_meas", "end_meas", "start_note", "end_note",
    "connected_note", "stem_dir",
]
EXTENDED_FIELDS = [
    "dataset_id", "score_id", "page_id", "scan_page", "yolo_line", "bbox_id",
    "image_path", "yolo_path", "xml_path", "unfolded_xml_path", "sibelius_path",
    "bps_notes_path", "image_sha256", "yolo_sha256", "scan_system_index",
    "xml_page", "xml_systems_json", "written_measure_start", "written_measure_end",
    "xml_measure", "xml_symbol", "staff", "target_type", "note_ids", "pitches",
    "repeat_occurrences_json", "repeat_occurrence_count", "repeat_group_id",
    "movement_scope_status", "page_mapping_status", "mapping_source",
    "match_source", "confidence", "alignment_status", "review_status",
    "human_approved", "reviewer", "reviewed_at", "review_source",
    "original_candidate_json", "corrected_value_json", "comment",
    "target_x_px", "target_y_px", "end_target_x_px", "end_target_y_px",
    "pipeline_version", "error_code", "error_message",
]
FIELDS = OFFICIAL_FIELDS + EXTENDED_FIELDS


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def select_manifest_pages(
    pages: list[dict], page_ids: set[str] | None
) -> list[dict]:
    """Return requested manifest pages in manifest order, rejecting typos."""

    if not page_ids:
        return pages
    available = {row["page_id"] for row in pages}
    missing = sorted(page_ids - available)
    if missing:
        raise ValueError(
            "Requested page_id values are not in the manifest: "
            + ", ".join(missing)
        )
    return [row for row in pages if row["page_id"] in page_ids]


def load_human_reviews(review_dir: Path) -> tuple[dict[tuple[str, int], dict], list[str]]:
    reviews: dict[tuple[str, int], dict] = {}
    errors = []
    for path in sorted(review_dir.glob("*_human_review*.csv")):
        for row in _read_csv(path):
            key = (row["page_id"], int(row["yolo_line"]))
            record = {"row": row, "source": str(path)}
            if key in reviews:
                errors.append(f"duplicate human review key {key}: {path}")
                continue
            reviews[key] = record
    return reviews, errors


def _split_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.replace(",", "+").split("+") if item.strip()]


def _human_values(review: dict, bps_by_id: dict[int, dict]) -> dict:
    row = review["row"]
    note_ids = _split_ids(row.get("note_ids", ""))
    if row.get("note_id"):
        note_ids = [int(row["note_id"])]
    start_note = row.get("start_note_id", "")
    end_note = row.get("end_note_id", "")
    if start_note:
        note_ids.append(int(start_note))
    if end_note:
        note_ids.append(int(end_note))
    note_ids = list(dict.fromkeys(note_ids))

    start_meas = (
        row.get("corrected_start_meas")
        or row.get("candidate_start_meas")
        or row.get("bps_time")
        or row.get("start_bps_time")
        or ""
    )
    end_meas = (
        row.get("corrected_end_meas")
        or row.get("candidate_end_meas")
        or row.get("end_note_bps_time")
        or start_meas
    )
    if not start_meas and start_note and int(start_note) in bps_by_id:
        start_meas = f"{bps_by_id[int(start_note)]['bps_time']:.3f}"
    if not end_meas and end_note and int(end_note) in bps_by_id:
        end_meas = f"{bps_by_id[int(end_note)]['bps_time']:.3f}"

    pitches = row.get("pitches") or row.get("pitch") or ""
    pitch_list = pitches.split("+") if pitches else []
    target_type = row.get("target_type", "")
    if not target_type:
        target_type = "span" if start_note and end_note else ("note" if note_ids else "measure_position")
    return {
        "start_meas": start_meas,
        "end_meas": end_meas,
        "start_note": int(start_note) if start_note else (note_ids[0] if len(note_ids) == 1 else ("NA" if not note_ids else note_ids[0])),
        "end_note": int(end_note) if end_note else (note_ids[-1] if note_ids else "NA"),
        "connected_note": json.dumps(note_ids) if note_ids else "NA",
        "note_ids": json.dumps(note_ids),
        "pitches": json.dumps(pitch_list, ensure_ascii=False),
        "xml_measure": row.get("xml_measure", ""),
        "staff": row.get("staff", ""),
        "target_type": target_type,
    }


def _empty_candidate(box: dict, system: int) -> dict:
    return {
        "class_id": str(box["class_id"]),
        "x": f"{box['x']:.6f}", "y": f"{box['y']:.6f}",
        "w": f"{box['w']:.6f}", "h": f"{box['h']:.6f}",
        "class": box["class"], "musical_time": "", "start_meas": "",
        "end_meas": "", "start_note": "", "end_note": "",
        "connected_note": "", "stem_dir": "NA", "txt_line": str(box["txt_line"]),
        "system": str(system), "xml_measure": "", "xml_symbol": "", "xml_staff": "",
        "target_type": "", "note_ids": "", "pitches": "",
        "repeat_occurrences_json": "", "repeat_occurrence_count": "",
        "repeat_group_id": "", "match_source": "", "confidence": "",
        "status": "unresolved", "target_x_px": "", "target_y_px": "",
    }


def _page_fingerprint(
    page: dict,
    scope_rows: list[dict],
    reviews: dict[tuple[str, int], dict],
    notes_json_path: Path,
    repeat_mapping_path: Path,
) -> str:
    page_id = page["page_id"]
    page_reviews = [
        {
            "yolo_line": yolo_line,
            "source": review["source"],
            "row": review["row"],
        }
        for (review_page, yolo_line), review in sorted(reviews.items())
        if review_page == page_id
    ]
    input_paths = [
        Path(page["image_path"]),
        Path(page["yolo_path"]),
        Path(page["xml_path"]),
        Path(page["bps_notes_path"]),
        notes_json_path,
        repeat_mapping_path,
    ]
    return stable_digest(
        {
            "pipeline_version": PIPELINE_VERSION,
            "page": page,
            "scope": sorted(
                (row for row in scope_rows if row["page_id"] == page_id),
                key=lambda row: int(row["scan_system_index"]),
            ),
            "reviews": page_reviews,
            "inputs": [path_signature(path) for path in input_paths],
        }
    )


def _load_resumable_page(
    *,
    checkpoint_path: Path,
    page_csv_path: Path,
    page_id: str,
    expected_rows: int,
    fingerprint: str,
) -> tuple[list[dict], dict, list[str]] | None:
    if not checkpoint_path.is_file() or not page_csv_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = _read_csv(page_csv_path)
    except (OSError, json.JSONDecodeError, csv.Error):
        return None
    if checkpoint.get("fingerprint") != fingerprint:
        return None
    if checkpoint.get("status") not in {"completed", "outside_only"}:
        return None
    if len(rows) != expected_rows or checkpoint.get("row_count") != len(rows):
        return None
    bbox_ids = [row.get("bbox_id", "") for row in rows]
    if len(set(bbox_ids)) != len(bbox_ids):
        return None
    if any(row.get("page_id") != page_id for row in rows):
        return None
    page_report = {
        "page_id": page_id,
        "status": checkpoint["status"],
        "error": "",
        "resumed": True,
    }
    validation_errors = [str(error) for error in checkpoint.get("validation_errors", [])]
    return rows, page_report, validation_errors


def bootstrap_existing_checkpoints(
    *,
    manifest_path: Path,
    scope_path: Path,
    notes_json_path: Path,
    repeat_mapping_dir: Path,
    review_dir: Path,
    output_dir: Path,
) -> dict:
    """Adopt a previously validated run without executing alignment work."""

    validation_path = output_dir / "validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed") or validation.get("validation_errors"):
        raise ValueError("Existing validation report is not a passing run")
    if validation.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError(
            "Existing pipeline version does not match current code: "
            f"{validation.get('pipeline_version')} != {PIPELINE_VERSION}"
        )

    pages = _read_csv(manifest_path)
    scope_rows = _read_csv(scope_path)
    reviews, review_errors = load_human_reviews(review_dir)
    if review_errors:
        raise ValueError(
            "Current review inputs contain validation errors: "
            + "; ".join(review_errors)
        )
    reports = {row["page_id"]: row for row in validation.get("page_runs", [])}
    master_rows = _read_csv(output_dir / "Xia_BPSD_alignment_master.csv")
    master_ids = {row.get("bbox_id", "") for row in master_rows}
    if len(master_rows) != validation.get("canonical_rows"):
        raise ValueError("Existing master row count differs from validation report")
    if len(master_ids) != len(master_rows):
        raise ValueError("Existing master contains duplicate bbox_id values")

    created = 0
    emit_progress(
        "checkpoint-bootstrap",
        0,
        len(pages),
        "validating existing page CSV files only",
    )
    for index, page in enumerate(pages, start=1):
        page_id = page["page_id"]
        page_csv_path = output_dir / "pages" / f"{page_id}.csv"
        rows = _read_csv(page_csv_path)
        expected_rows = int(page["yolo_rows"])
        bbox_ids = [row.get("bbox_id", "") for row in rows]
        if len(rows) != expected_rows:
            raise ValueError(
                f"{page_id}: expected {expected_rows} rows, got {len(rows)}"
            )
        if len(set(bbox_ids)) != len(bbox_ids) or not set(bbox_ids) <= master_ids:
            raise ValueError(f"{page_id}: invalid or non-master bbox_id values")
        if any(row.get("page_id") != page_id for row in rows):
            raise ValueError(f"{page_id}: page_id mismatch in page CSV")
        if any(row.get("pipeline_version") != PIPELINE_VERSION for row in rows):
            raise ValueError(f"{page_id}: pipeline_version mismatch in page CSV")
        if any(row.get("image_sha256") != page["image_sha256"] for row in rows):
            raise ValueError(f"{page_id}: image checksum mismatch in page CSV")
        if any(row.get("yolo_sha256") != page["yolo_sha256"] for row in rows):
            raise ValueError(f"{page_id}: YOLO checksum mismatch in page CSV")

        report = reports.get(page_id, {})
        status = report.get("status")
        if status not in {"completed", "outside_only"} or report.get("error"):
            raise ValueError(f"{page_id}: existing page run is not reusable")
        repeat_mapping_path = (
            repeat_mapping_dir / f"{page['score_id']}_repeat_mapping.csv"
        )
        fingerprint = _page_fingerprint(
            page,
            scope_rows,
            reviews,
            notes_json_path,
            repeat_mapping_path,
        )
        atomic_write_json(
            output_dir / "checkpoints" / f"{page_id}.json",
            {
                "pipeline_version": PIPELINE_VERSION,
                "page_id": page_id,
                "status": status,
                "fingerprint": fingerprint,
                "row_count": len(rows),
                "validation_errors": [],
                "adopted_from_validated_run": validation.get("run_id", ""),
            },
        )
        created += 1
        emit_progress(
            "checkpoint-bootstrap",
            index,
            len(pages),
            f"{page_id} adopted",
        )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "checkpoints_created": created,
        "alignment_pages_executed": 0,
        "source_run_id": validation.get("run_id", ""),
    }


def run_dataset(
    *, manifest_path: Path, scope_path: Path, notes_json_path: Path,
    repeat_mapping_dir: Path, review_dir: Path, output_dir: Path,
    resume: bool = False, page_ids: set[str] | None = None,
) -> dict:
    pages = select_manifest_pages(_read_csv(manifest_path), page_ids)
    scope_rows = _read_csv(scope_path)
    scope = {
        (row["page_id"], int(row["scan_system_index"])): row for row in scope_rows
    }
    reviews, review_errors = load_human_reviews(review_dir)
    categories = load_categories(notes_json_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_rows = []
    page_reports = []
    bps_cache: dict[str, dict[int, dict]] = {}
    resumed_pages = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        "candidate-pages",
        0,
        len(pages),
        f"starting (resume={resume})",
    )

    for page_index, page in enumerate(pages, start=1):
        page_id = page["page_id"]
        page_csv_path = output_dir / "pages" / f"{page_id}.csv"
        checkpoint_path = output_dir / "checkpoints" / f"{page_id}.json"
        repeat_mapping_path = (
            repeat_mapping_dir / f"{page['score_id']}_repeat_mapping.csv"
        )
        fingerprint = _page_fingerprint(
            page,
            scope_rows,
            reviews,
            notes_json_path,
            repeat_mapping_path,
        )
        if resume:
            resumed = _load_resumable_page(
                checkpoint_path=checkpoint_path,
                page_csv_path=page_csv_path,
                page_id=page_id,
                expected_rows=int(page["yolo_rows"]),
                fingerprint=fingerprint,
            )
            if resumed is not None:
                page_rows, page_report, page_validation_errors = resumed
                all_rows.extend(page_rows)
                page_reports.append(page_report)
                review_errors.extend(page_validation_errors)
                resumed_pages += 1
                emit_progress(
                    "candidate-pages",
                    page_index,
                    len(pages),
                    f"{page_id} resumed ({len(page_rows)} rows)",
                )
                continue

        emit_progress(
            "candidate-pages",
            page_index,
            len(pages),
            f"{page_id} running",
        )
        with Image.open(page["image_path"]) as opened_image:
            image = opened_image.convert("RGB")
        systems = detect_systems(image)
        boxes = load_yolo(Path(page["yolo_path"]), categories=categories)
        has_scope = any(
            scope[(page["page_id"], system.number)]["movement_scope_status"] == "in_bpsd_scope"
            for system in systems
        )
        candidate_by_line = {}
        run_error = ""
        page_status = "outside_only"
        if has_scope:
            candidate_dir = output_dir / "candidates" / page["score_id"] / page["page_id"]
            try:
                report = run_alignment(
                    image_path=Path(page["image_path"]),
                    yolo_path=Path(page["yolo_path"]),
                    xml_path=Path(page["xml_path"]),
                    bps_note_path=Path(page["bps_notes_path"]),
                    output_dir=candidate_dir,
                    page_number=int(page["page"]),
                    infer_fingerings=True,
                    notes_json_path=notes_json_path,
                    include_all_symbols=True,
                    repeat_mapping_path=repeat_mapping_path,
                )
                detailed = _read_csv(Path(report["outputs"]["detailed_csv"]))
                candidate_by_line = {int(row["txt_line"]): row for row in detailed}
                page_status = "completed"
            except Exception as error:
                run_error = f"{type(error).__name__}: {error}"
                page_status = "failed"

        if page["score_id"] not in bps_cache:
            bps_cache[page["score_id"]] = {
                note["note_id"]: note for note in load_bps_notes(Path(page["bps_notes_path"]))
            }
        bps_by_id = bps_cache[page["score_id"]]

        page_rows = []
        page_validation_errors = []
        for box in boxes:
            system_number = assign_system(box, systems, image.height)
            candidate = candidate_by_line.get(box["txt_line"], _empty_candidate(box, system_number))
            system_scope = scope[(page["page_id"], system_number)]
            row = {field: candidate.get(field, "") for field in OFFICIAL_FIELDS}
            row.update({
                "dataset_id": "Xia-BPSD-alignment-v1", "score_id": page["score_id"],
                "page_id": page["page_id"], "scan_page": page["page"],
                "yolo_line": box["txt_line"], "bbox_id": f"{page['page_id']}:Y{box['txt_line']}",
                "image_path": page["image_path"], "yolo_path": page["yolo_path"],
                "xml_path": page["xml_path"], "unfolded_xml_path": page["unfolded_xml_path"],
                "sibelius_path": page["sibelius_path"], "bps_notes_path": page["bps_notes_path"],
                "image_sha256": page["image_sha256"], "yolo_sha256": page["yolo_sha256"],
                "scan_system_index": system_number, "xml_page": system_scope["xml_page"],
                "xml_systems_json": system_scope["xml_systems_json"],
                "written_measure_start": system_scope["written_measure_start"],
                "written_measure_end": system_scope["written_measure_end"],
                "xml_measure": candidate.get("xml_measure", ""), "xml_symbol": candidate.get("xml_symbol", ""),
                "staff": candidate.get("xml_staff", ""), "target_type": candidate.get("target_type", ""),
                "note_ids": candidate.get("note_ids", ""), "pitches": candidate.get("pitches", ""),
                "repeat_occurrences_json": candidate.get("repeat_occurrences_json", ""),
                "repeat_occurrence_count": candidate.get("repeat_occurrence_count", ""),
                "repeat_group_id": candidate.get("repeat_group_id", ""),
                "movement_scope_status": system_scope["movement_scope_status"],
                "page_mapping_status": system_scope["page_mapping_status"],
                "mapping_source": system_scope["mapping_source"],
                "match_source": candidate.get("match_source", ""),
                "confidence": candidate.get("confidence", ""),
                "alignment_status": {"matched": "matched", "inferred": "candidate", "review": "ambiguous"}.get(candidate.get("status"), "unresolved"),
                "review_status": "needs_review", "human_approved": "false", "reviewer": "",
                "reviewed_at": "", "review_source": "", "original_candidate_json": "",
                "corrected_value_json": "", "comment": "", "target_x_px": candidate.get("target_x_px", ""),
                "target_y_px": candidate.get("target_y_px", ""), "pipeline_version": PIPELINE_VERSION,
                "error_code": "page_alignment_failed" if run_error else "", "error_message": run_error,
            })

            if system_scope["movement_scope_status"] == "outside_bpsd_scope":
                for field in ["musical_time", "start_meas", "end_meas", "start_note", "end_note", "connected_note", "xml_measure", "xml_symbol", "staff", "target_type", "note_ids", "pitches", "repeat_occurrences_json", "repeat_occurrence_count", "repeat_group_id", "match_source", "confidence", "target_x_px", "target_y_px"]:
                    row[field] = ""
                row["alignment_status"] = "xml_missing"
                row["review_status"] = "not_required"
            elif system_scope["review_status"] == "needs_review":
                for field in ["start_meas", "end_meas", "start_note", "end_note", "connected_note", "xml_measure", "xml_symbol", "staff", "target_type", "note_ids", "pitches", "repeat_occurrences_json", "repeat_occurrence_count", "repeat_group_id", "match_source", "confidence", "target_x_px", "target_y_px"]:
                    row[field] = ""
                row["alignment_status"] = "unresolved"
                row["error_code"] = "merged_system_mapping_review"
                row["error_message"] = "Scan system combines multiple XML systems; semantic candidate suppressed."

            review = reviews.get((page["page_id"], box["txt_line"]))
            if review is not None:
                reviewed_class = review["row"].get("class", "")
                if reviewed_class and reviewed_class != box["class"]:
                    page_validation_errors.append(
                        f"human review class mismatch {page['page_id']}:Y{box['txt_line']}: "
                        f"review={reviewed_class}, current={box['class']}"
                    )
                    review = None
            if review is not None:
                review_status = review["row"].get("review_status", "")
                row["review_source"] = review["source"]
                row["comment"] = review["row"].get("comment", "")
                if review_status in {"confirmed", "corrected"}:
                    original = {key: row[key] for key in ["start_meas", "end_meas", "start_note", "end_note", "connected_note", "xml_measure", "note_ids", "pitches", "alignment_status"]}
                    values = _human_values(review, bps_by_id)
                    row.update(values)
                    row["original_candidate_json"] = json.dumps(original, ensure_ascii=False)
                    row["corrected_value_json"] = json.dumps(review["row"], ensure_ascii=False)
                    row["alignment_status"] = "matched"
                    row["review_status"] = review_status
                    row["human_approved"] = "true"
                    row["reviewer"] = "human_user"
                    row["match_source"] = "human_review"
                    row["confidence"] = "1.000"
                    row["error_code"] = ""
                    row["error_message"] = ""
            page_rows.append(row)

        atomic_write_csv(page_csv_path, FIELDS, page_rows)
        page_report = {
            "page_id": page_id,
            "status": page_status,
            "error": run_error,
            "resumed": False,
        }
        if page_status in {"completed", "outside_only"}:
            atomic_write_json(
                checkpoint_path,
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "page_id": page_id,
                    "status": page_status,
                    "fingerprint": fingerprint,
                    "row_count": len(page_rows),
                    "validation_errors": page_validation_errors,
                },
            )
        else:
            checkpoint_path.unlink(missing_ok=True)
        all_rows.extend(page_rows)
        page_reports.append(page_report)
        review_errors.extend(page_validation_errors)
        emit_progress(
            "candidate-pages",
            page_index,
            len(pages),
            f"{page_id} {page_status} ({len(page_rows)} rows)",
        )

    page_groups: dict[str, list[dict]] = defaultdict(list)
    score_groups: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        page_groups[row["page_id"]].append(row)
        score_groups[row["score_id"]].append(row)

    emit_progress(
        "aggregate",
        0,
        len(score_groups) + 1,
        "writing sonata and master CSV files",
    )
    for score_index, (score_id, rows) in enumerate(score_groups.items(), start=1):
        atomic_write_csv(output_dir / "sonatas" / f"{score_id}.csv", FIELDS, rows)
        emit_progress(
            "aggregate",
            score_index,
            len(score_groups) + 1,
            f"{score_id} ({len(rows)} rows)",
        )
    master = output_dir / "Xia_BPSD_alignment_master.csv"
    atomic_write_csv(master, FIELDS, all_rows)
    emit_progress(
        "aggregate",
        len(score_groups) + 1,
        len(score_groups) + 1,
        f"master ({len(all_rows)} rows)",
    )

    bbox_ids = [row["bbox_id"] for row in all_rows]
    status_counts = Counter(row["alignment_status"] for row in all_rows)
    validation_errors = list(review_errors)
    expected_rows = sum(int(page["yolo_rows"]) for page in pages)
    if len(all_rows) != expected_rows:
        validation_errors.append(
            f"expected {expected_rows} selected-page rows, got {len(all_rows)}"
        )
    if len(set(bbox_ids)) != len(bbox_ids):
        validation_errors.append("duplicate bbox_id")
    report = {
        "run_id": run_id, "pipeline_version": PIPELINE_VERSION,
        "selected_page_ids": [page["page_id"] for page in pages],
        "expected_selected_rows": expected_rows,
        "pages": len(page_groups), "scores": len(score_groups), "canonical_rows": len(all_rows),
        "unique_bbox_ids": len(set(bbox_ids)), "human_approved_rows": sum(row["human_approved"] == "true" for row in all_rows),
        "outside_bpsd_scope_rows": sum(row["movement_scope_status"] == "outside_bpsd_scope" for row in all_rows),
        "alignment_status_counts": dict(status_counts), "page_runs": page_reports,
        "resumed_pages": resumed_pages,
        "validation_errors": validation_errors, "passed": not validation_errors,
        "outputs": {"master_csv": str(master), "pages_dir": str(output_dir / "pages"), "sonatas_dir": str(output_dir / "sonatas")},
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress(
        "validation",
        1,
        1,
        f"passed={report['passed']} errors={len(validation_errors)}",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--repeat-mapping-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--page-id",
        action="append",
        dest="page_ids",
        help=(
            "Run only this manifest page_id. Repeat for multiple pages; "
            "omit for the full manifest."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid per-page checkpoints and rerun only missing/stale/failed pages.",
    )
    parser.add_argument(
        "--checkpoint-existing-only",
        action="store_true",
        help=(
            "Validate an existing passing run and create per-page checkpoints; "
            "do not execute alignment or rewrite CSV outputs."
        ),
    )
    args = parser.parse_args()
    if args.checkpoint_existing_only:
        result = bootstrap_existing_checkpoints(
            manifest_path=args.manifest,
            scope_path=args.scope,
            notes_json_path=args.notes_json,
            repeat_mapping_dir=args.repeat_mapping_dir,
            review_dir=args.review_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    report = run_dataset(
        manifest_path=args.manifest,
        scope_path=args.scope,
        notes_json_path=args.notes_json,
        repeat_mapping_dir=args.repeat_mapping_dir,
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        page_ids=set(args.page_ids) if args.page_ids else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
