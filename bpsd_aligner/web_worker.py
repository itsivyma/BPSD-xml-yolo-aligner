"""Independent background worker for persisted Streamlit alignment jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

from bps_xml_alignment import load_categories
from bpsd_aligner.job_store import (
    acquire_job_lease,
    build_job_checkpoint_archive,
    job_cancellation_requested,
    load_page_checkpoint,
    publish_completed_job,
    release_job_lease,
    restore_job_checkpoint_archive,
    write_job_status,
    write_page_checkpoint,
)
from bpsd_aligner.pdf_utils import pdf_page_count, render_pdf_page
from bpsd_aligner.web_pipeline import (
    FINAL_BPS_FIELDS,
    PIPELINE_VERSION,
    build_batch_information_outputs,
    prepare_score_sources,
    run_uploaded_alignment,
)
from pipeline_checkpoint import atomic_write_csv, atomic_write_json


class JobCancelled(Exception):
    pass


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _sort_key(entry: tuple[int, int, dict]) -> tuple[float, int, int]:
    page, row_index, row = entry
    try:
        time = float(row.get("start_meas", ""))
    except (TypeError, ValueError):
        time = float("inf")
    return time, page, row_index


def _inside_job(job_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as error:
        raise ValueError(f"background job path is outside its job directory: {path}") from error
    return path


def run_background_job(request_path: Path) -> Path:
    """Execute one serialized web job and return its durable result JSON."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != "1.0":
        raise ValueError("unsupported background job request schema")
    if request.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("background job request uses a different pipeline version")
    job_dir = Path(request["job_dir"]).resolve()
    request_path.resolve().relative_to(job_dir)
    fingerprint = str(request["fingerprint"])
    if job_dir.name != fingerprint:
        raise ValueError("background job directory does not match its fingerprint")

    xml_path = _inside_job(job_dir, request["xml_path"])
    bps_path = _inside_job(job_dir, request["bps_notes_path"])
    notes_path = _inside_job(job_dir, request["notes_json_path"])
    unfolded_path = _inside_job(job_dir, request.get("unfolded_xml_path"))
    clean_pdf_path = _inside_job(job_dir, request.get("clean_pdf_path"))
    resume_archive = _inside_job(job_dir, request.get("resume_archive"))
    if xml_path is None or bps_path is None or notes_path is None:
        raise ValueError("background job is missing required shared inputs")

    pages = request.get("pages", [])
    if not pages:
        raise ValueError("background job has no score pages")
    output_dir = job_dir / "outputs"
    lease = None
    completed_pages = 0
    try:
        queue_timeout = int(os.environ.get("BPSD_ALIGNER_WORKER_QUEUE_TIMEOUT", "3600"))
        deadline = time.monotonic() + max(queue_timeout, 0)
        while lease is None:
            try:
                lease = acquire_job_lease(job_dir)
            except RuntimeError as error:
                if "exact alignment job" in str(error).lower() or time.monotonic() >= deadline:
                    raise
                write_job_status(
                    job_dir,
                    state="queued",
                    stage="waiting_for_worker",
                    total_pages=len(pages),
                    message="All worker slots are busy; this job is waiting in the queue.",
                )
                time.sleep(2)
        if job_cancellation_requested(job_dir):
            raise JobCancelled("Cancellation was requested before alignment started.")
        if resume_archive is not None:
            restore_job_checkpoint_archive(
                resume_archive.read_bytes(),
                job_dir,
                expected_fingerprint=fingerprint,
                expected_pipeline_version=PIPELINE_VERSION,
            )
        if clean_pdf_path is not None:
            page_count = pdf_page_count(clean_pdf_path)
            requested = max(int(page["page_number"]) for page in pages)
            if requested > page_count:
                raise ValueError(
                    f"Clean repetition PDF has {page_count} pages, but page {requested} is required."
                )

        write_job_status(
            job_dir,
            state="running",
            stage="shared_score",
            total_pages=len(pages),
            message="Preparing shared MusicXML and repeat mapping.",
        )
        prepared = prepare_score_sources(
            xml_path=xml_path,
            bps_notes_path=bps_path,
            output_dir=output_dir / "shared_score",
            score_id=request["score_id"],
            unfolded_xml_path=unfolded_path,
        )

        page_reports = []
        final_entries = []
        yolo_entries = []
        detailed_rows = []
        overlays: dict[str, str] = {}
        page_images: dict[str, str] = {}
        xml_events = []
        xml_nodes = []
        for page_index, page in enumerate(pages):
            if job_cancellation_requested(job_dir):
                raise JobCancelled("Cancellation was requested between score pages.")
            page_id = str(page["page_id"])
            page_number = int(page["page_number"])
            image_path = _inside_job(job_dir, page["image_path"])
            yolo_path = _inside_job(job_dir, page["yolo_path"])
            if image_path is None or yolo_path is None:
                raise ValueError(f"page {page_id} is missing image or YOLO input")
            page_images[page_id] = str(image_path)
            checkpoint = load_page_checkpoint(
                job_dir,
                fingerprint=fingerprint,
                pipeline_version=PIPELINE_VERSION,
                page_id=page_id,
                page_number=page_number,
            )
            if checkpoint is not None:
                report = checkpoint["report"]
                resumed = True
            else:
                clean_image = None
                if clean_pdf_path is not None:
                    clean_image = render_pdf_page(
                        clean_pdf_path,
                        page_number,
                        job_dir / "inputs" / "clean_pages" / f"page-{page_number:04d}.png",
                    )
                report = run_uploaded_alignment(
                    image_path=image_path,
                    yolo_path=yolo_path,
                    xml_path=xml_path,
                    bps_notes_path=bps_path,
                    notes_json_path=notes_path,
                    unfolded_xml_path=unfolded_path,
                    clean_image_path=clean_image,
                    output_dir=output_dir / "pages" / page_id,
                    page_number=page_number,
                    score_id=request["score_id"],
                    infer_fingerings=bool(request.get("infer_fingerings", True)),
                    prepared_score=prepared,
                    build_complete_exports=False,
                    system_start_measures=page.get("system_start_measures") or None,
                    page_end_measure=page.get("page_end_measure"),
                )
                write_page_checkpoint(
                    job_dir,
                    fingerprint=fingerprint,
                    pipeline_version=PIPELINE_VERSION,
                    page_id=page_id,
                    page_number=page_number,
                    report=report,
                    page_image=image_path,
                )
                resumed = False

            page_reports.append(report)
            outputs = {name: Path(path) for name, path in report["outputs"].items()}
            _fields, final = _read_csv(outputs["final_bps_csv"])
            for row_index, row in enumerate(final):
                final_entries.append((page_number, row_index, row))
            _fields, yolo = _read_csv(outputs["yolo_aligned_csv"])
            for row_index, row in enumerate(yolo):
                yolo_entries.append((page_number, row_index, row))
            _fields, detailed = _read_csv(outputs["detailed_csv"])
            for row in detailed:
                row["page_id"] = page_id
                row["page_number"] = str(page_number)
            detailed_rows.extend(detailed)
            if not xml_events:
                _fields, xml_events = _read_csv(outputs["xml_events_csv"])
            if not xml_nodes:
                _fields, xml_nodes = _read_csv(outputs["xml_nodes_csv"])
            for name, path in outputs.items():
                if not name.endswith("overlay") or not path.is_file():
                    continue
                key = (
                    f"class_{page_id}__{name.removeprefix('class_')}"
                    if name.startswith("class_")
                    else f"{page_id}__{name}"
                )
                overlays[key] = str(path)
            completed_pages = page_index + 1
            write_job_status(
                job_dir,
                state="running",
                stage="page_alignment",
                completed_pages=completed_pages,
                total_pages=len(pages),
                message=("Resumed" if resumed else "Completed") + f" page: {page_id}",
            )

        final_rows = [entry[2] for entry in sorted(final_entries, key=_sort_key)]
        yolo_rows = [entry[2] for entry in sorted(yolo_entries, key=_sort_key)]
        final_path = output_dir / "bps_omr_final.csv"
        atomic_write_csv(final_path, FINAL_BPS_FIELDS, final_rows)
        detailed_fields = list(dict.fromkeys(field for row in detailed_rows for field in row))
        detailed_path = output_dir / "page_alignment_detailed.csv"
        atomic_write_csv(detailed_path, detailed_fields, detailed_rows)
        complete = build_batch_information_outputs(
            yolo_rows=yolo_rows,
            xml_events=xml_events,
            xml_nodes=xml_nodes,
            output_dir=output_dir / "complete_exports",
        )
        errors = [
            error
            for page_report in page_reports
            for error in page_report.get("validation_errors", [])
        ] + list(complete["validation_errors"])
        warnings = list(
            dict.fromkeys(
                warning
                for page_report in page_reports
                for warning in page_report.get("warnings", [])
            )
        )
        counts = Counter(row.get("status", "") for row in detailed_rows)
        report = {
            "pipeline_version": PIPELINE_VERSION,
            "score_id": request["score_id"],
            "page_count": len(pages),
            "pages": [int(page["page_number"]) for page in pages],
            "alignment_rows": len(final_rows),
            "xml_event_rows": complete["xml_event_rows"],
            "xml_node_rows": complete["xml_node_rows"],
            "timeline_rows": complete["timeline_rows"],
            "all_information_rows": complete["all_information_rows"],
            "xml_span_rows": complete["xml_span_rows"],
            "performance_expanded_rows": complete["performance_expanded_rows"],
            "confirmed_rows": counts.get("matched", 0),
            "human_corrected_rows": 0,
            "yolo_rows_needing_review": len(final_rows) - counts.get("matched", 0),
            "yolo_status_counts": dict(counts),
            "class_overlay_count": sum(
                page_report.get("class_overlay_count", 0) for page_report in page_reports
            ),
            "clean_reference_pages_used": sum(
                bool(page_report.get("clean_reference", {}).get("used"))
                for page_report in page_reports
            ),
            "warnings": warnings,
            "validation_errors": errors,
            "passed": not errors,
            "final_csv_fields": FINAL_BPS_FIELDS,
            "empty_value_policy": "uncertain, unavailable, and not-applicable values are blank",
        }
        report_path = output_dir / "validation_report.json"
        atomic_write_json(report_path, report)
        zip_path = output_dir / "all_outputs.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(final_path, arcname=final_path.name)
            archive.write(report_path, arcname=report_path.name)
            for path in complete["outputs"].values():
                archive.write(path, arcname=f"complete_exports/{Path(path).name}")
            for name, path in overlays.items():
                archive.write(path, arcname=f"review_images/{name}.png")

        write_job_status(
            job_dir,
            state="running",
            stage="packaging_outputs",
            completed_pages=len(pages),
            total_pages=len(pages),
            message="Alignment is complete; packaging durable outputs.",
        )
        checkpoint_path = build_job_checkpoint_archive(
            job_dir, job_dir / "alignment_job_checkpoint.zip"
        )
        result = {
            "schema_version": "1.0",
            "fingerprint": fingerprint,
            "report": report,
            "final_bps_csv": str(final_path),
            **{name: str(path) for name, path in complete["outputs"].items()},
            "detailed_csv": str(detailed_path),
            "validation_json": str(report_path),
            "output_zip": str(zip_path),
            "job_checkpoint_zip": str(checkpoint_path),
            "overlays": overlays,
            "page_images": page_images,
            "bps_note_count": prepared.get("bps_note_count"),
            "bps_note_ids": prepared.get("bps_note_ids"),
            "class_map": {
                str(class_id): name for class_id, name in load_categories(notes_path).items()
            },
        }
        result_path = publish_completed_job(
            job_dir,
            result=result,
            completed_pages=len(pages),
            total_pages=len(pages),
        )
        return result_path
    except JobCancelled as error:
        write_job_status(
            job_dir,
            state="cancelled",
            stage="cancelled",
            completed_pages=completed_pages,
            total_pages=len(pages),
            message=str(error),
        )
        return job_dir / "job_result.json"
    except Exception as error:
        write_job_status(
            job_dir,
            state="failed",
            stage="failed",
            completed_pages=completed_pages,
            total_pages=len(pages),
            message="Background worker stopped before outputs were ready.",
            error=f"{type(error).__name__}: {error}",
        )
        (job_dir / "worker_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        release_job_lease(lease)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    run_background_job(args.request)


if __name__ == "__main__":
    main()
