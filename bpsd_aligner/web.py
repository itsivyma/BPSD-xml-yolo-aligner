"""Streamlit UI for raw score-page alignment and CSV inspection."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from bpsd_aligner.web_pipeline import (
    FINAL_BPS_FIELDS,
    PIPELINE_VERSION,
    build_batch_information_outputs,
    prepare_score_sources,
    run_uploaded_alignment,
)
from bpsd_aligner.pdf_utils import pdf_page_count, render_pdf_page
from bpsd_aligner.job_store import (
    acquire_job_lease,
    build_job_checkpoint_archive,
    job_directory,
    load_page_checkpoint,
    prune_job_store,
    request_job_cancellation,
    release_job_lease,
    restore_job_checkpoint_archive,
    validate_upload_batch,
    validate_page_count,
    write_job_manifest,
    write_job_status,
    write_page_checkpoint,
)
from bpsd_aligner.review_corrections import (
    REVIEW_ACTIONS,
    apply_review_decisions,
    apply_corrections_to_master_rows,
    build_editor_rows,
    build_review_checkpoint,
    build_review_queue,
    csv_bytes,
    load_review_checkpoint,
    render_review_focus_images,
)
from bpsd_aligner.web_utils import (
    apply_page_mapping_edits,
    group_review_overlays,
    pair_page_uploads,
    read_csv_bytes,
    sanitize_path_columns,
    summarize_rows,
)
from combine_yolo_xml import combine_dataset
from pipeline_checkpoint import atomic_write_csv, atomic_write_json


MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _require_access_token() -> None:
    """Optionally protect the app with a deployment-provided access token."""

    expected = os.environ.get("BPSD_ALIGNER_ACCESS_TOKEN", "")
    if not expected or st.session_state.get("bpsd_authenticated"):
        return
    st.title("BPSD XML–YOLO Aligner")
    st.caption("This deployment requires an access token.")
    attempts = int(st.session_state.get("bpsd_login_attempts", 0))
    if attempts >= 5:
        st.error("Too many failed attempts in this browser session. Reload later.")
        st.stop()
    with st.form("bpsd_access_form"):
        supplied = st.text_input("Access token", type="password")
        submitted = st.form_submit_button("Open aligner", type="primary")
    if submitted:
        if hmac.compare_digest(supplied, expected):
            st.session_state["bpsd_authenticated"] = True
            st.session_state["bpsd_login_attempts"] = 0
            st.rerun()
        st.session_state["bpsd_login_attempts"] = attempts + 1
        st.error("Incorrect access token.")
    st.stop()


@st.cache_resource
def _initialize_job_store() -> tuple[str, ...]:
    """Apply an explicitly configured retention policy once per process."""

    configured = os.environ.get("BPSD_ALIGNER_JOB_RETENTION_HOURS", "").strip()
    if not configured:
        return ()
    return tuple(prune_job_store(retention_hours=float(configured)))


def _valid_upload(uploaded, label: str) -> bool:
    if uploaded is None:
        return False
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error(f"{label} exceeds the 200 MB upload limit.")
        return False
    return True


def _save_upload(uploaded, directory: Path, fallback_name: str) -> Path:
    name = Path(uploaded.name).name or fallback_name
    path = directory / name
    path.write_bytes(uploaded.getvalue())
    return path


def _upload_fingerprint(uploaded_files: list, options: dict) -> str:
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for uploaded in uploaded_files:
        if uploaded is None:
            continue
        digest.update(Path(uploaded.name).name.encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def _queue_background_job(
    *,
    fingerprint: str,
    page_pairs: list[dict],
    score_id: str,
    infer_fingerings: bool,
    xml_upload,
    bps_upload,
    notes_upload,
    unfolded_upload,
    clean_pdf_upload,
    resume_job_upload,
) -> Path:
    """Persist uploads and launch an independent worker process."""

    job_dir = job_directory(fingerprint)
    status_path = job_dir / "job_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") in {"queued", "running"}:
            return job_dir
        if status.get("state") == "completed" and (job_dir / "job_result.json").is_file():
            return job_dir
    (job_dir / "cancel_requested.json").unlink(missing_ok=True)
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _save_upload(xml_upload, input_dir, "score.xml")
    bps_path = _save_upload(bps_upload, input_dir, "ann_score_note.csv")
    notes_path = _save_upload(notes_upload, input_dir, "notes.json")
    unfolded_path = (
        _save_upload(unfolded_upload, input_dir, "score_unfolded.xml")
        if unfolded_upload is not None
        else None
    )
    clean_pdf_path = (
        _save_upload(clean_pdf_upload, input_dir, "clean_repetition.pdf")
        if clean_pdf_upload is not None
        else None
    )
    resume_path = (
        _save_upload(resume_job_upload, input_dir, "resume_checkpoint.zip")
        if resume_job_upload is not None
        else None
    )
    page_requests = []
    for pair in page_pairs:
        page_dir = input_dir / "pages" / pair["stem"]
        page_dir.mkdir(parents=True, exist_ok=True)
        image_path = _save_upload(pair["image"], page_dir, "page.png")
        yolo_path = _save_upload(pair["yolo"], page_dir, "page.txt")
        page_requests.append(
            {
                "page_id": pair["stem"],
                "page_number": int(pair["page_number"]),
                "system_start_measures": pair.get("system_start_measures", []),
                "page_end_measure": pair.get("page_end_measure"),
                "image_path": str(image_path),
                "yolo_path": str(yolo_path),
            }
        )
    write_job_manifest(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version=PIPELINE_VERSION,
        inputs=[
            {"name": Path(uploaded.name).name, "size": int(uploaded.size)}
            for uploaded in [
                *(item for pair in page_pairs for item in (pair["image"], pair["yolo"])),
                xml_upload,
                bps_upload,
                notes_upload,
                unfolded_upload,
                clean_pdf_upload,
            ]
            if uploaded is not None
        ],
    )
    request = {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": fingerprint,
        "job_dir": str(job_dir),
        "score_id": score_id,
        "infer_fingerings": infer_fingerings,
        "xml_path": str(xml_path),
        "bps_notes_path": str(bps_path),
        "notes_json_path": str(notes_path),
        "unfolded_xml_path": str(unfolded_path) if unfolded_path else "",
        "clean_pdf_path": str(clean_pdf_path) if clean_pdf_path else "",
        "resume_archive": str(resume_path) if resume_path else "",
        "pages": page_requests,
    }
    request_path = job_dir / "job_request.json"
    atomic_write_json(request_path, request)
    write_job_status(
        job_dir,
        state="queued",
        stage="queued",
        total_pages=len(page_pairs),
        message="Waiting for a background worker slot.",
    )
    log_path = job_dir / "worker.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "bpsd_aligner.web_worker", str(request_path)],
            cwd=Path(__file__).parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    atomic_write_json(
        job_dir / "worker_process.json",
        {"pid": process.pid, "request": str(request_path), "log": str(log_path)},
    )
    return job_dir


def _load_background_result(job_dir: Path, fingerprint: str) -> dict:
    result_path = job_dir / "job_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("fingerprint") != fingerprint:
        raise ValueError("background result fingerprint does not match this session")
    result_version = str(result.get("report", {}).get("pipeline_version", ""))
    if result_version != PIPELINE_VERSION:
        raise ValueError(
            "This result was produced by alignment pipeline "
            f"{result_version or 'unknown'}, but the website now uses "
            f"{PIPELINE_VERSION}. Start a new alignment so stale page geometry "
            "is not reused."
        )
    required_paths = [
        "final_bps_csv",
        "yolo_aligned_csv",
        "xml_events_csv",
        "xml_nodes_csv",
        "yolo_xml_timeline_csv",
        "all_information_csv",
        "combined_master_csv",
        "alignment_links_csv",
        "xml_spans_csv",
        "performance_expanded_timeline_csv",
        "detailed_csv",
        "validation_json",
        "output_zip",
        "job_checkpoint_zip",
    ]
    missing = [name for name in required_paths if not Path(result[name]).is_file()]
    if missing:
        raise ValueError("background result is incomplete: " + ", ".join(missing))
    _fields, detailed_rows = read_csv_bytes(
        Path(result["detailed_csv"]).read_bytes()
    )
    return {
        **{key: value for key, value in result.items() if key != "detailed_csv"},
        "detailed_rows": detailed_rows,
        "review_decisions": {},
    }


@st.fragment(run_every=2.0)
def _render_background_job_status() -> None:
    reference = st.session_state.get("background_alignment_job")
    if not reference:
        return
    job_dir = Path(reference["job_dir"])
    status_path = job_dir / "job_status.json"
    st.subheader("Background alignment job")
    if not status_path.is_file():
        st.warning("The background job has not written a status file yet.")
        if st.button("Refresh background status", key="refresh_background_missing"):
            st.rerun()
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed = int(status.get("completed_pages", 0))
    total = int(status.get("total_pages", 0))
    left, middle, right = st.columns(3)
    left.metric("State", status.get("state", "unknown"))
    middle.metric("Stage", status.get("stage", ""))
    right.metric("Pages", f"{completed}/{total}")
    if total:
        st.progress(min(completed / total, 1.0), text=status.get("message", ""))
    elif status.get("message"):
        st.caption(status["message"])
    controls = st.columns(2)
    if status.get("state") in {"queued", "running"}:
        st.caption("Progress refreshes automatically every 2 seconds. You may close this tab.")
    if controls[0].button("Refresh background status", use_container_width=True):
        st.rerun()
    if status.get("state") == "completed":
        fingerprint = reference["fingerprint"]
        loaded_job = st.session_state.get("raw_alignment_job")
        if loaded_job and loaded_job.get("fingerprint") == fingerprint:
            controls[1].success("Outputs loaded")
        else:
            failure_key = "background_result_load_failure"
            previous_failure = st.session_state.get(failure_key, {})
            result_path = job_dir / "job_result.json"
            if not result_path.is_file():
                st.info(
                    "Alignment is complete; the worker is still packaging the "
                    "durable result. This panel will retry automatically."
                )
                return
            should_load = (
                previous_failure.get("fingerprint") != fingerprint
                or bool(previous_failure.get("transient"))
            )
            if not should_load:
                st.error(
                    "Alignment completed, but the result could not be loaded "
                    f"automatically: {previous_failure.get('error', 'unknown error')}"
                )
                should_load = controls[1].button(
                    "Retry loading outputs", type="primary", use_container_width=True
                )
            if should_load:
                try:
                    job = _load_background_result(job_dir, fingerprint)
                except Exception as error:
                    transient = isinstance(error, FileNotFoundError) or (
                        "result is incomplete" in str(error).lower()
                    )
                    st.session_state[failure_key] = {
                        "fingerprint": fingerprint,
                        "error": f"{type(error).__name__}: {error}",
                        "transient": transient,
                    }
                    if transient:
                        st.info(
                            "Alignment is complete; output files are still being "
                            "finalized. This panel will retry automatically."
                        )
                    else:
                        st.error(
                            "Alignment completed, but the result could not be loaded "
                            f"automatically: {type(error).__name__}: {error}"
                        )
                else:
                    st.session_state["raw_alignment_job"] = job
                    st.session_state.pop(failure_key, None)
                    st.rerun()
    elif status.get("state") in {"queued", "running"}:
        if controls[1].button(
            "Request cancellation", use_container_width=True
        ):
            request_job_cancellation(job_dir)
            st.warning("Cancellation requested. The worker will stop between pages.")
    elif status.get("state") == "failed":
        st.error(status.get("error") or "Background alignment failed.")
        log_path = job_dir / "worker.log"
        if log_path.is_file():
            with st.expander("Worker log", expanded=False):
                st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:])


def _page_checkpoint_artifacts(
    report: dict, *, page_id: str, page_number: int, page_image: bytes | str | Path
) -> dict:
    """Rehydrate the small, portable page state from persistent outputs."""

    outputs = {name: Path(path) for name, path in report["outputs"].items()}
    final_rows = read_csv_bytes(outputs["final_bps_csv"].read_bytes())[1]
    detailed_rows = read_csv_bytes(outputs["detailed_csv"].read_bytes())[1]
    for row in detailed_rows:
        row["page_id"] = page_id
        row["page_number"] = str(page_number)
    overlays = {}
    for name, path in outputs.items():
        if not name.endswith("overlay") or not path.is_file():
            continue
        key = (
            f"class_{page_id}__{name.removeprefix('class_')}"
            if name.startswith("class_")
            else f"{page_id}__{name}"
        )
        overlays[key] = str(path)
    return {
        "report": report,
        "final_rows": final_rows,
        "detailed_rows": detailed_rows,
        "yolo_rows": read_csv_bytes(outputs["yolo_aligned_csv"].read_bytes())[1],
        "xml_event_rows": read_csv_bytes(outputs["xml_events_csv"].read_bytes())[1],
        "xml_node_rows": read_csv_bytes(outputs["xml_nodes_csv"].read_bytes())[1],
        "overlays": overlays,
        "page_image": page_image,
    }


def _asset_bytes(value: bytes | bytearray | str | Path) -> bytes:
    """Read a persisted web artifact only when the selected UI needs it."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return Path(value).read_bytes()


def _render_large_download(
    *,
    label: str,
    value: bytes | bytearray | str | Path,
    file_name: str,
    key: str,
) -> None:
    """Avoid loading a potentially huge archive until the user requests it."""

    if st.checkbox(
        f"Prepare {label}",
        key=f"prepare_{key}",
        help="Large archives are read into download memory only after this is enabled.",
    ):
        st.download_button(
            label,
            _asset_bytes(value),
            file_name,
            "application/zip",
            key=f"download_{key}",
            use_container_width=True,
        )


def _summary_cards(summary: dict[str, object]) -> None:
    first, second, third = st.columns(3)
    first.metric("Rows", f"{summary['rows']:,}")
    second.metric("Scores", summary["scores"])
    third.metric("Statuses", len(summary["statuses"]))


def _apply_and_store_review_outputs(
    job: dict, editor_records: list[dict], reviewer: str
) -> list[str]:
    note_count = job.get("bps_note_count")
    valid_note_ids = job.get("bps_note_ids")
    corrected_rows, corrections, accuracy, errors = apply_review_decisions(
        job["detailed_rows"],
        editor_records,
        reviewer=reviewer.strip() or "User",
        valid_note_ids=(
            {str(value) for value in valid_note_ids}
            if valid_note_ids is not None
            else (
                {str(index) for index in range(int(note_count))}
                if note_count is not None
                else None
            )
        ),
        class_map=job.get("class_map"),
    )
    if errors:
        return errors
    corrections_bytes = json.dumps(
        corrections, ensure_ascii=False, indent=2
    ).encode("utf-8")
    accuracy_bytes = json.dumps(
        accuracy, ensure_ascii=False, indent=2
    ).encode("utf-8")
    master_fields, master_rows = read_csv_bytes(_asset_bytes(job["yolo_aligned_csv"]))
    corrected_master = apply_corrections_to_master_rows(master_rows, corrections)
    xml_event_fields, xml_events = read_csv_bytes(_asset_bytes(job["xml_events_csv"]))
    xml_node_fields, xml_nodes = read_csv_bytes(_asset_bytes(job["xml_nodes_csv"]))
    corrected_images = {}
    correction_by_key = {
        f"{entry.get('page_id')}:Y{entry.get('yolo_line')}": entry
        for entry in corrections["entries"]
    }
    for source in job["detailed_rows"]:
        key = f"{source.get('page_id')}:Y{source.get('txt_line')}"
        correction = correction_by_key.get(key)
        image_data = _workspace_source_image(job, source)
        if correction is None or image_data is None:
            continue
        image_row = dict(source)
        image_row.update(correction.get("corrected", {}))
        image_row["review_target_x_px"] = correction.get(
            "review_target_x_px", ""
        )
        image_row["review_target_y_px"] = correction.get(
            "review_target_y_px", ""
        )
        try:
            full, crop = render_review_focus_images(image_data, image_row)
        except ValueError:
            continue
        corrected_images[f"{key}_full.png"] = full
        corrected_images[f"{key}_crop.png"] = crop

    with tempfile.TemporaryDirectory(prefix="bpsd-corrected-") as temporary:
        target_dir = Path(temporary)
        atomic_write_csv(target_dir / "yolo_aligned_corrected.csv", master_fields, corrected_master)
        atomic_write_csv(target_dir / "xml_events.csv", xml_event_fields, xml_events)
        atomic_write_csv(target_dir / "xml_nodes.csv", xml_node_fields, xml_nodes)
        rebuilt = build_batch_information_outputs(
            yolo_rows=corrected_master,
            xml_events=xml_events,
            xml_nodes=xml_nodes,
            output_dir=target_dir / "complete",
        )
        if not rebuilt["passed"]:
            return list(rebuilt["validation_errors"])
        corrected_csv = csv_bytes(corrected_rows, FINAL_BPS_FIELDS)
        archive_path = target_dir / "bpsd_alignment_corrected_outputs.zip"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("bps_omr_final_corrected.csv", corrected_csv)
            archive.writestr("human_corrections.json", corrections_bytes)
            archive.writestr("accuracy_report.json", accuracy_bytes)
            for name, path in rebuilt["outputs"].items():
                archive.write(path, arcname=f"corrected/{path.name}")
            for name, data in corrected_images.items():
                archive.writestr(f"corrected_review_images/{name}", data)
        complete_outputs = {
            name: path.read_bytes() for name, path in rebuilt["outputs"].items()
        }
        corrected_zip = archive_path.read_bytes()
    job["human_review_outputs"] = {
        "corrected_csv": corrected_csv,
        "corrections_json": corrections_bytes,
        "accuracy_json": accuracy_bytes,
        "accuracy": accuracy,
        "decisions": len(corrections["entries"]),
        "complete_outputs": complete_outputs,
        "corrected_images": corrected_images,
        "corrected_zip": corrected_zip,
    }
    st.session_state["raw_alignment_job"] = job
    return []


def _workspace_source_image(job: dict, row: dict) -> bytes | None:
    page_id = str(row.get("page_id", ""))
    page_images = job.get("page_images", {})
    if page_id in page_images:
        return _asset_bytes(page_images[page_id])
    overlays = job.get("overlays", {})
    selected = (
        overlays.get(f"{page_id}__review_overlay")
        or overlays.get(f"{page_id}__all_symbols_overlay")
    )
    return _asset_bytes(selected) if selected is not None else None


def _render_review_outputs(job: dict) -> None:
    review_outputs = job.get("human_review_outputs")
    if not review_outputs:
        return
    st.subheader("Corrected review outputs")
    first_download, second_download, third_download = st.columns(3)
    first_download.download_button(
        "Corrected BPS-OMR CSV",
        review_outputs["corrected_csv"],
        "bps_omr_final_corrected.csv",
        "text/csv",
        use_container_width=True,
    )
    second_download.download_button(
        "Corrections JSON",
        review_outputs["corrections_json"],
        "human_corrections.json",
        "application/json",
        use_container_width=True,
    )
    third_download.download_button(
        "Accuracy report JSON",
        review_outputs["accuracy_json"],
        "accuracy_report.json",
        "application/json",
        use_container_width=True,
    )
    st.caption(
        "Accuracy compares original machine values with reviewed ground truth. "
        "Blank or uncertain ground-truth fields are not scored."
    )
    with st.expander("Accuracy details"):
        st.json(review_outputs["accuracy"])
    st.download_button(
        "Download all corrected CSVs + review images ZIP",
        review_outputs["corrected_zip"],
        "bpsd_alignment_corrected_outputs.zip",
        "application/zip",
        use_container_width=True,
    )
    with st.expander("Corrected complete-data CSVs", expanded=False):
        labels = {
            "yolo_aligned_csv": "Corrected YOLO Align CSV",
            "yolo_xml_timeline_csv": "Corrected XML + YOLO Timeline",
            "all_information_csv": "Corrected All Information",
            "combined_master_csv": "Corrected Combined Master",
            "alignment_links_csv": "Corrected Alignment Links",
            "xml_spans_csv": "XML Spans",
            "performance_expanded_timeline_csv": (
                "Corrected Performance-expanded Timeline"
            ),
        }
        for name, label in labels.items():
            data = review_outputs["complete_outputs"].get(name)
            if data is not None:
                st.download_button(
                    label,
                    data,
                    f"{name.removesuffix('_csv')}_corrected.csv",
                    "text/csv",
                    key=f"corrected_download_{name}",
                    use_container_width=True,
                )
    if review_outputs["corrected_images"]:
        with st.expander("Corrected review images", expanded=False):
            image_name = st.selectbox(
                "Correction image",
                sorted(review_outputs["corrected_images"]),
                key="corrected_review_image_select",
            )
            st.image(
                review_outputs["corrected_images"][image_name],
                caption=image_name,
                use_container_width=True,
            )


def _candidate_printed_measure(candidate: dict) -> str:
    """Return the scan/printed measure label used by the review UI.

    New results carry this explicitly. The conservative fallback repairs
    older pickup-score results only when BPSD time and XML measure differ by
    at most one; large repeat-expanded differences must not be guessed.
    """

    explicit = str(candidate.get("printed_measure", "") or "").strip()
    if explicit:
        return explicit
    xml_measure = str(candidate.get("xml_measure", "") or "").strip()
    try:
        timeline_measure = str(int(float(candidate.get("start_meas", ""))))
        if xml_measure and abs(int(xml_measure) - int(timeline_measure)) <= 1:
            return timeline_measure
    except (TypeError, ValueError):
        pass
    return xml_measure


def _review_note_label(candidate: dict) -> str:
    """Describe a note without requiring the reviewer to know its BPSD ID."""

    staff = str(candidate.get("staff", ""))
    staff_label = {"1": "上 staff", "2": "下 staff"}.get(
        staff, f"staff {staff or '—'}"
    )
    order = candidate.get("measure_note_order")
    order_label = f"第 {order} 個音" if str(order or "").strip() else "音符"
    return (
        f"第 {_candidate_printed_measure(candidate) or '—'} 小節 · "
        f"{staff_label} · "
        f"{candidate.get('pitch') or '音高不明'} · {order_label}"
    )


def _fill_missing_note_orders(candidates: list[dict]) -> None:
    """Backfill measure-local note order for results made by older versions."""

    groups: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(
            (_candidate_printed_measure(candidate), str(candidate.get("staff", ""))),
            [],
        ).append(candidate)
    for group in groups.values():
        group.sort(
            key=lambda candidate: (
                float(candidate.get("x_px", 0) or 0),
                float(candidate.get("y_px", 0) or 0),
                str(candidate.get("note_id", "")),
            )
        )
        for order, candidate in enumerate(group, start=1):
            candidate.setdefault("measure_note_order", order)


def _merge_page_note_candidates(
    current_candidates: list[dict], detailed_rows: list[dict], page_id: str
) -> list[dict]:
    """Recover a page-wide index from local candidates in legacy results."""

    merged = []
    seen = set()
    candidate_groups = [current_candidates]
    for row in detailed_rows:
        if str(row.get("page_id", "")) != str(page_id):
            continue
        try:
            row_candidates = json.loads(
                str(row.get("review_note_candidates_json", "") or "[]")
            )
        except json.JSONDecodeError:
            continue
        if isinstance(row_candidates, list):
            candidate_groups.append(row_candidates)
    for group in candidate_groups:
        for candidate in group:
            if not isinstance(candidate, dict):
                continue
            sequence = str(candidate.get("xml_note_sequence", "")).strip()
            identity = (
                ("sequence", sequence)
                if sequence
                else (
                    "geometry",
                    str(candidate.get("note_id", "")),
                    str(candidate.get("xml_measure", "")),
                    str(candidate.get("printed_measure", "")),
                    str(candidate.get("staff", "")),
                    str(candidate.get("pitch", "")),
                    str(candidate.get("x_px", "")),
                    str(candidate.get("y_px", "")),
                )
            )
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(candidate)
    return merged


def _endpoint_note_input_value(candidate: dict | None) -> str:
    if candidate is None:
        return ""
    staff = {"1": "上", "2": "下"}.get(
        str(candidate.get("staff", "")), str(candidate.get("staff", ""))
    )
    return ", ".join(
        [
            _candidate_printed_measure(candidate),
            staff,
            str(candidate.get("pitch", "")),
            str(candidate.get("measure_note_order", "")),
        ]
    )


def _candidate_note_id(candidate: dict | None) -> str:
    if candidate is None or candidate.get("note_id") is None:
        return ""
    return str(candidate["note_id"])


def _normalize_pitch(value: object) -> str:
    return str(value or "").strip().replace("♯", "#").replace("♭", "b").upper()


def _resolve_endpoint_note_input(
    value: str, candidates: list[dict]
) -> tuple[dict | None, str]:
    """Resolve `measure, staff, pitch, order` into one XML/BPSD note."""

    compact = str(value).strip()
    if not compact:
        return None, ""
    parts = [part.strip() for part in re.split(r"[,，/|]", compact)]
    if len(parts) != 4:
        return None, "請輸入四項：小節, staff, 音高, 第幾個音"
    measure = re.sub(r"^(第)?|小節$", "", parts[0]).strip()
    staff_text = parts[1].lower().replace("staff", "").strip()
    staff = {"上": "1", "upper": "1", "下": "2", "lower": "2"}.get(
        staff_text, staff_text
    )
    pitch = _normalize_pitch(parts[2])
    order = re.sub(r"^(第)?|個音$|音$", "", parts[3]).strip()
    matches = [
        candidate
        for candidate in candidates
        if _candidate_printed_measure(candidate) == measure
        and str(candidate.get("staff", "")) == staff
        and _normalize_pitch(candidate.get("pitch")) == pitch
        and str(candidate.get("measure_note_order", "")) == order
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, "找到多個相同音符，請確認該小節第幾個音"
    return None, "找不到這個音符，請檢查小節、staff、音高與順序"


def _snap_click_to_note_candidate(
    click: dict,
    candidates: list[dict],
    crop_geometry: dict,
    *,
    max_display_distance: float = 42.0,
) -> tuple[dict | None, float | None]:
    """Snap a displayed-image click to the nearest XML notehead candidate."""

    try:
        click_x = float(click["x"])
        click_y = float(click["y"])
        display_width = float(click["width"])
        display_height = float(click["height"])
        crop_width = float(crop_geometry["width"])
        crop_height = float(crop_geometry["height"])
        crop_left = float(crop_geometry["left"])
        crop_top = float(crop_geometry["top"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if min(display_width, display_height, crop_width, crop_height) <= 0:
        return None, None

    closest = None
    closest_distance = None
    for candidate in candidates:
        try:
            displayed_x = (
                (float(candidate["x_px"]) - crop_left)
                * display_width
                / crop_width
            )
            displayed_y = (
                (float(candidate["y_px"]) - crop_top)
                * display_height
                / crop_height
            )
        except (KeyError, TypeError, ValueError):
            continue
        distance = ((click_x - displayed_x) ** 2 + (click_y - displayed_y) ** 2) ** 0.5
        if closest_distance is None or distance < closest_distance:
            closest = candidate
            closest_distance = distance
    if closest_distance is None or closest_distance > max_display_distance:
        return None, closest_distance
    return closest, closest_distance


def _handle_note_image_click(
    click: dict | None,
    candidates: list[dict],
    geometry: dict,
    *,
    review_key: str,
    surface: str,
    click_mode: str,
    manual_start_key: str,
    manual_end_key: str,
    feedback_key: str,
    is_span: bool,
) -> bool:
    """Apply one new image click to review state and request a UI rerun."""

    if not isinstance(click, dict) or not click.get("unix_time"):
        return False
    last_click_key = f"workspace_last_image_click_{surface}_{review_key}"
    click_token = click["unix_time"]
    if st.session_state.get(last_click_key) == click_token:
        return False
    st.session_state[last_click_key] = click_token
    snapped, distance = _snap_click_to_note_candidate(click, candidates, geometry)
    if snapped is None:
        distance_text = (
            f"（最近距離 {distance:.0f}px）" if distance is not None else ""
        )
        st.session_state[feedback_key] = (
            "error",
            "點擊位置離 XML 音頭太遠，未套用" + distance_text,
        )
        return True

    selected_value = _endpoint_note_input_value(snapped)
    target_key = manual_start_key if click_mode == "開始音符" else manual_end_key
    st.session_state[target_key] = selected_value
    if not is_span:
        st.session_state[manual_start_key] = selected_value
        st.session_state[manual_end_key] = selected_value
    note_id = _candidate_note_id(snapped)
    suffix = f"；note ID {note_id}" if note_id else "；此 XML 音符沒有 BPSD note ID"
    st.session_state[feedback_key] = (
        "success",
        f"已選{click_mode}：{_review_note_label(snapped)}{suffix}",
    )
    return True


def _render_review_workspace(job: dict) -> None:
    st.subheader("人工複核")
    st.caption("每次只確認一個符號：看圖 → 選答案 → 儲存並自動到下一筆。")
    detailed_rows = job["detailed_rows"]
    decisions = job.setdefault("review_decisions", {})
    pages = sorted({str(row.get("page_id", "")) for row in detailed_rows if row.get("page_id")})
    classes = sorted({str(row.get("class", "")) for row in detailed_rows if row.get("class")})
    with st.expander("篩選與跳轉（通常不用調整）", expanded=False):
        filter_left, filter_middle, filter_right = st.columns(3)
        selected_page = filter_left.selectbox(
            "頁面",
            ["All pages", *pages],
            key="workspace_page_filter",
        )
        selected_class = filter_middle.selectbox(
            "符號 class",
            ["All classes", *classes],
            key="workspace_class_filter",
        )
        queue_mode = filter_right.selectbox(
            "顯示範圍",
            ["Needs review", "All including matched", "Matched spot check"],
            key="workspace_queue_mode",
        )
    include_matched = queue_mode != "Needs review"
    machine_status = "matched" if queue_mode == "Matched spot check" else None
    queue = build_review_queue(
        detailed_rows,
        page_id=None if selected_page == "All pages" else selected_page,
        class_name=None if selected_class == "All classes" else selected_class,
        machine_status=machine_status,
        include_matched=include_matched,
        decisions=decisions,
    )
    if not queue:
        st.success("No symbols match the current review filters.")
        return

    reviewed_count = sum(
        row["saved_action"] not in {"pending", "skipped"} for row in queue
    )
    st.progress(
        reviewed_count / len(queue),
        text=f"已完成 {reviewed_count} / {len(queue)} 筆",
    )
    queue_keys = [row["review_key"] for row in queue]
    selector_key = "workspace_item_selector"
    pending_key = "workspace_pending_item"
    pending_item = st.session_state.pop(pending_key, None)
    if pending_item in queue_keys:
        # This runs before the selectbox is instantiated, which is the only
        # point at which Streamlit permits programmatic widget-state changes.
        st.session_state[selector_key] = pending_item
    if st.session_state.get(selector_key) not in queue_keys:
        st.session_state[selector_key] = queue_keys[0]
    selected_key = st.selectbox(
        "目前項目（可直接跳到其他筆）",
        queue_keys,
        format_func=lambda key: next(
            (
                f"{index + 1}/{len(queue)}  {row.get('page_id')}  "
                f"Y{row.get('txt_line')}  {row.get('class')}  "
                f"[{row.get('status')}; saved={row.get('saved_action')}]"
            )
            for index, row in enumerate(queue)
            if row["review_key"] == key
        ),
        key=selector_key,
    )
    current_index = next(
        index for index, row in enumerate(queue) if row["review_key"] == selected_key
    )
    current = queue[current_index]
    saved = decisions.get(current["review_key"], {})
    nav_left, nav_center, nav_right = st.columns([1, 3, 1])
    if nav_left.button(
        "← 上一筆",
        disabled=current_index == 0,
        key="workspace_previous",
        use_container_width=True,
    ):
        st.session_state[pending_key] = queue[current_index - 1]["review_key"]
        st.rerun()
    nav_center.markdown(
        f"**{current.get('page_id')} · Y{current.get('txt_line')} · "
        f"{current.get('class')}**"
    )
    if nav_right.button(
        "下一筆 →",
        disabled=current_index == len(queue) - 1,
        key="workspace_next",
        use_container_width=True,
    ):
        st.session_state[pending_key] = queue[current_index + 1]["review_key"]
        st.rerun()

    try:
        note_candidates = json.loads(
            str(current.get("review_note_candidates_json", "") or "[]")
        )
    except json.JSONDecodeError:
        note_candidates = []
    note_candidates = [
        candidate for candidate in note_candidates if isinstance(candidate, dict)
    ]
    is_span = str(current.get("target_type", "")) == "span"
    if is_span:
        note_candidates = _merge_page_note_candidates(
            note_candidates, detailed_rows, str(current.get("page_id", ""))
        )
    _fill_missing_note_orders(note_candidates)
    selected_candidate = None
    selected_candidate_key = "keep"
    candidate_widget_key = f"workspace_note_candidate_{current['review_key']}"
    current_start_note = str(saved.get("start_note", current.get("start_note", "")))
    current_end_note = str(saved.get("end_note", current.get("end_note", "")))
    current_start_candidate = next(
        (
            candidate
            for candidate in note_candidates
            if str(candidate.get("note_id", "")) == current_start_note
        ),
        None,
    )
    current_end_candidate = next(
        (
            candidate
            for candidate in note_candidates
            if str(candidate.get("note_id", "")) == current_end_note
        ),
        None,
    )
    manual_start_key = f"workspace_start_note_manual_{current['review_key']}"
    manual_end_key = f"workspace_end_note_manual_{current['review_key']}"
    if manual_start_key not in st.session_state:
        st.session_state[manual_start_key] = _endpoint_note_input_value(
            current_start_candidate
        )
    if manual_end_key not in st.session_state:
        st.session_state[manual_end_key] = _endpoint_note_input_value(
            current_end_candidate
        )
    preview_start_endpoint, _preview_start_error = _resolve_endpoint_note_input(
        st.session_state[manual_start_key], note_candidates
    )
    preview_end_endpoint, _preview_end_error = _resolve_endpoint_note_input(
        st.session_state[manual_end_key], note_candidates
    )
    if note_candidates and not is_span:
        candidate_keys = [
            "keep",
            *[str(index) for index in range(1, len(note_candidates) + 1)],
        ]

        def candidate_label(key: str) -> str:
            if key == "keep":
                return "機器圈選正確，不需更換音頭"
            candidate = note_candidates[int(key) - 1]
            return (
                f"橘色 #{key}｜印刷小節 "
                f"{_candidate_printed_measure(candidate) or '—'}｜"
                f"staff {candidate.get('staff') or '—'}｜"
                f"音高 {candidate.get('pitch') or '—'}｜"
                f"時間 {candidate.get('start_meas') or '—'}"
            )
        selected_candidate_key = st.session_state.get(candidate_widget_key, "keep")
        if selected_candidate_key not in candidate_keys:
            selected_candidate_key = "keep"
        if selected_candidate_key != "keep":
            selected_candidate = note_candidates[int(selected_candidate_key) - 1]

    image_row = dict(current)
    preview_candidate = selected_candidate
    if preview_candidate is not None:
        image_row["review_target_x_px"] = preview_candidate.get("x_px", "")
        image_row["review_target_y_px"] = preview_candidate.get("y_px", "")
    elif saved.get("review_target_x_px") and saved.get("review_target_y_px"):
        image_row["review_target_x_px"] = saved["review_target_x_px"]
        image_row["review_target_y_px"] = saved["review_target_y_px"]
    if preview_start_endpoint is not None:
        image_row["review_start_target_x_px"] = preview_start_endpoint.get("x_px", "")
        image_row["review_start_target_y_px"] = preview_start_endpoint.get("y_px", "")
    if preview_end_endpoint is not None:
        image_row["review_end_target_x_px"] = preview_end_endpoint.get("x_px", "")
        image_row["review_end_target_y_px"] = preview_end_endpoint.get("y_px", "")

    click_mode = "開始音符"
    if note_candidates:
        click_mode = st.radio(
            "直接點圖片選音頭",
            ["開始音符", "結束音符"],
            horizontal=True,
            key=f"workspace_click_mode_{current['review_key']}",
            help="選擇目標後，直接點左側高解析度圖片中的音頭。",
        )

    st.markdown(
        """
        <style>
        .st-key-review_sticky_image {
            position: sticky;
            top: 3.75rem;
            z-index: 20;
            background: var(--background-color, white);
            padding: 0.35rem 0.5rem 0.75rem;
            border-radius: 0.75rem;
            height: 68vh;
            overflow-x: hidden;
            overflow-y: auto;
        }
        .st-key-review_sticky_image img {
            max-height: 48vh;
            object-fit: contain;
        }
        div[data-testid="stColumn"]:has(.st-key-review_controls_marker) {
            height: 68vh;
            overflow-y: auto;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
            padding: 0.35rem 0.75rem 0.75rem 0.25rem;
            border-radius: 0.75rem;
        }
        .st-key-review_action_bar {
            position: sticky;
            bottom: 0;
            z-index: 30;
            background: var(--background-color, white);
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            padding: 0.65rem 0 0.35rem;
        }
        @media (max-width: 900px) {
            .st-key-review_sticky_image {
                position: static;
                height: auto;
                overflow: visible;
            }
            div[data-testid="stColumn"]:has(.st-key-review_controls_marker) {
                height: auto;
                overflow: visible;
                padding: 0;
            }
            .st-key-review_action_bar {
                position: static;
                border-top: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    image_column, controls_column = st.columns([1, 1], gap="large")
    source_image = _workspace_source_image(job, current)
    with image_column:
        with st.container(key="review_sticky_image"):
            st.markdown("#### ① 圖片保持在這裡")
            if source_image is not None:
                try:
                    full_image, crop_image, crop_geometry = render_review_focus_images(
                        source_image,
                        image_row,
                        include_crop_geometry=True,
                    )
                except ValueError as error:
                    st.warning(str(error))
                else:
                    if note_candidates:
                        feedback_key = (
                            f"workspace_image_click_feedback_{current['review_key']}"
                        )
                        with Image.open(io.BytesIO(crop_image)) as rendered_crop:
                            click = streamlit_image_coordinates(
                                rendered_crop.copy(),
                                use_column_width="always",
                                key=f"review_click_image_{current['review_key']}",
                                cursor="crosshair",
                            )
                        if _handle_note_image_click(
                            click,
                            note_candidates,
                            crop_geometry,
                            review_key=current["review_key"],
                            surface="crop",
                            click_mode=click_mode,
                            manual_start_key=manual_start_key,
                            manual_end_key=manual_end_key,
                            feedback_key=feedback_key,
                            is_span=is_span,
                        ):
                            st.rerun()
                        feedback = st.session_state.get(feedback_key)
                        if feedback:
                            getattr(st, feedback[0])(feedback[1])
                        clear_start, clear_end = st.columns(2)
                        if clear_start.button(
                            "清除開始音符",
                            key=f"clear_clicked_start_{current['review_key']}",
                            use_container_width=True,
                        ):
                            st.session_state[manual_start_key] = ""
                            st.session_state.pop(feedback_key, None)
                            st.rerun()
                        if clear_end.button(
                            "清除結束音符",
                            key=f"clear_clicked_end_{current['review_key']}",
                            use_container_width=True,
                        ):
                            st.session_state[manual_end_key] = ""
                            st.session_state.pop(feedback_key, None)
                            st.rerun()
                    else:
                        st.image(crop_image, use_container_width=True)
                    st.caption(
                        "紅框＝YOLO；青／紫＝機器答案；綠色 S✓／紫色 E✓＝你的點選"
                    )
                    has_notehead_target = any(
                        all(
                            str(current.get(field, "")).strip() not in {"", "NA"}
                            for field in fields
                        )
                        for fields in (
                            ("target_x_px", "target_y_px"),
                            ("end_target_x_px", "end_target_y_px"),
                        )
                    )
                    if has_notehead_target:
                        st.caption(
                            f"目前模式：{click_mode}。直接點音頭，系統會吸附到最近的 XML 音符。"
                        )
                    else:
                        st.warning("這筆沒有可畫出的機器音頭座標，請使用右側進階欄位。")
                    with st.expander("顯示完整頁", expanded=False):
                        if note_candidates:
                            with Image.open(io.BytesIO(full_image)) as rendered_full:
                                full_click = streamlit_image_coordinates(
                                    rendered_full.copy(),
                                    use_column_width="always",
                                    key=(
                                        f"review_click_full_page_"
                                        f"{current['review_key']}"
                                    ),
                                    cursor="crosshair",
                                )
                            full_geometry = {
                                "left": 0,
                                "top": 0,
                                "width": crop_geometry["page_width"],
                                "height": crop_geometry["page_height"],
                            }
                            if _handle_note_image_click(
                                full_click,
                                note_candidates,
                                full_geometry,
                                review_key=current["review_key"],
                                surface="full",
                                click_mode=click_mode,
                                manual_start_key=manual_start_key,
                                manual_end_key=manual_end_key,
                                feedback_key=feedback_key,
                                is_span=is_span,
                            ):
                                st.rerun()
                            st.caption(
                                "完整頁也可直接點選；適合跨 system 的 slur／tie。"
                            )
                        else:
                            st.image(full_image, use_container_width=True)
            else:
                st.info("舊工作階段沒有原始頁面圖片；請用目前版本重新 Align。")

    with controls_column:
        with st.container(key="review_controls_marker"):
            st.caption("右欄可獨立捲動；左側圖片會保持可見。")
        st.markdown("#### ② 在這裡選擇或輸入")
        if note_candidates and not is_span:
            selected_candidate_key = st.selectbox(
                "機器圈錯音時，請選正確的橘色編號",
                candidate_keys,
                format_func=candidate_label,
                key=candidate_widget_key,
                help="選擇後，左圖會用綠色 C 預覽新的音頭。",
            )
            selected_candidate = (
                None
                if selected_candidate_key == "keep"
                else note_candidates[int(selected_candidate_key) - 1]
            )
            st.caption("選橘色編號即可更正，不需要自己查 note ID。")
        elif not is_span:
            st.info("舊版結果沒有音頭候選；重新 Align 這一批頁面後即可使用。")

        with st.expander("機器判定詳情", expanded=False):
            detail_left, detail_right = st.columns(2)
            detail_left.metric("狀態", current.get("status", "") or "—")
            detail_right.metric("信心值", current.get("confidence", "") or "—")
            st.caption(
                f"時間 {current.get('start_meas') or '—'} → "
                f"{current.get('end_meas') or '—'} · staff "
                f"{current.get('xml_staff') or '—'}"
            )

        with st.expander(
            "進階：輸入開始／結束音符，或修改時間、staff、class", expanded=is_span
        ):
            if note_candidates:
                endpoint_left, endpoint_right = st.columns(2)
                with endpoint_left:
                    start_input = st.text_input(
                        "開始音符",
                        value=_endpoint_note_input_value(current_start_candidate),
                        key=manual_start_key,
                        placeholder="例如：52, 下, E3, 2",
                    )
                with endpoint_right:
                    end_input = st.text_input(
                        "結束音符",
                        value=_endpoint_note_input_value(current_end_candidate),
                        key=manual_end_key,
                        placeholder="例如：53, 上, G4, 1",
                    )
                start_endpoint, start_error = _resolve_endpoint_note_input(
                    start_input, note_candidates
                )
                end_endpoint, end_error = _resolve_endpoint_note_input(
                    end_input, note_candidates
                )
                endpoint_errors = [error for error in (start_error, end_error) if error]
                if start_error:
                    st.error(f"開始音符：{start_error}")
                if end_error:
                    st.error(f"結束音符：{end_error}")
                st.caption(
                    "格式：小節, 上／下, 音高, 該 staff 在小節內由左到右第幾個音。"
                    "例如 `52, 下, E3, 2`；清空代表不確定並留空。"
                )
                start_note = _candidate_note_id(start_endpoint)
                end_note = _candidate_note_id(end_endpoint)
                start_choice = start_input
                end_choice = end_input
            else:
                start_endpoint = end_endpoint = None
                start_choice = end_choice = "keep"
                endpoint_errors = []
                st.warning("這是舊版結果，沒有可選音符清單；可暫時輸入 note ID。")
                legacy_left, legacy_right = st.columns(2)
                start_note = legacy_left.text_input(
                    "開始 note ID",
                    value=current_start_note,
                    key=f"workspace_start_note_{current['review_key']}",
                )
                end_note = legacy_right.text_input(
                    "結束 note ID",
                    value=current_end_note,
                    key=f"workspace_end_note_{current['review_key']}",
                )
            field_left, field_middle = st.columns(2)
            start_meas = field_left.text_input(
                "開始時間",
                value=str(
                    start_endpoint.get("start_meas", "")
                    if start_endpoint is not None
                    else saved.get("start_meas", current.get("start_meas", ""))
                ),
                key=(
                    f"workspace_start_{current['review_key']}_{start_choice}"
                ),
            )
            end_meas = field_middle.text_input(
                "結束時間",
                value=str(
                    end_endpoint.get("end_meas", "")
                    if end_endpoint is not None
                    else saved.get("end_meas", current.get("end_meas", ""))
                ),
                key=f"workspace_end_{current['review_key']}_{end_choice}",
            )
            staff = st.text_input(
                "Staff",
                value=str(saved.get("staff", current.get("xml_staff", ""))),
                key=f"workspace_staff_{current['review_key']}",
            )
            connected_note = st.text_input(
                "Connected note IDs",
                value=str(saved.get("connected_note", current.get("connected_note", ""))),
                key=f"workspace_connected_{current['review_key']}",
            )
            corrected_class_id = st.text_input(
                "更正後 class ID",
                value=str(saved.get("corrected_class_id", current.get("class_id", ""))),
                key=f"workspace_class_id_{current['review_key']}",
            )
            corrected_class = st.text_input(
                "更正後 class 名稱",
                value=str(saved.get("corrected_class", current.get("class", ""))),
                key=f"workspace_class_{current['review_key']}",
            )
            comment = st.text_area(
                "人工複核備註",
                value=str(saved.get("comment", "")),
                key=f"workspace_comment_{current['review_key']}",
            )

    def save_decision(action: str, note_candidate: dict | None = None) -> None:
        corrected_target = note_candidate or {}
        corrected_note_id = _candidate_note_id(corrected_target)
        if action == "confirm" and note_candidate is None:
            decision_start_note = str(current.get("start_note", ""))
            decision_end_note = str(current.get("end_note", ""))
            corrected_start = str(current.get("start_meas", ""))
            corrected_end = str(current.get("end_meas", ""))
            corrected_staff = str(current.get("xml_staff", ""))
            corrected_connected = str(current.get("connected_note", ""))
        else:
            decision_start_note = corrected_note_id or start_note
            decision_end_note = corrected_note_id or end_note
            corrected_start = str(corrected_target.get("start_meas", start_meas))
            corrected_end = str(corrected_target.get("end_meas", end_meas))
            corrected_staff = str(corrected_target.get("staff", staff))
            corrected_connected = str(
                corrected_target.get("connected_note", connected_note)
            )
        decisions[current["review_key"]] = {
            "action": action,
            "page_id": str(current.get("page_id", "")),
            "yolo_line": str(current.get("txt_line", "")),
            "class": str(current.get("class", "")),
            "corrected_class_id": corrected_class_id,
            "corrected_class": corrected_class,
            "machine_status": str(current.get("status", "")),
            "start_meas": corrected_start,
            "end_meas": corrected_end,
            "start_note": decision_start_note,
            "end_note": decision_end_note,
            "connected_note": (
                json.dumps(
                    [
                        int(note_id) if str(note_id).isdigit() else note_id
                        for note_id in dict.fromkeys([start_note, end_note])
                        if str(note_id).strip()
                    ]
                )
                if action != "confirm"
                and note_candidate is None
                and (start_endpoint is not None or end_endpoint is not None)
                else corrected_connected
            ),
            "staff": corrected_staff,
            "comment": comment,
            "review_target_x_px": str(
                corrected_target.get("x_px", saved.get("review_target_x_px", ""))
            ),
            "review_target_y_px": str(
                corrected_target.get("y_px", saved.get("review_target_y_px", ""))
            ),
            "review_target_pitch": str(
                corrected_target.get("pitch", saved.get("review_target_pitch", ""))
            ),
            "review_start_target_x_px": str(
                start_endpoint.get("x_px", "")
                if start_endpoint is not None
                else ""
            ),
            "review_start_target_y_px": str(
                start_endpoint.get("y_px", "")
                if start_endpoint is not None
                else ""
            ),
            "review_end_target_x_px": str(
                end_endpoint.get("x_px", "") if end_endpoint is not None else ""
            ),
            "review_end_target_y_px": str(
                end_endpoint.get("y_px", "") if end_endpoint is not None else ""
            ),
        }
        job["review_decisions"] = decisions
        st.session_state["raw_alignment_job"] = job
        if current_index < len(queue) - 1:
            st.session_state[pending_key] = queue[current_index + 1]["review_key"]

    with controls_column:
        with st.container(key="review_action_bar"):
            st.markdown("#### ③ 儲存並前往下一筆")
            clicked_action = None
            if st.button(
                "✓ 機器答案正確",
                type="primary" if selected_candidate is None else "secondary",
                use_container_width=True,
                help="保留青色圈與機器時間，然後自動前往下一筆。",
            ):
                clicked_action = "confirm"
            if st.button(
                "✓ 儲存所選音頭更正",
                type="primary" if selected_candidate is not None else "secondary",
                disabled=selected_candidate is None,
                use_container_width=True,
                help="以綠色 C 的音頭更新時間、note ID 與 staff，然後自動前往下一筆。",
            ):
                save_decision("correct", selected_candidate)
                st.rerun()
            if note_candidates and st.button(
                "✓ 儲存圖片點選音頭",
                type="primary" if is_span else "secondary",
                disabled=bool(endpoint_errors)
                or (start_endpoint is None and end_endpoint is None),
                use_container_width=True,
                help="儲存圖片中 S✓／E✓ 對應的時間與 note ID。",
            ):
                save_decision("correct")
                st.rerun()
            if st.button("稍後再看 →", use_container_width=True):
                clicked_action = "skipped"

            with st.expander("其他判定與手動更正", expanded=False):
                st.caption("無法用上方兩個主要答案處理時才使用。")
                if st.button(
                    "儲存手動欄位更正",
                    use_container_width=True,
                    disabled=bool(endpoint_errors),
                    help=(
                        "請先修正開始／結束音符格式"
                        if endpoint_errors
                        else None
                    ),
                ):
                    clicked_action = "correct"
                if st.button("僅保留掃描符號", use_container_width=True):
                    clicked_action = "scan_only"
                if st.button("YOLO class 錯誤", use_container_width=True):
                    clicked_action = "wrong_class"
                special_left, special_right = st.columns(2)
                if special_left.button("BBox 畫錯", use_container_width=True):
                    clicked_action = "bad_bbox"
                if special_right.button("不是符號", use_container_width=True):
                    clicked_action = "not_a_symbol"
                if special_left.button("不確定", use_container_width=True):
                    clicked_action = "uncertain"
                if special_right.button("略過", use_container_width=True):
                    clicked_action = "skipped"
            if clicked_action:
                save_decision(clicked_action)
                st.rerun()

    st.caption(
        f"本筆已儲存：{saved.get('action', '尚未')} · "
        f"本次工作階段共儲存 {len(decisions)} 筆"
    )
    reviewer = st.text_input(
        "Workspace reviewer",
        value="User",
        key="workspace_reviewer",
    )
    checkpoint_left, checkpoint_right = st.columns(2)
    checkpoint_left.download_button(
        "Download review checkpoint JSON",
        json.dumps(
            build_review_checkpoint(
                decisions,
                reviewer,
                alignment_fingerprint=job["fingerprint"],
                score_id=job["report"].get("score_id", ""),
                pipeline_version=job["report"].get("pipeline_version", ""),
            ),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "review_checkpoint.json",
        "application/json",
        disabled=not decisions,
        use_container_width=True,
    )
    uploaded_checkpoint = checkpoint_right.file_uploader(
        "Resume from review checkpoint",
        type=["json"],
        key="workspace_checkpoint_upload",
    )
    st.caption(
        "Checkpoint 已綁定這次 alignment 的圖片、YOLO、XML、BPSD、"
        "notes.json 與設定；不同批次不會被套用。"
    )
    if uploaded_checkpoint is not None and st.button(
        "Load uploaded checkpoint",
        key="workspace_load_checkpoint",
        use_container_width=True,
    ):
        try:
            checkpoint_payload = json.loads(uploaded_checkpoint.getvalue())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            st.error(f"Invalid review checkpoint JSON: {error}")
        else:
            restored, restore_errors = load_review_checkpoint(
                checkpoint_payload,
                detailed_rows,
                expected_fingerprint=job["fingerprint"],
                expected_score_id=job["report"].get("score_id", ""),
                expected_pipeline_version=job["report"].get(
                    "pipeline_version", ""
                ),
            )
            if restore_errors:
                for error in restore_errors:
                    st.error(error)
            else:
                job["review_decisions"] = restored
                st.session_state["raw_alignment_job"] = job
                st.success(f"Restored {len(restored)} review decisions.")
                st.rerun()
    if st.button(
        "Apply all saved workspace decisions",
        disabled=not decisions,
        key="workspace_apply_all",
        use_container_width=True,
    ):
        errors = _apply_and_store_review_outputs(
            job, list(decisions.values()), reviewer
        )
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.success(f"Applied {len(decisions)} saved review decisions.")


def _render_completed_job(job: dict) -> None:
    required_job_fields = {
        "fingerprint",
        "final_bps_csv",
        "yolo_aligned_csv",
        "xml_events_csv",
        "xml_nodes_csv",
        "yolo_xml_timeline_csv",
        "all_information_csv",
        "combined_master_csv",
        "alignment_links_csv",
        "xml_spans_csv",
        "performance_expanded_timeline_csv",
        "detailed_rows",
        "validation_json",
        "output_zip",
        "overlays",
    }
    if not required_job_fields.issubset(job):
        st.info(
            "This session contains output from an older pipeline version. "
            "Click Align all uploaded pages to generate the strict final CSV."
        )
        return
    report = job["report"]
    if report["passed"]:
        st.success("Alignment and validation completed.")
    else:
        st.error("Outputs were created, but validation found errors.")
        for error in report["validation_errors"]:
            st.error(error)
    first, second, third, fourth = st.columns(4)
    first.metric("Pages", report.get("page_count", 1))
    second.metric("YOLO boxes", report["alignment_rows"])
    third.metric("Confirmed values", report.get("confirmed_rows", 0))
    fourth.metric("Needs review", report["yolo_rows_needing_review"])
    st.caption(f"Job ID: {job.get('fingerprint', '')[:12]}")
    if job.get("job_checkpoint_zip"):
        _render_large_download(
            label="Download resumable alignment checkpoint ZIP",
            value=job["job_checkpoint_zip"],
            file_name="alignment_job_checkpoint.zip",
            key="alignment_checkpoint_zip",
        )
    if "clean_reference_pages_used" in report:
        st.caption(
            "Clean repetition PDF geometry used on "
            f"{report['clean_reference_pages_used']} of "
            f"{report.get('page_count', 1)} pages."
        )
    for warning in report["warnings"]:
        st.warning(warning)

    _render_review_workspace(job)
    _render_review_outputs(job)

    st.subheader("Review images")
    overlays = job["overlays"]
    if overlays:
        review_pages = group_review_overlays(overlays, job["detailed_rows"])
        if review_pages:
            selected_page = st.selectbox(
                "Score image",
                list(review_pages),
                format_func=lambda page_id: (
                    f"{page_id} "
                    f"({review_pages[page_id]['needs_review']} needs review)"
                ),
                key="review_page_select",
            )
            selected_page_data = review_pages[selected_page]
            overview_tab, class_tab = st.tabs(
                ["Overview", "One class at a time"]
            )
            with overview_tab:
                review_overlay = selected_page_data["review_overlay"]
                if review_overlay is not None:
                    selected_overlay = _asset_bytes(review_overlay)
                    st.image(
                        selected_overlay,
                        caption=f"{selected_page}: needs review",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download review overlay at full resolution",
                        selected_overlay,
                        f"{selected_page}_review_overlay.png",
                        "image/png",
                        use_container_width=True,
                    )
                else:
                    st.info("No review overlay was generated for this page.")
            with class_tab:
                class_overlays = selected_page_data["classes"]
                if class_overlays:
                    selected_class = st.selectbox(
                        "YOLO class (all symbols on this page)",
                        sorted(class_overlays),
                        key=f"class_overlay_select_{selected_page}",
                    )
                    selected_overlay = _asset_bytes(class_overlays[selected_class])
                    st.image(
                        selected_overlay,
                        caption=f"{selected_page} — class: {selected_class}",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download this class image at full resolution",
                        selected_overlay,
                        f"{selected_page}_class_{selected_class}.png",
                        "image/png",
                        use_container_width=True,
                    )
                else:
                    st.info("No per-class review images were generated for this page.")
        else:
            st.info(
                "This session checkpoint predates page-grouped review images. "
                "Run alignment once with the current version to generate them."
            )
    else:
        st.info("No review overlay was generated for this input.")
    st.caption(
        "Overlay colors: green = direct match; blue = inferred candidate; "
        "orange = needs review; gray/red = unresolved. Class images show "
        "YOLO ID, written measure range, BPSD time range, and status."
    )

    with st.expander("Rows needing review"):
        review_rows = [
            {
                "YOLO line": row.get("txt_line", ""),
                "page": row.get("page_id", ""),
                "class": row.get("class", ""),
                "written start": row.get(
                    "start_xml_measure", row.get("xml_measure", "")
                ),
                "written end": row.get(
                    "end_xml_measure", row.get("xml_measure", "")
                ),
                "BPSD start": row.get("start_meas", ""),
                "BPSD end": row.get("end_meas", ""),
                "match source": row.get("match_source", ""),
                "confidence": row.get("confidence", ""),
                "status": row.get("status", ""),
            }
            for row in job["detailed_rows"]
            if row.get("status") != "matched"
        ]
        if review_rows:
            st.dataframe(review_rows, use_container_width=True)
        else:
            st.success("No machine-generated rows are marked for review.")

    with st.expander("Human review and corrections", expanded=False):
        st.write(
            "Confirm keeps the machine values. Correct applies your edited values and "
            "sets human_corrected=1. Reject clears uncertain semantic fields. "
            "Leave any unknown value blank."
        )
        reviewer = st.text_input(
            "Reviewer",
            value="User",
            key="human_review_reviewer",
        )
        include_matched = st.checkbox(
            "Include machine-matched rows for spot checking",
            value=False,
            key="human_review_include_matched",
        )
        editor_source = build_editor_rows(
            job["detailed_rows"], include_matched=include_matched
        )
        if editor_source:
            edited = st.data_editor(
                editor_source,
                key=f"human_review_editor_{int(include_matched)}",
                hide_index=True,
                use_container_width=True,
                disabled=["page_id", "yolo_line", "class", "machine_status"],
                column_config={
                    "action": st.column_config.SelectboxColumn(
                        "Action",
                        options=list(REVIEW_ACTIONS),
                        required=True,
                    ),
                    "page_id": "Page",
                    "yolo_line": "YOLO line",
                    "corrected_class": "Corrected class",
                    "corrected_class_id": "Corrected class ID",
                    "machine_status": "Machine status",
                    "start_meas": "Start time",
                    "end_meas": "End time",
                    "start_note": "Start note ID",
                    "end_note": "End note ID",
                    "connected_note": "Connected note IDs",
                    "staff": "Staff",
                    "comment": "Comment",
                },
            )
            if st.button(
                "Apply reviewed decisions",
                type="primary",
                key="apply_human_review_decisions",
                use_container_width=True,
            ):
                edited_records = (
                    edited.to_dict("records")
                    if hasattr(edited, "to_dict")
                    else list(edited)
                )
                review_errors = _apply_and_store_review_outputs(
                    job,
                    edited_records,
                    reviewer,
                )
                if review_errors:
                    for error in review_errors:
                        st.error(error)
                else:
                    st.success(
                        f"Applied {job['human_review_outputs']['decisions']} "
                        "reviewed decisions."
                    )
        else:
            st.success("No rows are currently available for review.")

    st.subheader("完整資料輸出")
    st.caption(
        "XML 只匯出一次，不會因為上傳多張樂譜圖片而重複。"
    )
    yolo_download, xml_download, timeline_download = st.columns(3)
    yolo_download.download_button(
        "1. YOLO Align CSV",
        _asset_bytes(job["yolo_aligned_csv"]),
        "yolo_aligned.csv",
        "text/csv",
        use_container_width=True,
    )
    xml_download.download_button(
        "2. XML Events CSV",
        _asset_bytes(job["xml_events_csv"]),
        "xml_events.csv",
        "text/csv",
        use_container_width=True,
    )
    timeline_download.download_button(
        "3. XML + YOLO 時間排序 CSV",
        _asset_bytes(job["yolo_xml_timeline_csv"]),
        "yolo_xml_timeline.csv",
        "text/csv",
        use_container_width=True,
    )
    st.caption(
        "Timeline 保留每個 YOLO bbox 與每個 XML event，並依 start/end time 排序。"
    )
    with st.expander("更多完整／原始 XML 輸出", expanded=False):
        raw_xml, all_info, combined, links = st.columns(4)
        raw_xml.download_button(
            "XML Nodes（原始節點）",
            _asset_bytes(job["xml_nodes_csv"]),
            "xml_nodes.csv",
            "text/csv",
            use_container_width=True,
        )
        all_info.download_button(
            "All Information",
            _asset_bytes(job["all_information_csv"]),
            "all_information.csv",
            "text/csv",
            use_container_width=True,
        )
        combined.download_button(
            "Combined Master",
            _asset_bytes(job["combined_master_csv"]),
            "combined_master.csv",
            "text/csv",
            use_container_width=True,
        )
        links.download_button(
            "Alignment Links",
            _asset_bytes(job["alignment_links_csv"]),
            "alignment_links.csv",
            "text/csv",
            use_container_width=True,
        )
        spans, performance = st.columns(2)
        spans.download_button(
            "XML Spans（跨頁起訖）",
            _asset_bytes(job["xml_spans_csv"]),
            "xml_spans.csv",
            "text/csv",
            use_container_width=True,
        )
        performance.download_button(
            "Performance-expanded Timeline",
            _asset_bytes(job["performance_expanded_timeline_csv"]),
            "performance_expanded_timeline.csv",
            "text/csv",
            use_container_width=True,
        )
        st.caption(
            "XML Nodes 保留 tag、attributes、text、XPath 與父子關係；"
            "All Information 同時包含 YOLO、XML events 與 XML nodes。"
        )

    st.subheader("BPS-OMR Final CSV")
    st.download_button(
        "Download BPS-OMR final CSV",
        _asset_bytes(job["final_bps_csv"]),
        "bps_omr_final.csv",
        "text/csv",
        use_container_width=True,
    )
    st.caption(
        "Exactly one row per YOLO bounding box. Columns follow BPS-OMR "
        "annotations plus human_corrected. Uncertain or unavailable values are blank."
    )

    with st.expander("Validation details"):
        st.download_button(
            "Validation JSON",
            _asset_bytes(job["validation_json"]),
            "validation_report.json",
            "application/json",
            use_container_width=True,
        )
    _render_large_download(
        label="Final CSV + review images ZIP",
        value=job["output_zip"],
        file_name="bpsd_alignment_outputs.zip",
        key="final_outputs_zip",
    )
    with st.expander("Alignment status counts"):
        st.json(report.get("yolo_status_counts", {}))


st.set_page_config(page_title="BPSD XML–YOLO Aligner", page_icon="🎼", layout="wide")
_require_access_token()
_initialize_job_store()
st.title("BPSD XML–YOLO Aligner")
st.caption("Upload score pages and export a strict BPS-OMR bounding-box CSV")

align_tab, inspect_tab, combine_tab, guide_tab = st.tabs(
    ["Run alignment", "Inspect CSV", "Advanced combine", "CLI guide"]
)

with align_tab:
    _render_background_job_status()
    st.subheader("Upload one or more score pages")
    st.write(
        "Upload matching image/TXT pairs. One repetition MusicXML, optional unfolded "
        "MusicXML, BPSD note annotation, and notes.json are shared by every page."
    )
    st.info(
        "Version guide: upload the written-score XML from score_xml_repetitions "
        "in Repetition MusicXML. Upload the performance-order XML from "
        "score_xml_unfolded in Unfolded MusicXML. Do not upload .sib files. "
        "A clean score used for image geometry must be the repetition version, "
        "not the unfolded version."
    )
    left, right = st.columns(2)
    with left:
        image_uploads = st.file_uploader(
            "Score images *",
            type=["jpg", "jpeg", "png"],
            key="raw_images",
            accept_multiple_files=True,
            help="Use scanned score pages, normally from finished/Xia/images.",
        )
        yolo_uploads = st.file_uploader(
            "YOLO TXT files *",
            type=["txt"],
            key="raw_yolos",
            accept_multiple_files=True,
            help="Use TXT files from finished/Xia/labels with the same filename stems as the images.",
        )
        notes_upload = st.file_uploader(
            "YOLO class map (notes.json) *", type=["json"], key="raw_notes_json"
        )
    with right:
        bps_upload = st.file_uploader(
            "BPSD note annotations (ann_score_note CSV) *",
            type=["csv"],
            key="raw_bps_notes",
        )
        xml_upload = st.file_uploader(
            "Repetition MusicXML — score_xml_repetitions/*.xml *",
            type=["xml", "musicxml"],
            key="raw_xml",
            help=(
                "Required. Use the written/printed score XML from "
                "0_RawData/score_xml_repetitions. Do not use unfolded XML here."
            ),
        )
        unfolded_upload = st.file_uploader(
            "Unfolded MusicXML — score_xml_unfolded/*.xml (recommended)",
            type=["xml", "musicxml"],
            key="raw_unfolded_xml",
            help=(
                "Use the repeat-expanded XML from 0_RawData/score_xml_unfolded. "
                "This supplies performance order; it is not used as the scan layout."
            ),
        )
        clean_pdf_upload = st.file_uploader(
            "Clean repetition PDF — score_pdf_repetitions/*.pdf (recommended)",
            type=["pdf"],
            key="raw_clean_repetition_pdf",
            help=(
                "Upload the one whole-score PDF matching Repetition MusicXML. "
                "It improves system, barline, notehead, slur, and tie geometry. "
                "Do not use score_pdf_unfolded."
            ),
        )
        resume_job_upload = st.file_uploader(
            "Resume alignment job checkpoint ZIP (optional)",
            type=["zip"],
            key="raw_alignment_checkpoint_zip",
            help=(
                "Select the same score inputs, then upload a checkpoint ZIP from "
                "a previous run. The input fingerprint must match."
            ),
        )

    settings_left, settings_middle, settings_right = st.columns(3)
    score_id = settings_left.text_input("Score ID", value="uploaded-score")
    first_page = settings_middle.number_input(
        "First MusicXML page", min_value=1, value=1, step=1
    )
    infer_fingerings = settings_right.checkbox("Infer fingering candidates", value=True)
    infer_pages = st.checkbox(
        "Infer MusicXML page from the final number in each filename",
        value=True,
        help=(
            "Example: Beethoven_Op110-01-04.jpeg and .txt use MusicXML page 4. "
            "Turn this off to assign consecutive pages starting from First MusicXML page."
        ),
    )
    run_in_background = st.checkbox(
        "Run alignment in a background worker",
        value=True,
        help=(
            "Recommended for multi-page scores. The job continues if this browser "
            "tab closes; return and refresh the saved job status to load outputs."
        ),
    )

    page_pairs = []
    pairing_error = ""
    if image_uploads or yolo_uploads:
        try:
            page_pairs = pair_page_uploads(
                image_uploads,
                yolo_uploads,
                first_page=int(first_page),
                infer_page_from_filename=infer_pages,
            )
        except ValueError as error:
            pairing_error = str(error)
            st.error(pairing_error)
        else:
            pairing_rows = [
                    {
                        "page stem": pair["stem"],
                        "MusicXML page": pair["page_number"],
                        "Scan system starts": ", ".join(
                            str(value)
                            for value in pair.get("system_start_measures", [])
                        ),
                        "Scan page end": pair.get("page_end_measure"),
                        "image": pair["image"].name,
                        "YOLO TXT": pair["yolo"].name,
                    }
                    for pair in page_pairs
                ]
            edited_pairing = st.data_editor(
                pairing_rows,
                key="page_pairing_editor",
                use_container_width=True,
                hide_index=True,
                disabled=["page stem", "image", "YOLO TXT"],
                column_config={
                    "MusicXML page": st.column_config.NumberColumn(
                        "MusicXML page", min_value=1, step=1, required=True
                    ),
                    "Scan system starts": st.column_config.TextColumn(
                        "掃描譜每行起始小節",
                        help=(
                            "只有掃描譜與 MusicXML 換行不同時需要填。依畫面由上到下，"
                            "輸入每個 system 的第一個印刷小節號，例如："
                            "198, 202, 206, 211, 223, 235"
                        ),
                    ),
                    "Scan page end": st.column_config.NumberColumn(
                        "掃描譜本頁最後小節",
                        min_value=1,
                        step=1,
                        help=(
                            "可留空；若下一張掃描頁也有填 system 起始小節，系統會自動推算。"
                            "只有本頁最後小節與 MusicXML 分頁也不同時才需手動填。"
                        ),
                    ),
                },
            )
            st.caption(
                "掃描譜與 MusicXML 換行相同時，兩個掃描譜欄位都留空。若不同，"
                "請在「掃描譜每行起始小節」依上到下輸入印刷小節號；這些數字會決定"
                "畫面位置，音高、時間與 note ID 仍取自同號 MusicXML 小節。"
            )
            try:
                edited_pairing_rows = (
                    edited_pairing.to_dict("records")
                    if hasattr(edited_pairing, "to_dict")
                    else list(edited_pairing)
                )
                page_pairs = apply_page_mapping_edits(
                    page_pairs, edited_pairing_rows
                )
            except ValueError as error:
                pairing_error = str(error)
                st.error(pairing_error)

    required_common = [xml_upload, bps_upload, notes_upload]
    start = st.button(
        "Align all uploaded pages",
        type="primary",
        disabled=not page_pairs or bool(pairing_error) or not all(required_common),
        use_container_width=True,
    )
    if start:
        page_files = [item for pair in page_pairs for item in (pair["image"], pair["yolo"])]
        labels_and_uploads = [
            *[(f"Page file {uploaded.name}", uploaded) for uploaded in page_files],
            ("MusicXML", xml_upload),
            ("BPSD notes", bps_upload),
            ("notes.json", notes_upload),
        ]
        all_uploads = [
            *(uploaded for _label, uploaded in labels_and_uploads),
            unfolded_upload,
            clean_pdf_upload,
            resume_job_upload,
        ]
        upload_batch_valid = True
        try:
            upload_totals = validate_upload_batch(all_uploads)
            validate_page_count(len(page_pairs))
        except ValueError as error:
            upload_batch_valid = False
            st.error(str(error))
        else:
            st.caption(
                f"Upload batch: {upload_totals['files']} files, "
                f"{upload_totals['bytes'] / (1024 ** 2):.1f} MB"
            )
        if upload_batch_valid and all(
            _valid_upload(uploaded, label) for label, uploaded in labels_and_uploads
        ):
            if clean_pdf_upload is not None and not _valid_upload(
                clean_pdf_upload, "Clean repetition PDF"
            ):
                st.stop()
            options = {
                "score_id": score_id,
                "page_numbers": [pair["page_number"] for pair in page_pairs],
                "scan_system_starts": [
                    pair.get("system_start_measures", []) for pair in page_pairs
                ],
                "scan_page_ends": [pair.get("page_end_measure") for pair in page_pairs],
                "infer_fingerings": infer_fingerings,
                "pipeline_version": PIPELINE_VERSION,
            }
            fingerprint = _upload_fingerprint(
                [
                    *page_files,
                    *required_common,
                    unfolded_upload,
                    clean_pdf_upload,
                ],
                options,
            )
            cached = st.session_state.get("raw_alignment_job")
            if cached and cached.get("fingerprint") == fingerprint:
                st.info("The inputs match the completed session checkpoint; reusing outputs.")
            elif run_in_background:
                try:
                    background_dir = _queue_background_job(
                        fingerprint=fingerprint,
                        page_pairs=page_pairs,
                        score_id=score_id,
                        infer_fingerings=infer_fingerings,
                        xml_upload=xml_upload,
                        bps_upload=bps_upload,
                        notes_upload=notes_upload,
                        unfolded_upload=unfolded_upload,
                        clean_pdf_upload=clean_pdf_upload,
                        resume_job_upload=resume_job_upload,
                    )
                except Exception as error:
                    st.error(f"Unable to queue background job: {type(error).__name__}: {error}")
                else:
                    st.session_state["background_alignment_job"] = {
                        "fingerprint": fingerprint,
                        "job_dir": str(background_dir),
                    }
                    st.success(
                        "Background job started. Progress appears at the top of "
                        "this tab and refreshes automatically every 2 seconds."
                    )
            else:
                with st.status("Running alignment", expanded=True) as status:
                    progress = st.progress(0, text="Preparing uploads")
                    persistent_job_dir = job_directory(fingerprint)
                    with nullcontext(persistent_job_dir) as temporary_path:
                        lease_path = None
                        try:
                            lease_path = acquire_job_lease(temporary_path)
                        except RuntimeError as error:
                            status.update(label="Alignment is already busy", state="error")
                            st.error(str(error))
                            st.stop()
                        if resume_job_upload is not None:
                            try:
                                restored_files = restore_job_checkpoint_archive(
                                    resume_job_upload.getvalue(),
                                    temporary_path,
                                    expected_fingerprint=fingerprint,
                                    expected_pipeline_version=PIPELINE_VERSION,
                                )
                            except Exception:
                                release_job_lease(lease_path)
                                lease_path = None
                                raise
                            st.write(
                                f"Restored {restored_files} checkpoint files for "
                                f"job {fingerprint[:12]}."
                            )
                        try:
                            input_dir = temporary_path / "inputs"
                            output_dir = temporary_path / "outputs"
                            input_dir.mkdir(parents=True, exist_ok=True)
                            xml_path = _save_upload(xml_upload, input_dir, "score.xml")
                            bps_path = _save_upload(
                                bps_upload, input_dir, "ann_score_note.csv"
                            )
                            notes_path = _save_upload(
                                notes_upload, input_dir, "notes.json"
                            )
                            unfolded_path = (
                                _save_upload(
                                    unfolded_upload, input_dir, "score_unfolded.xml"
                                )
                                if unfolded_upload is not None
                                else None
                            )
                            clean_pdf_path = (
                                _save_upload(
                                    clean_pdf_upload,
                                    input_dir,
                                    "clean_repetition.pdf",
                                )
                                if clean_pdf_upload is not None
                                else None
                            )
                            write_job_manifest(
                                temporary_path,
                                fingerprint=fingerprint,
                                pipeline_version=PIPELINE_VERSION,
                                inputs=[
                                    {
                                        "name": Path(uploaded.name).name,
                                        "size": int(uploaded.size),
                                    }
                                    for uploaded in all_uploads
                                    if uploaded is not None
                                ],
                            )
                            completed_page_count = 0
                            write_job_status(
                                temporary_path,
                                state="running",
                                stage="uploads_saved",
                                total_pages=len(page_pairs),
                                message="Uploaded inputs are saved and validated.",
                            )
                        except Exception:
                            release_job_lease(lease_path)
                            lease_path = None
                            raise
                        try:
                            if clean_pdf_path is not None:
                                clean_page_count = pdf_page_count(clean_pdf_path)
                                requested_page = max(
                                    pair["page_number"] for pair in page_pairs
                                )
                                if requested_page > clean_page_count:
                                    raise ValueError(
                                        f"Clean repetition PDF has {clean_page_count} "
                                        f"pages, but page {requested_page} is required. "
                                        "Check that the PDF and MusicXML are the same "
                                        "repetition-layout score."
                                    )
                                st.write(
                                    f"Clean repetition PDF: {clean_page_count} pages; "
                                    "requested pages are in range."
                                )
                            write_job_status(
                                temporary_path,
                                state="running",
                                stage="shared_score",
                                total_pages=len(page_pairs),
                                message="Preparing whole-score XML and repeat mapping.",
                            )
                            st.write("Shared score 0/2: preparing whole-score sources")
                            prepared_score = prepare_score_sources(
                                xml_path=xml_path,
                                bps_notes_path=bps_path,
                                output_dir=output_dir / "shared_score",
                                score_id=score_id,
                                unfolded_xml_path=unfolded_path,
                                progress_callback=lambda step, total, message: st.write(
                                    f"Shared score {step}/{total}: {message}"
                                ),
                            )
                            page_reports = []
                            final_entries = []
                            detailed_rows = []
                            yolo_entries = []
                            xml_event_rows = []
                            xml_node_rows = []
                            overlays = {}
                            page_images = {}
                            total_steps = len(page_pairs) * 6
                            page_checkpoints = st.session_state.setdefault(
                                "raw_alignment_page_checkpoints", {}
                            )
                            for page_index, pair in enumerate(page_pairs):
                                checkpoint_key = (
                                    f"{fingerprint}:{pair['stem']}:{pair['page_number']}"
                                )
                                checkpoint = page_checkpoints.get(checkpoint_key)
                                if checkpoint is None:
                                    disk_checkpoint = load_page_checkpoint(
                                        temporary_path,
                                        fingerprint=fingerprint,
                                        pipeline_version=PIPELINE_VERSION,
                                        page_id=pair["stem"],
                                        page_number=pair["page_number"],
                                    )
                                    if disk_checkpoint is not None:
                                        stored_image = disk_checkpoint["page_image"]
                                        checkpoint = _page_checkpoint_artifacts(
                                            disk_checkpoint["report"],
                                            page_id=pair["stem"],
                                            page_number=pair["page_number"],
                                            page_image=stored_image,
                                        )
                                        page_checkpoints[checkpoint_key] = checkpoint
                                if checkpoint:
                                    page_reports.append(checkpoint["report"])
                                    for row_index, row in enumerate(
                                        checkpoint["final_rows"]
                                    ):
                                        final_entries.append(
                                            (pair["page_number"], row_index, row)
                                        )
                                    detailed_rows.extend(checkpoint["detailed_rows"])
                                    for row_index, row in enumerate(
                                        checkpoint.get("yolo_rows", [])
                                    ):
                                        yolo_entries.append(
                                            (pair["page_number"], row_index, row)
                                        )
                                    if not xml_event_rows and checkpoint.get(
                                        "xml_event_rows"
                                    ):
                                        xml_event_rows = checkpoint["xml_event_rows"]
                                    if not xml_node_rows and checkpoint.get(
                                        "xml_node_rows"
                                    ):
                                        xml_node_rows = checkpoint["xml_node_rows"]
                                    overlays.update(checkpoint["overlays"])
                                    if checkpoint.get("page_image"):
                                        page_images[pair["stem"]] = checkpoint[
                                            "page_image"
                                        ]
                                    completed_steps = (page_index + 1) * 6
                                    progress.progress(
                                        completed_steps / total_steps,
                                        text=(
                                            f"{completed_steps}/{total_steps} "
                                            f"{pair['stem']}: resumed page checkpoint"
                                        ),
                                    )
                                    st.write(
                                        f"{completed_steps}/{total_steps} "
                                        f"{pair['stem']}: resumed page checkpoint"
                                    )
                                    completed_page_count += 1
                                    write_job_status(
                                        temporary_path,
                                        state="running",
                                        stage="page_alignment",
                                        completed_pages=completed_page_count,
                                        total_pages=len(page_pairs),
                                        message=(
                                            f"Resumed page checkpoint: {pair['stem']}"
                                        ),
                                    )
                                    continue
                                page_input_dir = input_dir / pair["stem"]
                                page_input_dir.mkdir(parents=True, exist_ok=True)
                                image_path = _save_upload(
                                    pair["image"], page_input_dir, "page.png"
                                )
                                page_image = str(image_path)
                                page_images[pair["stem"]] = page_image
                                yolo_path = _save_upload(
                                    pair["yolo"], page_input_dir, "page.txt"
                                )
                                clean_image_path = None
                                if clean_pdf_path is not None:
                                    clean_image_path = render_pdf_page(
                                        clean_pdf_path,
                                        pair["page_number"],
                                        input_dir
                                        / "clean_pages"
                                        / f"page-{pair['page_number']:04d}.png",
                                    )
                                    st.write(
                                        f"{pair['stem']}: rendered clean PDF page "
                                        f"{pair['page_number']} (checkpoint saved)."
                                    )

                                def update_progress(
                                    step: int,
                                    _page_total: int,
                                    message: str,
                                    *,
                                    page_index: int = page_index,
                                    stem: str = pair["stem"],
                                ) -> None:
                                    overall_step = page_index * 6 + step
                                    progress.progress(
                                        overall_step / total_steps,
                                        text=(
                                            f"{overall_step}/{total_steps} "
                                            f"{stem}: {message}"
                                        ),
                                    )
                                    st.write(
                                        f"{overall_step}/{total_steps} {stem}: {message}"
                                    )

                                page_report = run_uploaded_alignment(
                                    image_path=image_path,
                                    yolo_path=yolo_path,
                                    xml_path=xml_path,
                                    bps_notes_path=bps_path,
                                    notes_json_path=notes_path,
                                    unfolded_xml_path=unfolded_path,
                                    clean_image_path=clean_image_path,
                                    output_dir=output_dir / "pages" / pair["stem"],
                                    page_number=pair["page_number"],
                                    score_id=score_id,
                                    infer_fingerings=infer_fingerings,
                                    progress_callback=update_progress,
                                    prepared_score=prepared_score,
                                    build_complete_exports=False,
                                    system_start_measures=pair.get(
                                        "system_start_measures"
                                    ),
                                    page_end_measure=pair.get("page_end_measure"),
                                )
                                page_reports.append(page_report)
                                page_outputs = {
                                    name: Path(path)
                                    for name, path in page_report["outputs"].items()
                                }
                                page_final_rows = []
                                with page_outputs["final_bps_csv"].open(
                                    newline="", encoding="utf-8-sig"
                                ) as file:
                                    for row_index, row in enumerate(csv.DictReader(file)):
                                        page_final_rows.append(row)
                                        final_entries.append(
                                            (pair["page_number"], row_index, row)
                                        )
                                page_detailed = read_csv_bytes(
                                    page_outputs["detailed_csv"].read_bytes()
                                )[1]
                                for row in page_detailed:
                                    row["page_id"] = pair["stem"]
                                    row["page_number"] = str(pair["page_number"])
                                detailed_rows.extend(page_detailed)
                                page_yolo_rows = read_csv_bytes(
                                    page_outputs["yolo_aligned_csv"].read_bytes()
                                )[1]
                                for row_index, row in enumerate(page_yolo_rows):
                                    yolo_entries.append(
                                        (pair["page_number"], row_index, row)
                                    )
                                checkpoint_xml_events = []
                                checkpoint_xml_nodes = []
                                if not xml_event_rows:
                                    checkpoint_xml_events = read_csv_bytes(
                                        page_outputs["xml_events_csv"].read_bytes()
                                    )[1]
                                    xml_event_rows = checkpoint_xml_events
                                if not xml_node_rows:
                                    checkpoint_xml_nodes = read_csv_bytes(
                                        page_outputs["xml_nodes_csv"].read_bytes()
                                    )[1]
                                    xml_node_rows = checkpoint_xml_nodes
                                page_overlays = {}
                                for name, path in page_outputs.items():
                                    if not name.endswith("overlay") or not path.is_file():
                                        continue
                                    key = (
                                        f"class_{pair['stem']}__{name.removeprefix('class_')}"
                                        if name.startswith("class_")
                                        else f"{pair['stem']}__{name}"
                                    )
                                    page_overlays[key] = str(path)
                                overlays.update(page_overlays)
                                page_checkpoints[checkpoint_key] = {
                                    "report": page_report,
                                    "final_rows": page_final_rows,
                                    "detailed_rows": page_detailed,
                                    "yolo_rows": page_yolo_rows,
                                    "xml_event_rows": checkpoint_xml_events,
                                    "xml_node_rows": checkpoint_xml_nodes,
                                    "overlays": page_overlays,
                                    "page_image": page_image,
                                }
                                write_page_checkpoint(
                                    temporary_path,
                                    fingerprint=fingerprint,
                                    pipeline_version=PIPELINE_VERSION,
                                    page_id=pair["stem"],
                                    page_number=pair["page_number"],
                                    report=page_report,
                                    page_image=image_path,
                                )
                                completed_page_count += 1
                                write_job_status(
                                    temporary_path,
                                    state="running",
                                    stage="page_alignment",
                                    completed_pages=completed_page_count,
                                    total_pages=len(page_pairs),
                                    message=f"Completed page: {pair['stem']}",
                                )

                            def final_sort_key(entry):
                                page, row_index, row = entry
                                try:
                                    time = float(row.get("start_meas", ""))
                                except (TypeError, ValueError):
                                    time = float("inf")
                                return time, page, row_index

                            final_rows = [
                                entry[2]
                                for entry in sorted(final_entries, key=final_sort_key)
                            ]
                            yolo_rows = [
                                entry[2]
                                for entry in sorted(yolo_entries, key=final_sort_key)
                            ]
                            final_path = output_dir / "bps_omr_final.csv"
                            atomic_write_csv(final_path, FINAL_BPS_FIELDS, final_rows)
                            complete_exports = build_batch_information_outputs(
                                yolo_rows=yolo_rows,
                                xml_events=xml_event_rows,
                                xml_nodes=xml_node_rows,
                                output_dir=output_dir / "complete_exports",
                            )
                            errors = [
                                error
                                for page_report in page_reports
                                for error in page_report["validation_errors"]
                            ]
                            errors.extend(complete_exports["validation_errors"])
                            warnings = [
                                warning
                                for page_report in page_reports
                                for warning in page_report["warnings"]
                            ]
                            status_counts = Counter(
                                row.get("status", "") for row in detailed_rows
                            )
                            report = {
                                "pipeline_version": PIPELINE_VERSION,
                                "score_id": score_id,
                                "page_count": len(page_reports),
                                "pages": [pair["page_number"] for pair in page_pairs],
                                "alignment_rows": len(final_rows),
                                "xml_event_rows": complete_exports["xml_event_rows"],
                                "xml_node_rows": complete_exports["xml_node_rows"],
                                "timeline_rows": complete_exports["timeline_rows"],
                                "all_information_rows": complete_exports[
                                    "all_information_rows"
                                ],
                                "xml_span_rows": complete_exports["xml_span_rows"],
                                "performance_expanded_rows": complete_exports[
                                    "performance_expanded_rows"
                                ],
                                "confirmed_rows": status_counts.get("matched", 0),
                                "human_corrected_rows": sum(
                                    row["human_corrected"] == "1" for row in final_rows
                                ),
                                "yolo_rows_needing_review": len(final_rows)
                                - status_counts.get("matched", 0),
                                "yolo_status_counts": dict(status_counts),
                                "class_overlay_count": sum(
                                    page_report.get("class_overlay_count", 0)
                                    for page_report in page_reports
                                ),
                                "clean_reference_pages_used": sum(
                                    bool(
                                        page_report.get("clean_reference", {}).get(
                                            "used"
                                        )
                                    )
                                    for page_report in page_reports
                                ),
                                "warnings": list(dict.fromkeys(warnings)),
                                "validation_errors": errors,
                                "passed": not errors,
                                "final_csv_fields": FINAL_BPS_FIELDS,
                                "empty_value_policy": (
                                    "uncertain, unavailable, and not-applicable values are blank"
                                ),
                            }
                            report_path = output_dir / "validation_report.json"
                            atomic_write_json(report_path, report)
                            zip_path = output_dir / "all_outputs.zip"
                            with zipfile.ZipFile(
                                zip_path, "w", compression=zipfile.ZIP_DEFLATED
                            ) as archive:
                                archive.write(final_path, arcname=final_path.name)
                                archive.write(report_path, arcname=report_path.name)
                                for export_name, export_path in complete_exports[
                                    "outputs"
                                ].items():
                                    archive.write(
                                        export_path,
                                        arcname=f"complete_exports/{export_path.name}",
                                    )
                                for name, data in overlays.items():
                                    if isinstance(data, (str, Path)):
                                        archive.write(
                                            data, arcname=f"review_images/{name}.png"
                                        )
                                    else:
                                        archive.writestr(
                                            f"review_images/{name}.png", data
                                        )
                            validation_json = report_path.read_bytes()
                            write_job_status(
                                temporary_path,
                                state="completed",
                                stage="outputs_ready",
                                completed_pages=completed_page_count,
                                total_pages=len(page_pairs),
                                message=(
                                    "Alignment outputs passed validation."
                                    if report["passed"]
                                    else "Outputs are ready with validation errors."
                                ),
                            )
                            job_checkpoint_path = build_job_checkpoint_archive(
                                temporary_path,
                                temporary_path / "alignment_job_checkpoint.zip",
                            )
                            st.session_state["raw_alignment_job"] = {
                                "fingerprint": fingerprint,
                                "report": report,
                                "final_bps_csv": str(final_path),
                                **{
                                    name: str(path)
                                    for name, path in complete_exports["outputs"].items()
                                },
                                "detailed_rows": detailed_rows,
                                "validation_json": validation_json,
                                "output_zip": str(zip_path),
                                "job_checkpoint_zip": str(job_checkpoint_path),
                                "overlays": overlays,
                                "page_images": page_images,
                                "review_decisions": {},
                                "bps_note_count": prepared_score.get(
                                    "bps_note_count"
                                ),
                                "bps_note_ids": prepared_score.get("bps_note_ids"),
                                "class_map": {
                                    str(item["id"]): str(item["name"])
                                    for item in json.loads(
                                        notes_upload.getvalue()
                                    ).get("categories", [])
                                },
                            }
                        except Exception as error:
                            write_job_status(
                                temporary_path,
                                state="failed",
                                stage="failed",
                                completed_pages=completed_page_count,
                                total_pages=len(page_pairs),
                                message="Alignment stopped before all outputs were ready.",
                                error=f"{type(error).__name__}: {error}",
                            )
                            status.update(label="Alignment failed", state="error")
                            st.error(f"{type(error).__name__}: {error}")
                        else:
                            status.update(
                                label=(
                                    "Alignment completed"
                                    if report["passed"]
                                    else "Alignment completed with validation errors"
                                ),
                                state="complete" if report["passed"] else "error",
                            )
                        finally:
                            release_job_lease(lease_path)
    if "raw_alignment_job" in st.session_state:
        _render_completed_job(st.session_state["raw_alignment_job"])

with inspect_tab:
    st.subheader("Inspect an existing combined CSV")
    uploaded_csv = st.file_uploader("Combined master CSV", type=["csv"], key="inspect_csv")
    if _valid_upload(uploaded_csv, "CSV"):
        try:
            fields, rows = read_csv_bytes(uploaded_csv.getvalue())
        except (UnicodeDecodeError, csv.Error) as error:
            st.error(f"Unable to read CSV: {error}")
        else:
            summary = summarize_rows(rows)
            _summary_cards(summary)
            st.write("Row origins", summary["origins"])
            st.write("Alignment statuses", summary["statuses"])
            st.dataframe(rows[:200], use_container_width=True)
            st.caption(f"Showing the first {min(200, len(rows)):,} rows and {len(fields):,} columns.")

with combine_tab:
    st.subheader("Combine existing stage outputs")
    yolo_master = st.file_uploader("YOLO/BPS master CSV", type=["csv"], key="advanced_yolo")
    xml_events = st.file_uploader("XML events CSV", type=["csv"], key="advanced_xml")
    remove_paths = st.checkbox("Replace local source paths with filenames", value=True)
    if st.button(
        "Combine prepared CSV files",
        disabled=not (yolo_master and xml_events),
        key="advanced_combine_button",
    ):
        if _valid_upload(yolo_master, "YOLO master") and _valid_upload(
            xml_events, "XML events"
        ):
            with st.status("Combining prepared CSV files", expanded=True) as status:
                with tempfile.TemporaryDirectory(prefix="bpsd-web-combine-") as temporary:
                    temporary_path = Path(temporary)
                    yolo_path = _save_upload(yolo_master, temporary_path, "yolo_master.csv")
                    xml_path = _save_upload(xml_events, temporary_path, "xml_events.csv")
                    output_path = temporary_path / "output"
                    try:
                        report = combine_dataset(yolo_path, xml_path, output_path)
                    except Exception as error:
                        status.update(label="Combination failed", state="error")
                        st.error(f"{type(error).__name__}: {error}")
                    else:
                        combined = (output_path / "combined_master.csv").read_bytes()
                        if remove_paths:
                            combined = sanitize_path_columns(combined)
                        st.session_state["advanced_combined"] = {
                            "report": report,
                            "csv": combined,
                            "links": (output_path / "alignment_links.csv").read_bytes(),
                            "validation": json.dumps(
                                report, ensure_ascii=False, indent=2
                            ).encode("utf-8"),
                        }
                        status.update(
                            label="Combination completed",
                            state="complete" if report["passed"] else "error",
                        )
    if "advanced_combined" in st.session_state:
        advanced = st.session_state["advanced_combined"]
        if advanced["report"]["passed"]:
            st.success("Prepared CSV files combined successfully.")
        else:
            st.error("Combination validation found errors.")
        downloads = st.columns(3)
        downloads[0].download_button(
            "Download combined master",
            advanced["csv"],
            "combined_master.csv",
            "text/csv",
        )
        downloads[1].download_button(
            "Download alignment links",
            advanced["links"],
            "alignment_links.csv",
            "text/csv",
        )
        downloads[2].download_button(
            "Download validation JSON",
            advanced["validation"],
            "validation_report.json",
            "application/json",
        )

with guide_tab:
    st.subheader("Run the same pipeline in a terminal")
    st.code(
        """bpsd-aligner align --help
bpsd-aligner dry-run --help
bpsd-aligner xml-export --help
bpsd-aligner combine --help""",
        language="bash",
    )
    st.info(
        "The BPSD note annotation CSV is required for official note IDs and timeline fields. "
        "The unfolded MusicXML is recommended when the written score contains repeats."
    )
