"""Streamlit UI for raw score-page alignment and CSV inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from bpsd_aligner.web_pipeline import build_output_zip, run_uploaded_alignment
from bpsd_aligner.web_utils import read_csv_bytes, sanitize_path_columns, summarize_rows
from combine_yolo_xml import combine_dataset


MAX_UPLOAD_BYTES = 200 * 1024 * 1024


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


def _summary_cards(summary: dict[str, object]) -> None:
    first, second, third = st.columns(3)
    first.metric("Rows", f"{summary['rows']:,}")
    second.metric("Scores", summary["scores"])
    third.metric("Statuses", len(summary["statuses"]))


def _render_completed_job(job: dict) -> None:
    report = job["report"]
    if report["passed"]:
        st.success("Alignment and validation completed.")
    else:
        st.error("Outputs were created, but validation found errors.")
        for error in report["validation_errors"]:
            st.error(error)
    first, second, third, fourth = st.columns(4)
    first.metric("YOLO boxes", report["alignment_rows"])
    second.metric("XML events", report["xml_event_rows"])
    third.metric("Combined rows", report["combined_rows"])
    fourth.metric("Warnings", len(report["warnings"]))
    for warning in report["warnings"]:
        st.warning(warning)

    st.subheader("Review images")
    overlays = job["overlays"]
    if overlays:
        overlay_names = list(overlays)
        selected = st.selectbox("Overlay", overlay_names, key="overlay_select")
        st.image(overlays[selected], caption=selected, use_container_width=True)
    else:
        st.info("No review overlay was generated for this input.")

    st.subheader("Download results")
    columns = st.columns(5)
    columns[0].download_button(
        "All-information CSV",
        job["all_information"],
        "all_information.csv",
        "text/csv",
        use_container_width=True,
    )
    columns[1].download_button(
        "Combined event CSV",
        job["combined_master"],
        "combined_master.csv",
        "text/csv",
        use_container_width=True,
    )
    columns[2].download_button(
        "Detailed YOLO CSV",
        job["detailed_csv"],
        "alignment_detailed.csv",
        "text/csv",
        use_container_width=True,
    )
    columns[3].download_button(
        "Validation JSON",
        job["validation_json"],
        "validation_report.json",
        "application/json",
        use_container_width=True,
    )
    columns[4].download_button(
        "All outputs ZIP",
        job["output_zip"],
        "bpsd_alignment_outputs.zip",
        "application/zip",
        use_container_width=True,
    )
    with st.expander("Alignment status counts"):
        st.json(report["combined_status_counts"])


st.set_page_config(page_title="BPSD XML–YOLO Aligner", page_icon="🎼", layout="wide")
st.title("BPSD XML–YOLO Aligner")
st.caption("Upload a score page and export lossless XML + YOLO/BPS-OMR CSV data")

align_tab, inspect_tab, combine_tab, guide_tab = st.tabs(
    ["Run alignment", "Inspect CSV", "Advanced combine", "CLI guide"]
)

with align_tab:
    st.subheader("Upload one score page")
    st.write(
        "The image, YOLO TXT, MusicXML, BPSD note annotation, and notes.json are "
        "processed together. Uploaded files are written only to a temporary job directory."
    )
    left, right = st.columns(2)
    with left:
        image_upload = st.file_uploader(
            "Score image *", type=["jpg", "jpeg", "png"], key="raw_image"
        )
        yolo_upload = st.file_uploader("YOLO TXT *", type=["txt"], key="raw_yolo")
        xml_upload = st.file_uploader(
            "BPSD MusicXML (written/repetition score) *",
            type=["xml", "musicxml"],
            key="raw_xml",
        )
    with right:
        bps_upload = st.file_uploader(
            "BPSD note annotations (ann_score_note CSV) *",
            type=["csv"],
            key="raw_bps_notes",
        )
        notes_upload = st.file_uploader(
            "YOLO class map (notes.json) *", type=["json"], key="raw_notes_json"
        )
        unfolded_upload = st.file_uploader(
            "Unfolded MusicXML (recommended when the score repeats)",
            type=["xml", "musicxml"],
            key="raw_unfolded_xml",
        )

    settings_left, settings_middle, settings_right = st.columns(3)
    score_id = settings_left.text_input("Score ID", value="uploaded-score")
    page_number = settings_middle.number_input("MusicXML page", min_value=1, value=1, step=1)
    infer_fingerings = settings_right.checkbox("Infer fingering candidates", value=True)

    required = [image_upload, yolo_upload, xml_upload, bps_upload, notes_upload]
    start = st.button(
        "Align all information",
        type="primary",
        disabled=not all(required),
        use_container_width=True,
    )
    if start:
        labels = ["Image", "YOLO TXT", "MusicXML", "BPSD notes", "notes.json"]
        if all(_valid_upload(uploaded, label) for uploaded, label in zip(required, labels)):
            options = {
                "score_id": score_id,
                "page_number": int(page_number),
                "infer_fingerings": infer_fingerings,
            }
            fingerprint = _upload_fingerprint(
                [*required, unfolded_upload], options
            )
            cached = st.session_state.get("raw_alignment_job")
            if cached and cached.get("fingerprint") == fingerprint:
                st.info("The inputs match the completed session checkpoint; reusing outputs.")
            else:
                with st.status("Running alignment", expanded=True) as status:
                    progress = st.progress(0, text="Preparing uploads")
                    with tempfile.TemporaryDirectory(prefix="bpsd-web-upload-") as temporary:
                        temporary_path = Path(temporary)
                        input_dir = temporary_path / "inputs"
                        output_dir = temporary_path / "outputs"
                        input_dir.mkdir(parents=True)
                        image_path = _save_upload(image_upload, input_dir, "page.png")
                        yolo_path = _save_upload(yolo_upload, input_dir, "page.txt")
                        xml_path = _save_upload(xml_upload, input_dir, "score.xml")
                        bps_path = _save_upload(bps_upload, input_dir, "ann_score_note.csv")
                        notes_path = _save_upload(notes_upload, input_dir, "notes.json")
                        unfolded_path = (
                            _save_upload(unfolded_upload, input_dir, "score_unfolded.xml")
                            if unfolded_upload is not None
                            else None
                        )
                        try:
                            def update_progress(step: int, total: int, message: str) -> None:
                                progress.progress(step / total, text=f"{step}/{total} {message}")
                                st.write(f"{step}/{total} {message}")

                            report = run_uploaded_alignment(
                                image_path=image_path,
                                yolo_path=yolo_path,
                                xml_path=xml_path,
                                bps_notes_path=bps_path,
                                notes_json_path=notes_path,
                                unfolded_xml_path=unfolded_path,
                                output_dir=output_dir,
                                page_number=int(page_number),
                                score_id=score_id,
                                infer_fingerings=infer_fingerings,
                                progress_callback=update_progress,
                            )
                            zip_path = build_output_zip(report, output_dir / "all_outputs.zip")
                            output_paths = {
                                name: Path(path) for name, path in report["outputs"].items()
                            }
                            overlays = {
                                name: path.read_bytes()
                                for name, path in output_paths.items()
                                if name.endswith("overlay") and path.is_file()
                            }
                            validation_json = json.dumps(
                                report, ensure_ascii=False, indent=2
                            ).encode("utf-8")
                            st.session_state["raw_alignment_job"] = {
                                "fingerprint": fingerprint,
                                "report": report,
                                "all_information": output_paths["all_information_csv"].read_bytes(),
                                "combined_master": output_paths["combined_master_csv"].read_bytes(),
                                "detailed_csv": output_paths["detailed_csv"].read_bytes(),
                                "validation_json": validation_json,
                                "output_zip": zip_path.read_bytes(),
                                "overlays": overlays,
                            }
                        except Exception as error:
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
