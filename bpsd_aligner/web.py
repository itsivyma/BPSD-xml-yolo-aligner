"""Streamlit UI for inspecting and combining BPSD XML–YOLO CSV files."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import streamlit as st

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


def _summary_cards(summary: dict[str, object]) -> None:
    first, second, third = st.columns(3)
    first.metric("Rows", f"{summary['rows']:,}")
    second.metric("Scores", summary["scores"])
    third.metric("Statuses", len(summary["statuses"]))


st.set_page_config(page_title="BPSD XML–YOLO Aligner", page_icon="🎼", layout="wide")
st.title("BPSD XML–YOLO Aligner")
st.caption("Lossless MusicXML + YOLO/BPS-OMR CSV export and review")

inspect_tab, combine_tab, guide_tab = st.tabs(["Inspect CSV", "Combine XML + YOLO", "CLI guide"])

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
    st.subheader("Build a lossless combined master")
    st.write(
        "Upload the YOLO/BPS master produced by the alignment stage and the XML events CSV "
        "produced by `bpsd-aligner xml-export`. Every YOLO bbox and XML event is retained."
    )
    yolo_upload = st.file_uploader("YOLO/BPS master CSV", type=["csv"], key="yolo_master")
    xml_upload = st.file_uploader("XML events CSV", type=["csv"], key="xml_events")
    remove_paths = st.checkbox("Replace local source paths with filenames", value=True)
    if st.button("Combine and validate", type="primary", disabled=not (yolo_upload and xml_upload)):
        if _valid_upload(yolo_upload, "YOLO master") and _valid_upload(xml_upload, "XML events"):
            with st.status("Combining data", expanded=True) as status:
                with tempfile.TemporaryDirectory(prefix="bpsd-aligner-") as temporary:
                    temporary_path = Path(temporary)
                    yolo_path = temporary_path / "yolo_master.csv"
                    xml_path = temporary_path / "xml_events.csv"
                    output_path = temporary_path / "output"
                    yolo_path.write_bytes(yolo_upload.getvalue())
                    xml_path.write_bytes(xml_upload.getvalue())
                    st.write("Inputs saved; building score-level links…")
                    try:
                        report = combine_dataset(yolo_path, xml_path, output_path, resume=False)
                    except Exception as error:
                        status.update(label="Combination failed", state="error")
                        st.exception(error)
                    else:
                        combined = (output_path / "combined_master.csv").read_bytes()
                        if remove_paths:
                            combined = sanitize_path_columns(combined)
                        st.session_state["combined_csv"] = combined
                        st.session_state["alignment_links_csv"] = (
                            output_path / "alignment_links.csv"
                        ).read_bytes()
                        st.session_state["validation_json"] = json.dumps(
                            report, ensure_ascii=False, indent=2
                        ).encode("utf-8")
                        status.update(label="Combination completed", state="complete")
    if "combined_csv" in st.session_state:
        st.success("Validation passed. Files are ready to download.")
        first, second, third = st.columns(3)
        first.download_button(
            "Download combined CSV",
            st.session_state["combined_csv"],
            "combined_master.csv",
            "text/csv",
        )
        second.download_button(
            "Download alignment links",
            st.session_state["alignment_links_csv"],
            "alignment_links.csv",
            "text/csv",
        )
        third.download_button(
            "Download validation JSON",
            st.session_state["validation_json"],
            "validation_report.json",
            "application/json",
        )

with guide_tab:
    st.subheader("Run the complete system in a terminal")
    st.code(
        """python -m venv .venv
source .venv/bin/activate
pip install .

bpsd-aligner --help
bpsd-aligner align --help
bpsd-aligner dry-run --help
bpsd-aligner xml-export --help
bpsd-aligner combine --help""",
        language="bash",
    )
    st.info(
        "Raw score images and BPSD dataset files are not included in this repository. "
        "Use paths to your licensed local copies."
    )
