"""Single-page upload pipeline shared by the Streamlit UI and tests."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image

from bps_xml_alignment import (
    load_bps_notes,
    load_categories,
    load_yolo,
    parse_musicxml_page,
    run_alignment,
)
from combine_yolo_xml import combine_dataset
from dataset_dry_run import FIELDS, OFFICIAL_FIELDS
from pipeline_checkpoint import atomic_write_csv, atomic_write_json, emit_progress
from repeat_mapping import build_repeat_mapping, write_repeat_mapping
from xml_export import BPS_FIELDS, EVENT_FIELDS, NODE_FIELDS, export_score


PIPELINE_VERSION = "0.3.2"
MAX_DECODED_IMAGE_PIXELS = 100_000_000
ProgressCallback = Callable[[int, int, str], None]
FINAL_BPS_FIELDS = [*OFFICIAL_FIELDS, "human_corrected"]
FINAL_UNCERTAIN_FIELDS = [
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
]
XML_SPAN_FIELDS = [
    "span_id",
    "score_id",
    "class",
    "span_type",
    "number",
    "start_xml_event_id",
    "end_xml_event_id",
    "start_anchor_xml_event_ids_json",
    "end_anchor_xml_event_ids_json",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "start_page",
    "end_page",
    "start_xml_measure",
    "end_xml_measure",
    "start_staff",
    "end_staff",
    "voice",
    "cross_page",
    "status",
]
PERFORMANCE_FIELDS = [
    "source_record_type",
    "record_id",
    "occurrence",
    "occurrence_count",
    "repeat_group_id",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
]


def safe_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _time_sort_key(row: dict) -> tuple:
    def number(value: object, fallback: float) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return fallback

    source = row.get("source_record_type") or row.get("row_origin", "")
    source_rank = {"yolo": 0, "xml_event": 1, "xml": 1, "xml_node": 2}.get(
        source, 3
    )
    return (
        number(row.get("start_meas"), float("inf")),
        number(row.get("end_meas"), float("inf")),
        source_rank,
        number(row.get("yolo_line"), float("inf")),
        str(row.get("class", "")),
        str(row.get("record_id", "")),
    )


def _sort_csv_by_musical_time(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = sorted(reader, key=_time_sort_key)
    atomic_write_csv(path, fields, rows)
    return len(rows)


def build_final_bps_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the strict bounding-box schema defined by BPS-OMR annotations.

    Machine candidates are useful for review overlays, but uncertain values
    must not become final annotations. Only direct matches or human-corrected
    rows retain semantic links and timing; literal NA values become blanks.
    """

    output = []
    for source in rows:
        corrected = (
            str(source.get("human_approved", "")).strip().lower() == "true"
            or source.get("match_source") == "human_review"
        )
        confirmed = source.get("alignment_status") == "matched" or corrected
        row = {
            field: "" if str(source.get(field, "")).strip().upper() == "NA"
            else str(source.get(field, ""))
            for field in OFFICIAL_FIELDS
        }
        if not confirmed:
            for field in FINAL_UNCERTAIN_FIELDS:
                row[field] = ""
        if row.get("class") != "stem":
            row["stem_dir"] = ""
        row["human_corrected"] = "1" if corrected else "0"
        output.append({field: row.get(field, "") for field in FINAL_BPS_FIELDS})
    return output


def _report_progress(
    step: int,
    total: int,
    message: str,
    callback: ProgressCallback | None,
) -> None:
    emit_progress("web-upload", step, total, message)
    if callback is not None:
        callback(step, total, message)


def _build_all_information_csv(
    combined_path: Path,
    events: list[dict],
    nodes: list[dict],
    destination: Path,
) -> tuple[int, int, int]:
    """Write one CSV containing every YOLO/XML-event row and XML source node."""

    with combined_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        combined_fields = list(reader.fieldnames or [])
        combined_rows = list(reader)
    source_field = "source_record_type"
    all_fields = (
        BPS_FIELDS
        + [source_field]
        + [field for field in combined_fields if field not in BPS_FIELDS]
    )
    for field in EVENT_FIELDS + NODE_FIELDS:
        if field not in all_fields:
            all_fields.append(field)
    rows = []
    yolo_rows = [row for row in combined_rows if row.get("row_origin") == "yolo"]
    for source in yolo_rows:
        row = {field: source.get(field, "") for field in all_fields}
        row[source_field] = "yolo"
        rows.append(row)
    for event in events:
        row = {field: "" for field in all_fields}
        row.update({field: event.get(field, "") for field in EVENT_FIELDS})
        row.update(
            {
                source_field: "xml_event",
                "record_id": event["xml_event_id"],
                "row_origin": "xml_event",
                "combined_status": "xml_source_event",
                "source_alignment_status": "xml_source_event",
                "xml_path": event.get("source_xml_path", ""),
                "xml_page": event.get("page", ""),
                "xml_measure": event.get("xml_measure", ""),
            }
        )
        rows.append(row)
    for node in nodes:
        row = {field: "" for field in all_fields}
        row.update({field: node.get(field, "") for field in NODE_FIELDS})
        row.update(
            {
                source_field: "xml_node",
                "class": f"xmlNode:{node.get('tag', 'unknown')}",
                "record_id": node["xml_node_id"],
                "row_origin": "xml_node",
                "combined_status": "xml_source_node",
                "source_alignment_status": "xml_source_node",
                "xml_path": node.get("source_xml_path", ""),
                "xml_page": node.get("page", ""),
                "xml_measure": node.get("measure_number", ""),
            }
        )
        rows.append(row)
    rows.sort(key=_time_sort_key)
    atomic_write_csv(destination, all_fields, rows)
    return len(yolo_rows), len(events), len(nodes)


def _build_yolo_xml_timeline_csv(
    yolo_path: Path,
    events: list[dict],
    destination: Path,
) -> tuple[int, int]:
    """Merge every aligned YOLO row and XML event as separate timed rows."""

    with yolo_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        yolo_fields = list(reader.fieldnames or [])
        yolo_rows = list(reader)
    source_field = "source_record_type"
    fields = BPS_FIELDS + [source_field] + [
        field for field in yolo_fields if field not in BPS_FIELDS
    ]
    for field in EVENT_FIELDS:
        if field not in fields:
            fields.append(field)
    rows = []
    for source in yolo_rows:
        row = {field: source.get(field, "") for field in fields}
        row.update(
            {
                source_field: "yolo",
                "record_id": source.get("bbox_id", ""),
                "row_origin": "yolo",
            }
        )
        rows.append(row)
    for event in events:
        row = {field: "" for field in fields}
        row.update({field: event.get(field, "") for field in EVENT_FIELDS})
        row.update(
            {
                source_field: "xml_event",
                "record_id": event["xml_event_id"],
                "row_origin": "xml_event",
                "xml_path": event.get("source_xml_path", ""),
                "xml_page": event.get("page", ""),
                "xml_measure": event.get("xml_measure", ""),
            }
        )
        rows.append(row)
    rows.sort(key=_time_sort_key)
    atomic_write_csv(destination, fields, rows)
    return len(yolo_rows), len(events)


def _json_value(value: object, fallback):
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, json.JSONDecodeError):
        return fallback
    return parsed


def _span_marker(event: dict) -> tuple[str, str, str] | None:
    """Return span kind, marker type, and pairing number for XML endpoints."""

    subtype = str(event.get("event_subtype", ""))
    attributes = _json_value(event.get("xml_attributes_json"), {})
    if not isinstance(attributes, dict):
        attributes = {}
    marker_type = str(attributes.get("type", "")).lower()
    number = str(attributes.get("number", "1"))
    if subtype in {"slur", "tie", "tied"}:
        kind = "tie" if subtype in {"tie", "tied"} else "slur"
        if marker_type in {"start", "stop", "continue"}:
            return kind, marker_type, number
    if event.get("event_type") == "direction" and subtype in {
        "wedge",
        "octave-shift",
        "pedal",
    }:
        if subtype == "wedge":
            marker = "stop" if marker_type == "stop" else "start"
        elif subtype == "octave-shift":
            marker = "stop" if marker_type == "stop" else "start"
        else:
            marker = "stop" if marker_type in {"stop", "change"} else "start"
        return subtype, marker, number
    return None


def build_xml_spans(events: list[dict], destination: Path) -> list[dict]:
    """Pair whole-score XML start/stop events, including cross-page spans."""

    note_by_id = {
        row.get("xml_event_id", ""): row
        for row in events
        if row.get("event_type") == "note"
    }
    open_spans: dict[tuple[str, str, str, str], list[dict]] = {}
    seen_tie_markers: set[tuple[str, str, str]] = set()
    rows = []

    def anchor(event: dict) -> dict:
        ids = _json_value(event.get("anchor_xml_event_ids_json"), [])
        return note_by_id.get(ids[0], {}) if isinstance(ids, list) and ids else {}

    for event in events:
        marker = _span_marker(event)
        if marker is None:
            continue
        kind, marker_type, number = marker
        event_anchor = anchor(event)
        if kind == "tie":
            anchor_ids = str(event.get("anchor_xml_event_ids_json", "[]"))
            semantic_marker = (anchor_ids, marker_type, number)
            if semantic_marker in seen_tie_markers:
                continue
            seen_tie_markers.add(semantic_marker)
            pairing_scope = "|".join(
                (
                    str(event.get("staff", "")),
                    str(event.get("voice", "")),
                    str(
                        event_anchor.get(
                            "midi_pitch", event_anchor.get("pitch", "")
                        )
                    ),
                )
            )
        elif kind == "slur":
            pairing_scope = str(event.get("voice", ""))
        else:
            pairing_scope = str(event.get("staff", ""))
        key = (
            str(event.get("score_id", "")),
            kind,
            pairing_scope,
            number,
        )
        if marker_type == "start":
            open_spans.setdefault(key, []).append(event)
            continue
        if marker_type == "continue":
            continue
        starts = open_spans.get(key, [])
        start = starts.pop() if starts else None
        if start is None:
            rows.append(
                {
                    **{field: "" for field in XML_SPAN_FIELDS},
                    "span_id": f"{event.get('score_id', '')}:SPAN{len(rows) + 1:07d}",
                    "score_id": event.get("score_id", ""),
                    "class": event.get("class", ""),
                    "span_type": kind,
                    "number": number,
                    "end_xml_event_id": event.get("xml_event_id", ""),
                    "end_meas": event.get("end_meas", ""),
                    "end_page": event.get("page", ""),
                    "end_xml_measure": event.get("xml_measure", ""),
                    "end_staff": event.get("staff", ""),
                    "voice": event.get("voice", ""),
                    "status": "stop_without_start",
                }
            )
            continue
        start_anchor = anchor(start)
        end_anchor = anchor(event)
        start_note = start_anchor.get("start_note", "")
        end_note = end_anchor.get("end_note", "")
        connected = [value for value in (start_note, end_note) if str(value) != ""]
        rows.append(
            {
                "span_id": f"{start.get('score_id', '')}:SPAN{len(rows) + 1:07d}",
                "score_id": start.get("score_id", ""),
                "class": start.get("class", kind),
                "span_type": kind,
                "number": number,
                "start_xml_event_id": start.get("xml_event_id", ""),
                "end_xml_event_id": event.get("xml_event_id", ""),
                "start_anchor_xml_event_ids_json": start.get(
                    "anchor_xml_event_ids_json", "[]"
                ),
                "end_anchor_xml_event_ids_json": event.get(
                    "anchor_xml_event_ids_json", "[]"
                ),
                "start_meas": start_anchor.get(
                    "start_meas", start.get("start_meas", "")
                ),
                "end_meas": end_anchor.get("end_meas", event.get("end_meas", "")),
                "start_note": start_note,
                "end_note": end_note,
                "connected_note": json.dumps(connected),
                "start_page": start.get("page", ""),
                "end_page": event.get("page", ""),
                "start_xml_measure": start.get("xml_measure", ""),
                "end_xml_measure": event.get("xml_measure", ""),
                "start_staff": start.get("staff", ""),
                "end_staff": event.get("staff", ""),
                "voice": start.get("voice", ""),
                "cross_page": str(start.get("page", ""))
                != str(event.get("page", "")),
                "status": "paired",
            }
        )
    for starts in open_spans.values():
        for start in starts:
            rows.append(
                {
                    **{field: "" for field in XML_SPAN_FIELDS},
                    "span_id": f"{start.get('score_id', '')}:SPAN{len(rows) + 1:07d}",
                    "score_id": start.get("score_id", ""),
                    "class": start.get("class", ""),
                    "span_type": _span_marker(start)[0],
                    "number": _span_marker(start)[2],
                    "start_xml_event_id": start.get("xml_event_id", ""),
                    "start_meas": start.get("start_meas", ""),
                    "start_page": start.get("page", ""),
                    "start_xml_measure": start.get("xml_measure", ""),
                    "start_staff": start.get("staff", ""),
                    "voice": start.get("voice", ""),
                    "status": "start_without_stop",
                }
            )
    rows.sort(key=_time_sort_key)
    atomic_write_csv(destination, XML_SPAN_FIELDS, rows)
    return rows


def build_performance_expanded_timeline(
    yolo_rows: list[dict], xml_events: list[dict], destination: Path
) -> list[dict]:
    """Expand repeat occurrences into scalar, performance-ordered rows."""

    source_fields = []
    for row in [*yolo_rows, *xml_events]:
        for field in row:
            if field not in source_fields and field not in PERFORMANCE_FIELDS:
                source_fields.append(field)
    fields = [*PERFORMANCE_FIELDS, *source_fields]
    output = []
    for source_type, rows, repeat_field in (
        ("yolo", yolo_rows, "repeat_occurrences_json"),
        ("xml_event", xml_events, "repeat_json"),
    ):
        for source in rows:
            occurrences = _json_value(source.get(repeat_field), [])
            if not isinstance(occurrences, list) or not occurrences:
                occurrences = [{}]
            for index, occurrence in enumerate(occurrences, start=1):
                if not isinstance(occurrence, dict):
                    occurrence = {}
                row = {field: source.get(field, "") for field in fields}
                row.update(
                    {
                        "source_record_type": source_type,
                        "record_id": source.get(
                            "bbox_id" if source_type == "yolo" else "xml_event_id",
                            "",
                        ),
                        "occurrence": occurrence.get("repeat_occurrence", index),
                        "occurrence_count": occurrence.get(
                            "repeat_occurrence_count", len(occurrences)
                        ),
                        "repeat_group_id": occurrence.get(
                            "repeat_group_id", source.get("repeat_group_id", "")
                        ),
                        "start_meas": occurrence.get(
                            "start_meas",
                            occurrence.get("bps_time", source.get("start_meas", "")),
                        ),
                        "end_meas": occurrence.get(
                            "end_meas",
                            occurrence.get("bps_time", source.get("end_meas", "")),
                        ),
                        "start_note": occurrence.get(
                            "start_note",
                            occurrence.get("note_id", source.get("start_note", "")),
                        ),
                        "end_note": occurrence.get(
                            "end_note",
                            occurrence.get("note_id", source.get("end_note", "")),
                        ),
                    }
                )
                output.append(row)
    output.sort(key=_time_sort_key)
    atomic_write_csv(destination, fields, output)
    return output


def build_batch_information_outputs(
    *,
    yolo_rows: list[dict],
    xml_events: list[dict],
    xml_nodes: list[dict],
    output_dir: Path,
) -> dict:
    """Build one non-duplicated set of complete exports for a page batch."""

    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_rows = [dict(row) for row in yolo_rows]
    xml_events = [dict(row) for row in xml_events]
    xml_nodes = [dict(row) for row in xml_nodes]
    yolo_rows.sort(key=_time_sort_key)
    xml_events.sort(key=_time_sort_key)

    yolo_path = output_dir / "yolo_aligned.csv"
    xml_dir = output_dir / "xml"
    xml_events_path = xml_dir / "xml_events.csv"
    xml_nodes_path = xml_dir / "xml_nodes.csv"
    atomic_write_csv(yolo_path, FIELDS, yolo_rows)
    atomic_write_csv(xml_events_path, EVENT_FIELDS, xml_events)
    atomic_write_csv(xml_nodes_path, NODE_FIELDS, xml_nodes)

    combined_dir = output_dir / "combined"
    combined_report = combine_dataset(
        yolo_path, xml_events_path, combined_dir, resume=False
    )
    timeline_path = output_dir / "yolo_xml_timeline.csv"
    timeline_yolo_count, timeline_event_count = _build_yolo_xml_timeline_csv(
        yolo_path, xml_events, timeline_path
    )
    all_information_path = output_dir / "all_information.csv"
    information_yolo_count, information_event_count, information_node_count = (
        _build_all_information_csv(
            combined_dir / "combined_master.csv",
            xml_events,
            xml_nodes,
            all_information_path,
        )
    )
    errors = list(combined_report.get("validation_errors", []))
    if timeline_yolo_count != len(yolo_rows):
        errors.append("Batch timeline does not preserve every YOLO row")
    if timeline_event_count != len(xml_events):
        errors.append("Batch timeline does not preserve every XML event")
    if information_yolo_count != len(yolo_rows):
        errors.append("Batch all-information CSV does not preserve every YOLO row")
    if information_event_count != len(xml_events):
        errors.append("Batch all-information CSV does not preserve every XML event")
    if information_node_count != len(xml_nodes):
        errors.append("Batch all-information CSV does not preserve every XML node")
    spans_path = output_dir / "xml_spans.csv"
    spans = build_xml_spans(xml_events, spans_path)
    performance_path = output_dir / "performance_expanded_timeline.csv"
    performance_rows = build_performance_expanded_timeline(
        yolo_rows, xml_events, performance_path
    )

    return {
        "passed": not errors,
        "validation_errors": errors,
        "yolo_rows": len(yolo_rows),
        "xml_event_rows": len(xml_events),
        "xml_node_rows": len(xml_nodes),
        "timeline_rows": len(yolo_rows) + len(xml_events),
        "all_information_rows": len(yolo_rows) + len(xml_events) + len(xml_nodes),
        "xml_span_rows": len(spans),
        "performance_expanded_rows": len(performance_rows),
        "outputs": {
            "yolo_aligned_csv": yolo_path,
            "xml_events_csv": xml_events_path,
            "xml_nodes_csv": xml_nodes_path,
            "yolo_xml_timeline_csv": timeline_path,
            "all_information_csv": all_information_path,
            "combined_master_csv": combined_dir / "combined_master.csv",
            "alignment_links_csv": combined_dir / "alignment_links.csv",
            "xml_spans_csv": spans_path,
            "performance_expanded_timeline_csv": performance_path,
        },
    }


def prepare_score_sources(
    *,
    xml_path: Path,
    bps_notes_path: Path,
    output_dir: Path,
    score_id: str,
    unfolded_xml_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = True,
) -> dict:
    """Parse whole-score sources once for every page in a website job."""

    score_id = safe_identifier(score_id, "uploaded-score")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "shared_score_report.json"
    source_hashes = {
        "xml_sha256": _sha256(xml_path),
        "bps_notes_sha256": _sha256(bps_notes_path),
        "unfolded_xml_sha256": _sha256(unfolded_xml_path)
        if unfolded_xml_path is not None
        else "",
    }
    if resume and report_path.is_file():
        try:
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            saved_paths = [
                Path(saved[field])
                for field in (
                    "repeat_csv",
                    "repeat_json",
                    "xml_nodes_csv",
                    "xml_events_csv",
                )
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            saved = None
        if (
            saved is not None
            and saved.get("pipeline_version") == PIPELINE_VERSION
            and saved.get("score_id") == score_id
            and all(saved.get(key) == value for key, value in source_hashes.items())
            and all(path.is_file() for path in saved_paths)
        ):
            _report_progress(
                2, 2, "Resumed shared score checkpoint", progress_callback
            )
            return {
                **saved,
                "repeat_csv": saved_paths[0],
                "repeat_json": saved_paths[1],
                "xml_nodes_csv": saved_paths[2],
                "xml_events_csv": saved_paths[3],
            }
    repeat_dir = output_dir / "repeat_mapping"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    _report_progress(0, 2, "Preparing shared repeat mapping", progress_callback)
    repeat_source = unfolded_xml_path or xml_path
    repeat_report = build_repeat_mapping(xml_path, repeat_source)
    repeat_csv = repeat_dir / f"{score_id}_repeat_mapping.csv"
    repeat_json = repeat_dir / f"{score_id}_repeat_mapping.json"
    write_repeat_mapping(repeat_report, repeat_csv, repeat_json)

    _report_progress(1, 2, "Exporting shared MusicXML nodes and events", progress_callback)
    nodes, events = export_score(
        {
            "score_id": score_id,
            "xml_path": str(xml_path),
            "bps_notes_path": str(bps_notes_path),
        },
        repeat_dir,
    )
    _sanitize_xml_source_paths(nodes, xml_path.name)
    _sanitize_xml_source_paths(events, xml_path.name)
    events.sort(key=_time_sort_key)
    xml_dir = output_dir / "xml"
    nodes_path = xml_dir / "xml_nodes.csv"
    events_path = xml_dir / "xml_events.csv"
    atomic_write_csv(nodes_path, NODE_FIELDS, nodes)
    atomic_write_csv(events_path, EVENT_FIELDS, events)
    warnings = []
    if unfolded_xml_path is None:
        warnings.append(
            "No unfolded MusicXML was uploaded; identity repeat mapping was used. "
            "Measures after printed repeats may need review."
        )
    if repeat_report["unresolved_unfolded_measures"]:
        warnings.append(
            f"Repeat mapping has {len(repeat_report['unresolved_unfolded_measures'])} "
            "unresolved measures."
        )
    errors = []
    missing_xml_classes = [
        event["xml_event_id"] for event in events if not event.get("class")
    ]
    if missing_xml_classes:
        errors.append(f"XML events without class names: {len(missing_xml_classes)}")
    bps_notes = load_bps_notes(bps_notes_path)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "score_id": score_id,
        **source_hashes,
        "identity_repeat_mapping": unfolded_xml_path is None,
        "xml_node_rows": len(nodes),
        "xml_event_rows": len(events),
        "bps_note_count": len(bps_notes),
        "bps_note_ids": [str(note["note_id"]) for note in bps_notes],
        "warnings": warnings,
        "validation_errors": errors,
        "passed": not errors,
        "repeat_csv": repeat_csv,
        "repeat_json": repeat_json,
        "xml_nodes_csv": nodes_path,
        "xml_events_csv": events_path,
    }
    atomic_write_json(
        report_path,
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in report.items()
        },
    )
    _report_progress(2, 2, "Shared score preprocessing complete", progress_callback)
    return report


def validate_upload_inputs(
    image_path: Path,
    yolo_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    notes_json_path: Path,
    page_number: int = 1,
    validate_repeat_mapping: bool = True,
    system_start_measures: list[int] | None = None,
    page_end_measure: int | None = None,
) -> dict[str, int]:
    with Image.open(image_path) as image:
        if image.width * image.height > MAX_DECODED_IMAGE_PIXELS:
            raise ValueError(
                f"Score image has {image.width * image.height:,} decoded pixels; "
                f"the limit is {MAX_DECODED_IMAGE_PIXELS:,}."
            )
        image.verify()
    categories = load_categories(notes_json_path)
    boxes = load_yolo(yolo_path, categories=categories)
    if not boxes:
        raise ValueError("YOLO TXT contains no bounding boxes")
    invalid_geometry = [
        box["txt_line"]
        for box in boxes
        if not (
            0 <= box["x"] <= 1
            and 0 <= box["y"] <= 1
            and 0 < box["w"] <= 1
            and 0 < box["h"] <= 1
        )
    ]
    if invalid_geometry:
        raise ValueError(
            f"YOLO rows have invalid normalized geometry: {invalid_geometry}"
        )
    missing_classes = sorted({box["class_id"] for box in boxes if not box["class"]})
    if missing_classes:
        raise ValueError(f"notes.json is missing YOLO class IDs: {missing_classes}")
    bps_notes = load_bps_notes(bps_notes_path)
    if not bps_notes:
        raise ValueError("BPSD note annotation CSV contains no notes")
    # Parsing the XML here gives a focused upload-validation error before the
    # expensive image alignment begins.
    if validate_repeat_mapping:
        build_repeat_mapping(xml_path, xml_path)
    xml_page = parse_musicxml_page(
        xml_path,
        page_number=page_number,
        system_start_measures=system_start_measures,
        page_end_measure=page_end_measure,
    )
    if not xml_page["measures"]:
        raise ValueError(f"MusicXML page {page_number} contains no measures")
    return {
        "yolo_boxes": len(boxes),
        "classes": len(categories),
        "bps_notes": len(bps_notes),
    }


def _canonical_master_rows(
    detailed_rows: list[dict[str, str]],
    *,
    score_id: str,
    page_id: str,
    page_number: int,
    image_path: Path,
    yolo_path: Path,
    xml_path: Path,
    unfolded_xml_path: Path | None,
    bps_notes_path: Path,
) -> list[dict[str, str]]:
    image_hash = _sha256(image_path)
    yolo_hash = _sha256(yolo_path)
    rows: list[dict[str, str]] = []
    for detailed in detailed_rows:
        row = {field: detailed.get(field, "") for field in OFFICIAL_FIELDS}
        status = detailed.get("status", "")
        alignment_status = {
            "matched": "matched",
            "inferred": "candidate",
            "review": "ambiguous",
        }.get(status, "unresolved")
        system = detailed.get("system", "")
        row.update(
            {
                "dataset_id": "BPSD-web-upload-v1",
                "score_id": score_id,
                "page_id": page_id,
                "scan_page": str(page_number),
                "yolo_line": detailed.get("txt_line", ""),
                "bbox_id": f"{page_id}:Y{detailed.get('txt_line', '')}",
                "image_path": image_path.name,
                "yolo_path": yolo_path.name,
                "xml_path": xml_path.name,
                "unfolded_xml_path": unfolded_xml_path.name if unfolded_xml_path else "",
                "sibelius_path": "",
                "bps_notes_path": bps_notes_path.name,
                "image_sha256": image_hash,
                "yolo_sha256": yolo_hash,
                "scan_system_index": system,
                "xml_page": str(page_number),
                "xml_systems_json": json.dumps([int(system)]) if str(system).isdigit() else "[]",
                "written_measure_start": detailed.get(
                    "start_xml_measure", detailed.get("xml_measure", "")
                ),
                "written_measure_end": detailed.get(
                    "end_xml_measure", detailed.get("xml_measure", "")
                ),
                "xml_measure": detailed.get("xml_measure", ""),
                "xml_symbol": detailed.get("xml_symbol", ""),
                "staff": detailed.get("xml_staff", ""),
                "target_type": detailed.get("target_type", ""),
                "note_ids": detailed.get("note_ids", ""),
                "pitches": detailed.get("pitches", ""),
                "repeat_occurrences_json": detailed.get("repeat_occurrences_json", ""),
                "repeat_occurrence_count": detailed.get("repeat_occurrence_count", ""),
                "repeat_group_id": detailed.get("repeat_group_id", ""),
                "movement_scope_status": "in_bpsd_scope",
                "page_mapping_status": "direct",
                "mapping_source": "web_uploaded_page_number",
                "match_source": detailed.get("match_source", ""),
                "confidence": detailed.get("confidence", ""),
                "alignment_status": alignment_status,
                "review_status": (
                    "not_required" if alignment_status == "matched" else "needs_review"
                ),
                "human_approved": "false",
                "reviewer": "",
                "reviewed_at": "",
                "review_source": "",
                "original_candidate_json": "",
                "corrected_value_json": "",
                "comment": "",
                "target_x_px": detailed.get("target_x_px", ""),
                "target_y_px": detailed.get("target_y_px", ""),
                "end_target_x_px": detailed.get("end_target_x_px", ""),
                "end_target_y_px": detailed.get("end_target_y_px", ""),
                "pipeline_version": PIPELINE_VERSION,
                "error_code": "",
                "error_message": "",
            }
        )
        rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def _sanitize_xml_source_paths(rows: list[dict], source_name: str) -> None:
    for row in rows:
        if "source_xml_path" in row:
            row["source_xml_path"] = source_name


def run_uploaded_alignment(
    *,
    image_path: Path,
    yolo_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    notes_json_path: Path,
    output_dir: Path,
    page_number: int = 1,
    score_id: str = "uploaded-score",
    unfolded_xml_path: Path | None = None,
    clean_image_path: Path | None = None,
    infer_fingerings: bool = True,
    progress_callback: ProgressCallback | None = None,
    prepared_score: dict | None = None,
    build_complete_exports: bool = True,
    system_start_measures: list[int] | None = None,
    page_end_measure: int | None = None,
) -> dict:
    """Run a complete single-page alignment and lossless CSV export."""

    score_id = safe_identifier(score_id, "uploaded-score")
    page_id = safe_identifier(image_path.stem, f"{score_id}-page-{page_number}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    _report_progress(0, 6, "Validating uploaded files", progress_callback)
    input_counts = validate_upload_inputs(
        image_path,
        yolo_path,
        xml_path,
        bps_notes_path,
        notes_json_path,
        page_number,
        validate_repeat_mapping=prepared_score is None,
        system_start_measures=system_start_measures,
        page_end_measure=page_end_measure,
    )
    if clean_image_path is not None:
        try:
            with Image.open(clean_image_path) as clean_image:
                clean_image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Rendered clean repetition PDF page is not a valid image: {error}"
            ) from error
        input_counts["clean_reference_pages"] = 1
    else:
        input_counts["clean_reference_pages"] = 0

    _report_progress(1, 6, "Mapping written and unfolded measures", progress_callback)
    if prepared_score is None:
        prepared_score = prepare_score_sources(
            xml_path=xml_path,
            bps_notes_path=bps_notes_path,
            output_dir=output_dir / "shared_score",
            score_id=score_id,
            unfolded_xml_path=unfolded_xml_path,
        )
    repeat_csv = Path(prepared_score["repeat_csv"])
    warnings.extend(prepared_score.get("warnings", []))
    errors.extend(prepared_score.get("validation_errors", []))

    _report_progress(2, 6, "Aligning YOLO boxes to score information", progress_callback)
    alignment_dir = output_dir / "alignment"
    alignment_report = run_alignment(
        image_path=image_path,
        yolo_path=yolo_path,
        xml_path=xml_path,
        bps_note_path=bps_notes_path,
        output_dir=alignment_dir,
        page_number=page_number,
        infer_fingerings=infer_fingerings,
        notes_json_path=notes_json_path,
        include_all_symbols=True,
        repeat_mapping_path=repeat_csv,
        clean_image_path=clean_image_path,
        system_start_measures=system_start_measures,
        page_end_measure=page_end_measure,
    )
    clean_reference = alignment_report.get("clean_reference", {})
    if clean_image_path is not None and not clean_reference.get("used"):
        warnings.append(
            "The clean repetition PDF page could not be matched structurally; "
            "alignment fell back to MusicXML-width geometry for this page."
        )
    detailed_path = Path(alignment_report["outputs"]["detailed_csv"])
    detailed_rows = _read_csv(detailed_path)
    missing_time_lines = [
        row.get("txt_line", "")
        for row in detailed_rows
        if row.get("start_meas") in {"", "NA", None}
        or row.get("end_meas") in {"", "NA", None}
    ]
    if missing_time_lines:
        errors.append(
            "YOLO rows without start/end time: " + ", ".join(missing_time_lines)
        )
    master_rows = _canonical_master_rows(
        detailed_rows,
        score_id=score_id,
        page_id=page_id,
        page_number=page_number,
        image_path=image_path,
        yolo_path=yolo_path,
        xml_path=xml_path,
        unfolded_xml_path=unfolded_xml_path,
        bps_notes_path=bps_notes_path,
    )
    master_rows.sort(key=_time_sort_key)
    yolo_master_path = output_dir / "yolo_master.csv"
    yolo_aligned_path = output_dir / "yolo_aligned.csv"
    final_bps_path = output_dir / "bps_omr_final.csv"
    review_queue_path = output_dir / "review_queue.csv"
    atomic_write_csv(yolo_master_path, FIELDS, master_rows)
    atomic_write_csv(yolo_aligned_path, FIELDS, master_rows)
    atomic_write_csv(
        final_bps_path,
        FINAL_BPS_FIELDS,
        build_final_bps_rows(master_rows),
    )
    atomic_write_csv(
        review_queue_path,
        FIELDS,
        [row for row in master_rows if row["alignment_status"] != "matched"],
    )

    _report_progress(3, 6, "Exporting every MusicXML node and event", progress_callback)
    xml_nodes_path = Path(prepared_score["xml_nodes_csv"])
    xml_events_path = Path(prepared_score["xml_events_csv"])
    nodes = _read_csv(xml_nodes_path)
    events = _read_csv(xml_events_path)

    _report_progress(4, 6, "Building lossless XML + YOLO master", progress_callback)
    combined_report = {
        "combined_rows": 0,
        "combined_status_counts": {},
    }
    complete_output_paths = {}
    if build_complete_exports:
        combined_dir = output_dir / "combined"
        combined_report = combine_dataset(
            yolo_master_path, xml_events_path, combined_dir, resume=False
        )
        if not combined_report["passed"]:
            errors.extend(combined_report["validation_errors"])
        sorted_combined_rows = _sort_csv_by_musical_time(
            combined_dir / "combined_master.csv"
        )
        if sorted_combined_rows != combined_report["combined_rows"]:
            errors.append("Time sorting changed the combined row count")
        timeline_path = output_dir / "yolo_xml_timeline.csv"
        timeline_yolo_count, timeline_event_count = _build_yolo_xml_timeline_csv(
            yolo_aligned_path, events, timeline_path
        )
        if timeline_yolo_count != len(master_rows):
            errors.append("Timeline CSV does not preserve every aligned YOLO row")
        if timeline_event_count != len(events):
            errors.append("Timeline CSV does not preserve every XML event")
        all_information_path = output_dir / "all_information.csv"
        included_yolo_count, included_event_count, included_node_count = (
            _build_all_information_csv(
                combined_dir / "combined_master.csv",
                events,
                nodes,
                all_information_path,
            )
        )
        if included_yolo_count != len(master_rows):
            errors.append("All-information CSV does not preserve every YOLO bbox")
        if included_event_count != len(events):
            errors.append("All-information CSV does not preserve every XML event")
        if included_node_count != len(nodes):
            errors.append("All-information CSV does not preserve every XML node")
        complete_output_paths = {
            "yolo_xml_timeline_csv": timeline_path,
            "all_information_csv": all_information_path,
            "combined_master_csv": combined_dir / "combined_master.csv",
            "alignment_links_csv": combined_dir / "alignment_links.csv",
        }

    _report_progress(5, 6, "Validating and packaging outputs", progress_callback)
    overlay_paths = {
        key: Path(value)
        for key, value in alignment_report["outputs"].items()
        if key.endswith("overlay") and value
    }
    outputs = {
        "official_csv": Path(alignment_report["outputs"]["csv"]),
        "detailed_csv": detailed_path,
        "final_bps_csv": final_bps_path,
        "yolo_aligned_csv": yolo_aligned_path,
        "review_queue_csv": review_queue_path,
        "yolo_master_csv": yolo_master_path,
        "xml_nodes_csv": xml_nodes_path,
        "xml_events_csv": xml_events_path,
        **complete_output_paths,
        **overlay_paths,
    }
    missing_outputs = [name for name, path in outputs.items() if not path.is_file()]
    if missing_outputs:
        errors.append(f"Missing expected outputs: {missing_outputs}")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "score_id": score_id,
        "page_id": page_id,
        "page_number": page_number,
        "input_counts": input_counts,
        "identity_repeat_mapping": prepared_score.get(
            "identity_repeat_mapping", unfolded_xml_path is None
        ),
        "alignment_rows": len(master_rows),
        "yolo_rows_with_time": len(master_rows) - len(missing_time_lines),
        "yolo_rows_needing_review": sum(
            row.get("status") != "matched" for row in detailed_rows
        ),
        "yolo_status_counts": dict(
            Counter(row.get("status", "") for row in detailed_rows)
        ),
        "class_overlay_count": alignment_report.get("class_overlay_count", 0),
        "clean_reference": clean_reference,
        "xml_node_rows": len(nodes),
        "xml_event_rows": len(events),
        "combined_rows": combined_report["combined_rows"],
        "timeline_rows": len(master_rows) + len(events) if build_complete_exports else 0,
        "all_information_rows": (
            len(master_rows) + len(events) + len(nodes) if build_complete_exports else 0
        ),
        "combined_status_counts": combined_report["combined_status_counts"],
        "sort_order": "start_meas,end_meas,source_record_type,yolo_line,class",
        "warnings": warnings,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    report_path = output_dir / "validation_report.json"
    atomic_write_json(report_path, report)
    report["outputs"]["validation_json"] = str(report_path)
    _report_progress(
        6,
        6,
        f"Completed: passed={report['passed']} errors={len(errors)}",
        progress_callback,
    )
    return report


def build_output_zip(report: dict, destination: Path) -> Path:
    """Package user-facing CSV, JSON, and review images."""

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names = set()
        for name, value in report.get("outputs", {}).items():
            path = Path(value)
            if path.is_file():
                archive_name = path.name
                if archive_name in used_names:
                    archive_name = f"{name}{path.suffix or '.bin'}"
                used_names.add(archive_name)
                archive.write(path, arcname=archive_name)
    return destination
