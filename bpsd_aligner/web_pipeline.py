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

from bpsd_aligner import __version__ as PIPELINE_VERSION
from bpsd_aligner.bps_omr_schema import musical_time_for_class
from bpsd_aligner.span_semantics import endpoint_note_ids, index_chord_members
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


MAX_DECODED_IMAGE_PIXELS = 100_000_000
ProgressCallback = Callable[[int, int, str], None]
FINAL_BPS_FIELDS = [
    *OFFICIAL_FIELDS,
    "human_corrected",
    "is_repeated_measure",
]
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
    "written_measure",
    "performance_measure",
    "is_repeated_measure",
    "repeat_status",
    "volta_numbers",
    "repeat_source",
    "repeat_mapping_status",
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
        human_confirmed = source.get("match_source") == "human_confirmation"
        xml_time_confirmed = (
            str(source.get("xml_time_confirmed", "")).strip().lower() == "true"
        )
        confirmed = corrected or human_confirmed or (
            source.get("alignment_status") == "matched" and xml_time_confirmed
        )
        row = {
            field: "" if str(source.get(field, "")).strip().upper() == "NA"
            else str(source.get(field, ""))
            for field in OFFICIAL_FIELDS
        }
        if not confirmed:
            for field in FINAL_UNCERTAIN_FIELDS:
                row[field] = ""
        if not str(row.get("class", "")).startswith("stem"):
            row["stem_dir"] = ""
        row["human_corrected"] = "1" if corrected else "0"
        repeated = str(source.get("is_repeated_measure", "")).strip().lower()
        if repeated not in {"1", "true", "yes"}:
            try:
                repeated = (
                    "1"
                    if int(str(source.get("repeat_occurrence_count", "1") or "1")) > 1
                    else "0"
                )
            except ValueError:
                repeated = "0"
        else:
            repeated = "1"
        row["is_repeated_measure"] = repeated
        output.append({field: row.get(field, "") for field in FINAL_BPS_FIELDS})
    return output


def validate_final_bps_rows(
    rows: list[dict[str, str]],
    *,
    expected_count: int | None = None,
) -> list[str]:
    """Validate final rows against the BPS-OMR bounding-box field semantics."""

    errors: list[str] = []
    if expected_count is not None and len(rows) != expected_count:
        errors.append(
            f"Final CSV row count {len(rows)} does not match YOLO count "
            f"{expected_count}"
        )

    def integer(value: object) -> bool:
        return str(value).strip().isdigit()

    for index, row in enumerate(rows, start=1):
        label = f"final row {index}"
        try:
            int(str(row.get("class_id", "")))
        except ValueError:
            errors.append(f"{label}: class_id must be an integer")
        if not str(row.get("class", "")).strip():
            errors.append(f"{label}: class must not be blank")

        for field in ("x", "y", "w", "h"):
            try:
                value = float(str(row.get(field, "")))
            except ValueError:
                errors.append(f"{label}: {field} must be numeric")
                continue
            if not 0 <= value <= 1:
                errors.append(f"{label}: {field} must be between 0 and 1")
            if field in {"w", "h"} and value <= 0:
                errors.append(f"{label}: {field} must be greater than 0")

        timeline = str(row.get("musical_time", "")).strip()
        if timeline not in {"", "0", "1"}:
            errors.append(f"{label}: musical_time must be 0, 1, or blank")
        expected_timeline = musical_time_for_class(row.get("class"))
        if expected_timeline is not None and timeline != str(expected_timeline):
            errors.append(
                f"{label}: musical_time for {row.get('class')} must be "
                f"{expected_timeline}"
            )

        start_text = str(row.get("start_meas", "")).strip()
        end_text = str(row.get("end_meas", "")).strip()
        if bool(start_text) != bool(end_text):
            errors.append(f"{label}: start_meas and end_meas must both be set or blank")
        elif start_text:
            try:
                start_time = float(start_text)
                end_time = float(end_text)
                if start_time > end_time:
                    errors.append(f"{label}: start_meas must not exceed end_meas")
            except ValueError:
                errors.append(f"{label}: start_meas and end_meas must be numeric")

        start_note = str(row.get("start_note", "")).strip()
        end_note = str(row.get("end_note", "")).strip()
        if bool(start_note) != bool(end_note):
            errors.append(f"{label}: start_note and end_note must both be set or blank")
        for field, value in (("start_note", start_note), ("end_note", end_note)):
            if value and not integer(value):
                errors.append(f"{label}: {field} must be an integer or blank")

        connected_text = str(row.get("connected_note", "")).strip()
        connected: list[str] = []
        if connected_text:
            try:
                decoded = json.loads(connected_text)
            except json.JSONDecodeError:
                errors.append(f"{label}: connected_note must be a JSON list or blank")
                decoded = []
            if not isinstance(decoded, list):
                errors.append(f"{label}: connected_note must be a JSON list or blank")
            else:
                connected = [str(value) for value in decoded]
                if any(not integer(value) for value in connected):
                    errors.append(
                        f"{label}: connected_note must contain only integer note IDs"
                    )
        if connected and not (start_note and end_note):
            errors.append(
                f"{label}: connected_note requires start_note and end_note"
            )
        if start_note and connected:
            if start_note not in connected or end_note not in connected:
                errors.append(
                    f"{label}: start_note and end_note must appear in connected_note"
                )
            if start_note == end_note and connected != [start_note]:
                errors.append(
                    f"{label}: equal start/end notes require one connected_note"
                )

        stem_dir = str(row.get("stem_dir", "")).strip()
        is_stem = str(row.get("class", "")).startswith("stem")
        if stem_dir and (not is_stem or stem_dir not in {"0", "1"}):
            errors.append(
                f"{label}: stem_dir must be blank except 0/1 on stem classes"
            )
        for field in ("human_corrected", "is_repeated_measure"):
            if str(row.get(field, "")).strip() not in {"0", "1"}:
                errors.append(f"{label}: {field} must be 0 or 1")
    return errors


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
    if subtype in {"slur", "tie", "tied", "wavy-line"}:
        kind = (
            "tie"
            if subtype in {"tie", "tied"}
            else "wavy-line"
            if subtype == "wavy-line"
            else "slur"
        )
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
    chord_members = index_chord_members(
        note_by_id.values(),
        lambda row: (
            (str(row.get("score_id", "")), str(row.get("chord_id", "")))
            if str(row.get("chord_id", "")).strip()
            else None
        ),
    )
    open_spans: dict[tuple[str, str, str, str], list[dict]] = {}
    seen_tie_markers: set[tuple[str, str, str]] = set()
    rows = []

    def anchor(event: dict) -> dict:
        ids = _json_value(event.get("anchor_xml_event_ids_json"), [])
        return note_by_id.get(ids[0], {}) if isinstance(ids, list) and ids else {}

    def anchor_chord(note: dict) -> list[dict]:
        chord_id = str(note.get("chord_id", "")).strip()
        if not chord_id:
            return [note]
        return chord_members.get((str(note.get("score_id", "")), chord_id), [note])

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
        elif kind in {"slur", "wavy-line"}:
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
        connected = endpoint_note_ids(
            start_anchor,
            end_anchor,
            start_members=anchor_chord(start_anchor),
            end_members=anchor_chord(end_anchor),
            expand_chords=kind == "slur",
        )
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
                        "written_measure": occurrence.get(
                            "written_measure",
                            source.get("written_measure", source.get("xml_measure", "")),
                        ),
                        "performance_measure": occurrence.get(
                            "performance_measure",
                            occurrence.get("unfolded_measure_index", ""),
                        ),
                        "is_repeated_measure": occurrence.get(
                            "is_repeated_measure",
                            source.get("is_repeated_measure", ""),
                        ),
                        "repeat_status": occurrence.get(
                            "repeat_status", source.get("repeat_status", "none")
                        ),
                        "volta_numbers": occurrence.get(
                            "volta_numbers", source.get("volta_numbers", "[]")
                        ),
                        "repeat_source": occurrence.get(
                            "repeat_source", source.get("repeat_source", "")
                        ),
                        "repeat_mapping_status": occurrence.get(
                            "mapping_status",
                            source.get("repeat_mapping_status", ""),
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
                    "xml_spans_csv",
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
                "xml_spans_csv": saved_paths[4],
            }
    repeat_dir = output_dir / "repeat_mapping"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    _report_progress(0, 2, "Preparing shared repeat mapping", progress_callback)
    repeat_report = build_repeat_mapping(xml_path, unfolded_xml_path)
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
    spans_path = xml_dir / "xml_spans.csv"
    spans = build_xml_spans(events, spans_path)
    warnings = []
    warnings.extend(repeat_report.get("structural_warnings", []))
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
        "identity_repeat_mapping": repeat_report.get("repeat_group_count", 0) == 0,
        "repeat_mapping_source": repeat_report.get(
            "repeat_source", "repetition_musicxml"
        ),
        "unfolded_validation": repeat_report.get("unfolded_validation", {}),
        "xml_node_rows": len(nodes),
        "xml_event_rows": len(events),
        "xml_span_rows": len(spans),
        "cross_page_xml_span_rows": sum(
            str(row.get("cross_page", "")).lower() == "true" for row in spans
        ),
        "bps_note_count": len(bps_notes),
        "bps_note_ids": [str(note["note_id"]) for note in bps_notes],
        "warnings": warnings,
        "validation_errors": errors,
        "passed": not errors,
        "repeat_csv": repeat_csv,
        "repeat_json": repeat_json,
        "xml_nodes_csv": nodes_path,
        "xml_events_csv": events_path,
        "xml_spans_csv": spans_path,
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
        occurrences = _json_value(detailed.get("repeat_occurrences_json"), [])
        first_occurrence = (
            occurrences[0]
            if isinstance(occurrences, list)
            and occurrences
            and isinstance(occurrences[0], dict)
            else {}
        )
        occurrence_count = detailed.get("repeat_occurrence_count", "")
        try:
            repeated = int(str(occurrence_count or "1")) > 1
        except ValueError:
            repeated = bool(first_occurrence.get("is_repeated_measure", False))
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
                "written_measure": first_occurrence.get(
                    "written_measure",
                    detailed.get(
                        "start_xml_measure", detailed.get("xml_measure", "")
                    ),
                ),
                "is_repeated_measure": "1" if repeated else "0",
                "repeat_occurrences_json": detailed.get("repeat_occurrences_json", ""),
                "repeat_occurrence_count": occurrence_count or "1",
                "repeat_group_id": detailed.get("repeat_group_id", ""),
                "repeat_status": first_occurrence.get("repeat_status", "none"),
                "volta_numbers": first_occurrence.get("volta_numbers", "[]"),
                "repeat_source": first_occurrence.get(
                    "repeat_source", "repetition_musicxml"
                ),
                "repeat_mapping_status": detailed.get(
                    "repeat_mapping_status",
                    first_occurrence.get("mapping_status", ""),
                ),
                "movement_scope_status": "in_bpsd_scope",
                "page_mapping_status": "direct",
                "mapping_source": "web_uploaded_page_number",
                "match_source": detailed.get("match_source", ""),
                "confidence": detailed.get("confidence", ""),
                "match_score": detailed.get(
                    "match_score", detailed.get("confidence", "")
                ),
                "confidence_calibrated": detailed.get(
                    "confidence_calibrated", "false"
                ),
                "geometry_score": detailed.get("geometry_score", ""),
                "candidate_margin": detailed.get("candidate_margin", ""),
                "count_agreement": detailed.get("count_agreement", ""),
                "xml_time_confirmed": detailed.get("xml_time_confirmed", ""),
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


def finalize_uploaded_batch(
    *,
    score_id: str,
    pages: list[int],
    page_reports: list[dict],
    final_entries: list[tuple[int, int, dict]],
    yolo_entries: list[tuple[int, int, dict]],
    detailed_rows: list[dict],
    xml_events: list[dict],
    xml_nodes: list[dict],
    output_dir: Path,
    overlays: dict,
    review_candidate_rows: list[dict] | None = None,
    review_candidate_set_rows: list[dict] | None = None,
) -> dict:
    """Build identical durable outputs for foreground and background jobs."""

    from bpsd_aligner.review_candidates import (
        CANDIDATE_FIELDS,
        CANDIDATE_SET_FIELDS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    def entry_sort_key(entry: tuple[int, int, dict]) -> tuple[float, int, int]:
        page, row_index, row = entry
        try:
            start = float(row.get("start_meas", ""))
        except (TypeError, ValueError):
            start = float("inf")
        return start, page, row_index

    final_rows = [entry[2] for entry in sorted(final_entries, key=entry_sort_key)]
    yolo_rows = [entry[2] for entry in sorted(yolo_entries, key=entry_sort_key)]
    final_path = output_dir / "bps_omr_final.csv"
    detailed_path = output_dir / "page_alignment_detailed.csv"
    candidates_path = output_dir / "review_note_candidates.csv"
    candidate_sets_path = output_dir / "review_candidate_sets.csv"
    atomic_write_csv(final_path, FINAL_BPS_FIELDS, final_rows)
    disk_detailed_rows = []
    for row in detailed_rows:
        prepared = dict(row)
        prepared["review_note_candidates_json"] = ""
        disk_detailed_rows.append(prepared)
    detailed_fields = list(
        dict.fromkeys(field for row in disk_detailed_rows for field in row)
    )
    atomic_write_csv(detailed_path, detailed_fields, disk_detailed_rows)
    atomic_write_csv(
        candidates_path, CANDIDATE_FIELDS, review_candidate_rows or []
    )
    atomic_write_csv(
        candidate_sets_path,
        CANDIDATE_SET_FIELDS,
        review_candidate_set_rows or [],
    )
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
        "score_id": score_id,
        "page_count": len(pages),
        "pages": pages,
        "alignment_rows": len(final_rows),
        "xml_event_rows": complete["xml_event_rows"],
        "xml_node_rows": complete["xml_node_rows"],
        "timeline_rows": complete["timeline_rows"],
        "all_information_rows": complete["all_information_rows"],
        "xml_span_rows": complete["xml_span_rows"],
        "performance_expanded_rows": complete["performance_expanded_rows"],
        "confirmed_rows": counts.get("matched", 0),
        "human_corrected_rows": sum(
            row.get("human_corrected") == "1" for row in final_rows
        ),
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
        for name, value in overlays.items():
            if isinstance(value, (str, Path)):
                archive.write(value, arcname=f"review_images/{name}.png")
            else:
                archive.writestr(f"review_images/{name}.png", value)
    return {
        "report": report,
        "final_rows": final_rows,
        "yolo_rows": yolo_rows,
        "complete": complete,
        "final_path": final_path,
        "detailed_path": detailed_path,
        "review_candidates_path": candidates_path,
        "review_candidate_sets_path": candidate_sets_path,
        "report_path": report_path,
        "zip_path": zip_path,
    }


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
    render_qa_images: bool = True,
    render_class_overlays: bool = True,
    render_auxiliary_overlays: bool = True,
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

    _report_progress(1, 6, "Parsing MusicXML repeat structure", progress_callback)
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
        whole_score_spans=_read_csv(Path(prepared_score["xml_spans_csv"])),
        render_qa_images=render_qa_images,
        render_class_overlays=render_class_overlays,
        render_auxiliary_overlays=render_auxiliary_overlays,
    )
    clean_reference = alignment_report.get("clean_reference", {})
    if clean_image_path is not None and not clean_reference.get("used"):
        warnings.append(
            "The clean repetition PDF page could not be matched structurally; "
            "alignment fell back to MusicXML-width geometry for this page."
        )
    detailed_path = Path(alignment_report["outputs"]["detailed_csv"])
    detailed_rows = _read_csv(detailed_path)
    unknown_timeline_classes = sorted(
        {
            str(row.get("class", ""))
            for row in detailed_rows
            if musical_time_for_class(row.get("class")) is None
        }
        - {""}
    )
    if unknown_timeline_classes:
        warnings.append(
            "musical_time remains blank for context-dependent or unknown classes: "
            + ", ".join(unknown_timeline_classes)
        )
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
    final_bps_rows = build_final_bps_rows(master_rows)
    atomic_write_csv(yolo_master_path, FIELDS, master_rows)
    atomic_write_csv(yolo_aligned_path, FIELDS, master_rows)
    atomic_write_csv(
        final_bps_path,
        FINAL_BPS_FIELDS,
        final_bps_rows,
    )
    errors.extend(
        validate_final_bps_rows(final_bps_rows, expected_count=len(master_rows))
    )
    atomic_write_csv(
        review_queue_path,
        FIELDS,
        [row for row in master_rows if row["alignment_status"] != "matched"],
    )

    _report_progress(3, 6, "Exporting every MusicXML node and event", progress_callback)
    xml_nodes_path = Path(prepared_score["xml_nodes_csv"])
    xml_events_path = Path(prepared_score["xml_events_csv"])
    xml_spans_path = Path(prepared_score["xml_spans_csv"])
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
        "review_note_candidates_csv": Path(
            alignment_report["outputs"]["review_note_candidates_csv"]
        ),
        "review_candidate_sets_csv": Path(
            alignment_report["outputs"]["review_candidate_sets_csv"]
        ),
        "final_bps_csv": final_bps_path,
        "yolo_aligned_csv": yolo_aligned_path,
        "review_queue_csv": review_queue_path,
        "yolo_master_csv": yolo_master_path,
        "xml_nodes_csv": xml_nodes_path,
        "xml_events_csv": xml_events_path,
        "xml_spans_csv": xml_spans_path,
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
        "final_rows_with_blank_time": sum(
            not row.get("start_meas") or not row.get("end_meas")
            for row in final_bps_rows
        ),
        "unknown_musical_time_classes": unknown_timeline_classes,
        "yolo_rows_with_time": len(master_rows) - len(missing_time_lines),
        "yolo_rows_needing_review": sum(
            row.get("status") != "matched" for row in detailed_rows
        ),
        "yolo_status_counts": dict(
            Counter(row.get("status", "") for row in detailed_rows)
        ),
        "class_overlay_count": alignment_report.get("class_overlay_count", 0),
        "class_overlay_available_count": alignment_report.get(
            "class_overlay_available_count", 0
        ),
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
