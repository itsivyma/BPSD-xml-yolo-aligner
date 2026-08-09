import csv
import json
from pathlib import Path

from combine_yolo_xml import (
    build_score_combined,
    build_score_links,
    combine_dataset,
)
from xml_export import BPS_FIELDS, EVENT_FIELDS


YOLO_EXTRA_FIELDS = [
    "score_id", "page_id", "yolo_line", "bbox_id", "class",
    "alignment_status", "human_approved", "review_status", "confidence",
    "note_ids", "connected_note", "start_note", "end_note", "target_type",
    "xml_page", "xml_systems_json", "xml_measure", "xml_symbol", "staff",
    "xml_path",
]
YOLO_FIELDS = BPS_FIELDS + [field for field in YOLO_EXTRA_FIELDS if field not in BPS_FIELDS]


def _event(event_id: str, event_type: str, subtype: str, **values) -> dict:
    row = {field: "" for field in EVENT_FIELDS}
    row.update(
        {
            "record_id": event_id.replace(":E", ":R"),
            "row_origin": "xml_only",
            "score_id": "score",
            "xml_event_id": event_id,
            "xml_node_id": event_id.replace(":E", ":N"),
            "event_type": event_type,
            "event_subtype": subtype,
            "page": "1",
            "xml_measure": "1",
            "staff": "1",
            "repeat_json": "[]",
            "anchor_xml_event_ids_json": "",
            "xml_attributes_json": "{}",
        }
    )
    row.update(values)
    return row


def _yolo(bbox_id: str, class_name: str, **values) -> dict:
    row = {field: "" for field in YOLO_FIELDS}
    row.update(
        {
            "score_id": "score",
            "page_id": "score-01",
            "yolo_line": bbox_id.rsplit("Y", 1)[-1],
            "bbox_id": bbox_id,
            "class": class_name,
            "alignment_status": "matched",
            "human_approved": "true",
            "review_status": "confirmed",
            "confidence": "1.000",
            "xml_page": "1",
            "xml_systems_json": "[]",
        }
    )
    row.update(values)
    return row


def _fixture_rows():
    note = _event(
        "score:E0000001",
        "note",
        "note",
        pitch="60",
        pitchName="C4",
        start_meas="1.000000",
        repeat_json=json.dumps(
            [{"note_id": 0, "note_match_status": "exact", "start_meas": 1.0}]
        ),
    )
    notation = _event(
        "score:E0000002",
        "notation",
        "staccato",
        anchor_xml_event_ids_json=json.dumps([note["xml_event_id"]]),
        xml_attributes_json=json.dumps({"placement": "above"}),
    )
    dynamic = _event(
        "score:E0000003",
        "direction",
        "dynamic",
        dynamic="f",
        start_meas="1.000000",
    )
    rest = _event("score:E0000004", "rest", "rest")
    fermata = _event(
        "score:E0000005",
        "notation",
        "fermata",
        anchor_xml_event_ids_json=json.dumps([rest["xml_event_id"]]),
    )
    attribute = _event("score:E0000006", "attribute", "clef")
    yolo_rows = [
        _yolo(
            "score-01:Y1",
            "articStaccatoAbove",
            note_ids="[0]",
            connected_note="[0]",
            start_note="0",
            end_note="0",
            target_type="note",
            start_meas="1.000",
        ),
        _yolo(
            "score-01:Y2",
            "dynamicF",
            note_ids="[]",
            connected_note="NA",
            target_type="measure_position",
            xml_measure="1",
            xml_symbol="f",
            start_meas="1.000",
        ),
        _yolo(
            "score-01:Y3",
            "fermataAbove",
            note_ids="[]",
            connected_note="NA",
            target_type="rest",
            xml_measure="1",
            staff="1",
            start_meas="1.000",
        ),
    ]
    return yolo_rows, [note, notation, dynamic, rest, fermata, attribute]


def test_build_links_and_combined_preserve_both_sources():
    yolo_rows, events = _fixture_rows()
    links = build_score_links(yolo_rows, events)

    assert len(links) == 5
    assert {link["link_method"] for link in links} == {
        "bps_note_anchor",
        "xml_notation_anchor",
        "xml_direction_measure_symbol",
        "xml_rest_anchor",
    }
    assert len({link["link_id"] for link in links}) == len(links)

    fields, combined = build_score_combined(YOLO_FIELDS, yolo_rows, events, links)
    assert fields[: len(BPS_FIELDS)] == BPS_FIELDS
    assert len([row for row in combined if row["row_origin"] == "yolo"]) == 3
    xml_only = [row for row in combined if row["row_origin"] == "xml"]
    assert [row["event_xml_event_id"] for row in xml_only] == ["score:E0000006"]
    assert all(row["combined_status"] == "aligned" for row in combined[:3])


def test_combine_dataset_checkpoint_resume(tmp_path: Path):
    yolo_rows, events = _fixture_rows()
    yolo_path = tmp_path / "yolo.csv"
    xml_path = tmp_path / "xml.csv"
    with yolo_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=YOLO_FIELDS)
        writer.writeheader()
        writer.writerows(yolo_rows)
    with xml_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    output = tmp_path / "combined"
    first = combine_dataset(yolo_path, xml_path, output, resume=True)
    second = combine_dataset(yolo_path, xml_path, output, resume=True)

    assert first["passed"] is True
    assert first["score_reports"][0]["resumed"] is False
    assert second["score_reports"][0]["resumed"] is True
    assert first["yolo_source_rows"] == 3
    assert first["xml_events_represented"] == 6
    assert (output / "alignment_links.csv").is_file()
    assert (output / "combined_master.csv").is_file()
