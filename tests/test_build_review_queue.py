import csv
from pathlib import Path

from build_review_queue import build_review_queue


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_review_queue_uses_existing_assets_and_resumes(tmp_path: Path):
    combined_fields = [
        "row_origin", "combined_status", "score_id", "page_id", "yolo_line",
        "bbox_id", "class_id", "class", "confidence", "start_meas",
        "end_meas", "start_note", "end_note", "connected_note", "note_ids",
        "pitches", "xml_event_ids_json", "xml_event_count", "review_status",
        "human_approved",
    ]
    combined_rows = [
        {
            "row_origin": "yolo", "combined_status": "candidate",
            "score_id": "score", "page_id": "score-01", "yolo_line": "2",
            "bbox_id": "score-01:Y2", "class_id": "1", "class": "fingering1",
            "confidence": "0.800", "xml_event_ids_json": "[]",
            "xml_event_count": "0", "review_status": "needs_review",
            "human_approved": "false",
        },
        {
            "row_origin": "yolo", "combined_status": "ambiguous",
            "score_id": "score", "page_id": "score-01", "yolo_line": "1",
            "bbox_id": "score-01:Y1", "class_id": "1", "class": "fingering1",
            "confidence": "0.600", "xml_event_ids_json": "[\"score:E1\"]",
            "xml_event_count": "1", "review_status": "needs_review",
            "human_approved": "false",
        },
        {
            "row_origin": "xml", "combined_status": "xml_only",
            "score_id": "score", "bbox_id": "", "class": "",
        },
    ]
    combined = tmp_path / "combined.csv"
    _write_csv(combined, combined_fields, combined_rows)
    links = tmp_path / "links.csv"
    _write_csv(
        links,
        ["bbox_id", "link_method"],
        [{"bbox_id": "score-01:Y1", "link_method": "bps_note_anchor"}],
    )

    review_dir = tmp_path / "review"
    sheet_csv = review_dir / "score-01" / "sheet.csv"
    sheet_png = review_dir / "score-01" / "sheet.png"
    _write_csv(
        sheet_csv,
        ["bbox_id"],
        [{"bbox_id": "score-01:Y1"}, {"bbox_id": "score-01:Y2"}],
    )
    sheet_png.write_bytes(b"existing review image")
    _write_csv(
        review_dir / "review_sheet_index.csv",
        ["page_id", "class", "part", "candidates", "png", "csv"],
        [{
            "page_id": "score-01", "class": "fingering1", "part": "1",
            "candidates": "2", "png": "score-01/sheet.png",
            "csv": "score-01/sheet.csv",
        }],
    )

    output = tmp_path / "output"
    first = build_review_queue(combined, links, review_dir, output, resume=True)
    second = build_review_queue(combined, links, review_dir, output, resume=True)

    assert first["passed"] is True
    assert first["queue_rows"] == 2
    assert first["batch_rows"] == 1
    assert first["score_reports"][0]["resumed"] is False
    assert second["score_reports"][0]["resumed"] is True
    with (output / "review_queue.csv").open(newline="", encoding="utf-8") as file:
        queue = list(csv.DictReader(file))
    assert [row["source_status"] for row in queue] == ["ambiguous", "candidate"]
    assert [row["priority_rank"] for row in queue] == ["1", "2"]
    assert all(row["asset_status"] == "ready" for row in queue)
