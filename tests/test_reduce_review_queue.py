import csv
from pathlib import Path

from build_review_queue import QUEUE_FIELDS
from reduce_review_queue import reduce_review_queue


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _queue_row(
    bbox_id: str,
    status: str,
    confidence: str,
    event_ids: str,
    class_name: str = "slur",
) -> dict:
    row = {field: "" for field in QUEUE_FIELDS}
    row.update(
        {
            "score_id": "score",
            "page_id": "score-01",
            "bbox_id": bbox_id,
            "class": class_name,
            "source_status": status,
            "confidence": confidence,
            "xml_event_ids_json": event_ids,
            "asset_status": "ready",
            "review_png": "/tmp/review.png",
            "review_csv": "/tmp/review.csv",
            "batch_id": "score-01:fingering1:part01",
        }
    )
    return row


def test_reduce_queue_proposes_only_strict_candidates_and_resumes(tmp_path: Path):
    queue_path = tmp_path / "queue.csv"
    queue_rows = [
        _queue_row("score-01:Y1", "candidate", "0.960", '["score:E1","score:E1"]'),
        _queue_row("score-01:Y2", "candidate", "0.900", '["score:E2"]'),
        _queue_row("score-01:Y3", "ambiguous", "0.990", '["score:E2"]'),
    ]
    _write(queue_path, QUEUE_FIELDS, queue_rows)

    master_fields = [
        "row_origin", "bbox_id", "page_mapping_status",
        "movement_scope_status", "error_code",
    ]
    master_rows = [
        {
            "row_origin": "yolo",
            "bbox_id": row["bbox_id"],
            "page_mapping_status": "direct",
            "movement_scope_status": "in_bpsd_scope",
            "error_code": "",
        }
        for row in queue_rows
    ]
    master_path = tmp_path / "master.csv"
    _write(master_path, master_fields, master_rows)

    links_path = tmp_path / "links.csv"
    _write(
        links_path,
        ["bbox_id", "xml_event_id", "is_primary"],
        [
            {"bbox_id": "score-01:Y1", "xml_event_id": "score:E1", "is_primary": "true"},
            {"bbox_id": "score-01:Y2", "xml_event_id": "score:E2", "is_primary": "true"},
            {"bbox_id": "score-01:Y3", "xml_event_id": "score:E2", "is_primary": "true"},
        ],
    )

    output = tmp_path / "reduced"
    first = reduce_review_queue(
        queue_path, master_path, links_path, output, threshold=0.95, resume=True
    )
    second = reduce_review_queue(
        queue_path, master_path, links_path, output, threshold=0.95, resume=True
    )

    assert first["passed"] is True
    assert first["auto_accept_proposal_rows"] == 1
    assert first["audit_sample_rows"] == 1
    assert first["manual_review_rows"] == 2
    assert first["manual_review_groups"] == 1
    assert first["master_modified"] is False
    assert first["score_reports"][0]["resumed"] is False
    assert second["score_reports"][0]["resumed"] is True


def test_reduce_queue_never_auto_accepts_uncalibrated_fingering(tmp_path: Path):
    queue_path = tmp_path / "queue.csv"
    row = _queue_row(
        "score-01:Y1", "candidate", "0.999", '["score:E1"]', "fingering1"
    )
    _write(queue_path, QUEUE_FIELDS, [row])
    master_path = tmp_path / "master.csv"
    _write(
        master_path,
        ["row_origin", "bbox_id", "page_mapping_status", "movement_scope_status", "error_code"],
        [{
            "row_origin": "yolo", "bbox_id": row["bbox_id"],
            "page_mapping_status": "direct", "movement_scope_status": "in_bpsd_scope",
            "error_code": "",
        }],
    )
    links_path = tmp_path / "links.csv"
    _write(
        links_path,
        ["bbox_id", "xml_event_id", "is_primary"],
        [{"bbox_id": row["bbox_id"], "xml_event_id": "score:E1", "is_primary": "true"}],
    )

    report = reduce_review_queue(
        queue_path, master_path, links_path, tmp_path / "out", threshold=0.95
    )

    assert report["passed"] is True
    assert report["auto_accept_proposal_rows"] == 0
    assert report["manual_review_rows"] == 1
    assert report["blocked_class_source_rows"] == 1
