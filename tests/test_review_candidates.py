import csv
import json

from bpsd_aligner.review_candidates import (
    hydrate_review_candidates,
    normalize_review_candidates,
)


def test_review_candidates_are_stored_once_and_hydrated(tmp_path):
    shared = {
        "note_id": 10,
        "xml_note_sequence": 5,
        "pitch": "C4",
        "xml_measure": 3,
        "staff": 1,
        "x_px": 100.0,
        "y_px": 200.0,
        "distance_px": 12.0,
    }
    rows = [
        {"txt_line": "1", "review_note_candidates_json": json.dumps([shared])},
        {
            "txt_line": "2",
            "review_note_candidates_json": json.dumps(
                [{**shared, "distance_px": 42.0}]
            ),
        },
    ]

    candidates_path, sets_path = normalize_review_candidates(
        rows, page_id="page-01", output_dir=tmp_path
    )

    with candidates_path.open(newline="", encoding="utf-8-sig") as file:
        candidates = list(csv.DictReader(file))
    assert len(candidates) == 1
    assert "distance_px" not in json.loads(candidates[0]["candidate_json"])
    assert all(row["review_note_candidates_json"] == "" for row in rows)
    assert rows[0]["review_candidate_set_id"] != rows[1]["review_candidate_set_id"]

    hydrate_review_candidates(rows, candidates_path, sets_path)

    assert json.loads(rows[0]["review_note_candidates_json"])[0]["note_id"] == 10
    assert json.loads(rows[1]["review_note_candidates_json"])[0]["pitch"] == "C4"
