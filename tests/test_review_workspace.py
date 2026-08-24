import json

from bpsd_aligner.review_workspace import (
    fill_missing_note_orders,
    merge_page_note_candidates,
    resolve_endpoint_note_input,
    snap_click_to_note_candidate,
)


def test_review_note_input_and_click_resolution_are_pure():
    candidates = [
        {
            "page_id": "page-1",
            "printed_measure": 52,
            "staff": 2,
            "pitch": "E3",
            "x_px": 120,
            "y_px": 80,
            "note_id": 10,
        }
    ]
    fill_missing_note_orders(candidates)

    resolved, error = resolve_endpoint_note_input("52, 下, E3, 1", candidates)
    assert error == ""
    assert resolved["note_id"] == 10

    clicked, distance = snap_click_to_note_candidate(
        {"x": 20, "y": 30, "width": 100, "height": 100},
        candidates,
        {"left": 100, "top": 50, "width": 100, "height": 100},
    )
    assert clicked["note_id"] == 10
    assert distance == 0


def test_merge_page_candidates_deduplicates_legacy_rows():
    candidate = {
        "xml_note_sequence": 7,
        "note_id": 10,
        "x_px": 20,
        "y_px": 30,
    }
    rows = [
        {
            "page_id": "page-1",
            "review_note_candidates_json": json.dumps([candidate]),
        },
        {
            "page_id": "page-1",
            "review_note_candidates_json": json.dumps([candidate]),
        },
    ]

    merged = merge_page_note_candidates([], rows, "page-1")

    assert len(merged) == 1
    assert merged[0]["page_id"] == "page-1"
