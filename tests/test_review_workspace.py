import json

from bpsd_aligner.review_workspace import (
    endpoint_note_input_values,
    fill_missing_note_orders,
    merge_page_note_candidates,
    resolve_endpoint_note_input,
    resolve_endpoint_note_inputs,
    snap_click_to_note_candidate,
    toggle_endpoint_note_input,
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


def test_endpoint_chord_can_select_and_toggle_multiple_noteheads():
    candidates = [
        {
            "candidate_id": f"c{index}",
            "printed_measure": 52,
            "start_meas": 51.0,
            "staff": 1,
            "pitch": pitch,
            "measure_note_order": index,
            "note_id": 100 + index,
        }
        for index, pitch in enumerate(("C4", "E4", "G4"), start=1)
    ]

    value = ""
    for candidate in candidates:
        value, added = toggle_endpoint_note_input(value, candidate, candidates)
        assert added is True

    resolved, error = resolve_endpoint_note_inputs(value, candidates)
    assert error == ""
    assert [candidate["note_id"] for candidate in resolved] == [101, 102, 103]
    assert value == endpoint_note_input_values(candidates)

    value, added = toggle_endpoint_note_input(value, candidates[1], candidates)
    assert added is False
    resolved, error = resolve_endpoint_note_inputs(value, candidates)
    assert error == ""
    assert [candidate["note_id"] for candidate in resolved] == [101, 103]


def test_endpoint_chord_rejects_notes_from_different_onsets():
    candidates = [
        {
            "printed_measure": 52,
            "start_meas": start,
            "staff": 1,
            "pitch": pitch,
            "measure_note_order": index,
            "note_id": index,
        }
        for index, (start, pitch) in enumerate(((51.0, "C4"), (51.5, "E4")), 1)
    ]
    value = endpoint_note_input_values(candidates)

    resolved, error = resolve_endpoint_note_inputs(value, candidates)

    assert resolved == []
    assert "同一開始時間" in error
