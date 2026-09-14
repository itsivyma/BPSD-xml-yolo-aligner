import json

from bpsd_aligner.review_workspace import (
    certain_endpoint_note_ids,
    endpoint_note_input_values,
    fill_missing_note_orders,
    merge_page_note_candidates,
    resolve_endpoint_note_input,
    resolve_endpoint_note_inputs,
    snap_click_to_note_candidate,
    toggle_endpoint_note_input,
    toggle_endpoint_selection,
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


def test_end_endpoint_can_select_multiple_notes_without_overwriting_start():
    start_candidates = [
        {
            "candidate_id": "start-c",
            "printed_measure": 10,
            "start_meas": 9.0,
            "staff": 1,
            "pitch": "C4",
            "measure_note_order": 1,
            "note_id": 10,
        }
    ]
    end_candidates = [
        {
            "candidate_id": f"end-{index}",
            "printed_measure": 11,
            "start_meas": 10.0,
            "staff": 1,
            "pitch": pitch,
            "measure_note_order": index,
            "note_id": 20 + index,
        }
        for index, pitch in enumerate(("D4", "F4", "A4"), start=1)
    ]
    start_value = endpoint_note_input_values(start_candidates)
    end_value = ""

    for candidate in end_candidates:
        start_value, end_value, added = toggle_endpoint_selection(
            role="end",
            start_value=start_value,
            end_value=end_value,
            candidate=candidate,
            start_candidates=start_candidates,
            end_candidates=end_candidates,
        )
        assert added is True

    resolved_start, start_error = resolve_endpoint_note_inputs(
        start_value, start_candidates
    )
    resolved_end, end_error = resolve_endpoint_note_inputs(end_value, end_candidates)
    assert start_error == end_error == ""
    assert [candidate["note_id"] for candidate in resolved_start] == [10]
    assert [candidate["note_id"] for candidate in resolved_end] == [21, 22, 23]

    start_value, end_value, added = toggle_endpoint_selection(
        role="end",
        start_value=start_value,
        end_value=end_value,
        candidate=end_candidates[1],
        start_candidates=start_candidates,
        end_candidates=end_candidates,
    )
    assert added is False
    resolved_end, end_error = resolve_endpoint_note_inputs(end_value, end_candidates)
    assert end_error == ""
    assert [candidate["note_id"] for candidate in resolved_end] == [21, 23]


def test_start_endpoint_selection_never_populates_or_overwrites_end():
    start_candidate = {
        "printed_measure": 10,
        "start_meas": 9.0,
        "staff": 1,
        "pitch": "C4",
        "measure_note_order": 1,
        "note_id": 10,
    }
    end_candidate = {
        "printed_measure": 11,
        "start_meas": 10.0,
        "staff": 1,
        "pitch": "G4",
        "measure_note_order": 1,
        "note_id": 20,
    }
    original_end = endpoint_note_input_values([end_candidate])

    start_value, end_value, added = toggle_endpoint_selection(
        role="start",
        start_value="",
        end_value=original_end,
        candidate=start_candidate,
        start_candidates=[start_candidate],
        end_candidates=[end_candidate],
    )

    assert added is True
    assert start_value == endpoint_note_input_values([start_candidate])
    assert end_value == original_end


def test_uncertain_endpoint_note_ids_stay_blank():
    exact = {"note_id": 10, "note_id_ambiguous": False}
    ambiguous = {"note_id": 11, "note_id_ambiguous": True}
    xml_only = {"note_id": None, "note_id_ambiguous": False}

    assert certain_endpoint_note_ids([exact]) == (["10"], False)
    assert certain_endpoint_note_ids([exact, ambiguous]) == ([], True)
    assert certain_endpoint_note_ids([xml_only]) == ([], True)
