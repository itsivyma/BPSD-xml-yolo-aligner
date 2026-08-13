import io
import json

from PIL import Image

from bpsd_aligner.review_corrections import (
    apply_review_decisions,
    build_review_checkpoint,
    build_editor_rows,
    build_review_queue,
    evaluate_ground_truth_rows,
    load_review_checkpoint,
    normalize_legacy_review_rows,
    render_review_focus_images,
)


def _detail(line: int, class_name: str, status: str, start: str, end: str) -> dict:
    return {
        "page_id": "page-01",
        "txt_line": str(line),
        "class_id": "56",
        "x": "0.5",
        "y": "0.5",
        "w": "0.1",
        "h": "0.02",
        "class": class_name,
        "musical_time": "0",
        "start_meas": start,
        "end_meas": end,
        "start_note": "10",
        "end_note": "20",
        "connected_note": "[10, 20]",
        "stem_dir": "NA",
        "xml_staff": "1",
        "status": status,
        "match_source": "machine",
    }


def test_editor_defaults_to_review_rows_and_can_include_matched():
    details = [
        _detail(1, "slur", "review", "1.0", "2.0"),
        _detail(2, "tie", "matched", "3.0", "4.0"),
    ]

    review_only = build_editor_rows(details)
    all_rows = build_editor_rows(details, include_matched=True)

    assert [row["yolo_line"] for row in review_only] == ["1"]
    assert [row["yolo_line"] for row in all_rows] == ["1", "2"]
    assert review_only[0]["action"] == "pending"


def test_apply_confirm_correct_and_reject_produces_strict_outputs():
    details = [
        _detail(1, "slur", "review", "1.0", "2.0"),
        _detail(2, "tie", "review", "3.0", "4.0"),
        _detail(3, "fermataAbove", "review", "5.0", "5.0"),
    ]
    editor = build_editor_rows(details)
    editor[0]["action"] = "confirm"
    editor[1].update(
        {
            "action": "correct",
            "start_meas": "3.5",
            "end_meas": "4.5",
            "start_note": "30",
            "end_note": "40",
            "connected_note": "[30, 40]",
            "staff": "2",
            "comment": "endpoint corrected",
        }
    )
    editor[2]["action"] = "reject"

    final_rows, payload, accuracy, errors = apply_review_decisions(
        details, editor, reviewer="Ivy", reviewed_at="2026-08-12T00:00:00Z"
    )

    assert errors == []
    by_class = {row["class"]: row for row in final_rows}
    assert by_class["slur"]["start_meas"] == "1.0"
    assert by_class["slur"]["human_corrected"] == "0"
    assert by_class["tie"]["start_meas"] == "3.5"
    assert by_class["tie"]["end_note"] == "40"
    assert by_class["tie"]["human_corrected"] == "1"
    assert by_class["fermataAbove"]["start_meas"] == ""
    assert by_class["fermataAbove"]["start_note"] == ""
    assert by_class["fermataAbove"]["human_corrected"] == "0"
    assert payload["reviewer"] == "Ivy"
    assert [entry["action"] for entry in payload["entries"]] == [
        "confirm",
        "correct",
        "reject",
    ]
    assert accuracy["reviewed_rows"] == 3
    assert accuracy["by_class"]["slur"]["exact_accuracy"] == 1.0
    assert accuracy["by_class"]["tie"]["exact_accuracy"] == 0.0
    assert accuracy["by_class"]["fermataAbove"]["exact_accuracy"] == 0.0


def test_correct_requires_a_real_change_and_time_must_be_numeric():
    details = [_detail(1, "slur", "review", "1.0", "2.0")]
    editor = build_editor_rows(details)
    editor[0]["action"] = "correct"

    _rows, _payload, _report, errors = apply_review_decisions(
        details, editor, reviewer="Ivy"
    )

    assert any("no value was changed" in error for error in errors)

    editor[0]["start_meas"] = "not-a-time"
    _rows, _payload, _report, errors = apply_review_decisions(
        details, editor, reviewer="Ivy"
    )
    assert any("must be numeric or blank" in error for error in errors)


def test_review_validation_rejects_invalid_ranges_notes_staff_and_class_map():
    details = [_detail(1, "slur", "review", "1.0", "2.0")]
    editor = build_editor_rows(details)
    editor[0].update(
        {
            "action": "correct",
            "start_meas": "4.0",
            "end_meas": "3.0",
            "start_note": "99",
            "end_note": "2",
            "connected_note": "[2]",
            "staff": "3",
        }
    )
    _rows, _payload, _report, errors = apply_review_decisions(
        details,
        editor,
        reviewer="Ivy",
        valid_note_ids={"1", "2"},
    )
    assert any("start_meas must not exceed" in error for error in errors)
    assert any("note ID 99 does not exist" in error for error in errors)
    assert any("must appear in connected_note" in error for error in errors)
    assert any("staff must be 1, 2" in error for error in errors)

    editor[0].update(
        {
            "action": "wrong_class",
            "corrected_class_id": "57",
            "corrected_class": "tie",
            "start_meas": "1",
            "end_meas": "2",
            "start_note": "1",
            "end_note": "2",
            "connected_note": "[1, 2]",
            "staff": "1",
        }
    )
    _rows, _payload, _report, errors = apply_review_decisions(
        details,
        editor,
        reviewer="Ivy",
        valid_note_ids={"1", "2"},
        class_map={"57": "slur"},
    )
    assert any("maps to slur, not tie" in error for error in errors)


def test_normalize_legacy_reviews_handles_point_span_and_note_groups(tmp_path):
    point = tmp_path / "score_staccato_human_reviews.csv"
    point.write_text(
        "page_id,yolo_line,class_id,class,bps_time,note_id,staff,review_status,comment\n"
        "page-01,12,8,articStaccatoAbove,3.667,33,1,confirmed,ok\n",
        encoding="utf-8",
    )
    span = tmp_path / "score_slur_human_reviews.csv"
    span.write_text(
        "page_id,yolo_line,class_id,class,start_note_id,end_note_id,review_status,comment\n"
        "page-01,19,56,slur,9,12,confirmed,ok\n",
        encoding="utf-8",
    )
    tuplet = tmp_path / "score_tuplet_human_reviews.csv"
    tuplet.write_text(
        "page_id,yolo_line,class_id,class,start_bps_time,end_note_bps_time,note_ids,staff,review_status,comment\n"
        "page-01,105,111,tuplet6,30.333,30.611,236+237+238,1,confirmed,ok\n",
        encoding="utf-8",
    )

    rows, errors = normalize_legacy_review_rows([point, span, tuplet])

    assert errors == []
    by_line = {row["yolo_line"]: row for row in rows}
    assert by_line["12"]["expected_start_meas"] == "3.667"
    assert by_line["12"]["expected_start_note"] == "33"
    assert by_line["19"]["expected_start_note"] == "9"
    assert by_line["19"]["expected_end_note"] == "12"
    assert by_line["105"]["expected_end_meas"] == "30.611"
    assert json.loads(by_line["105"]["expected_connected_note"]) == [
        "236",
        "237",
        "238",
    ]


def test_evaluate_ground_truth_reports_fields_by_class_and_missing_rows():
    truth = [
        {
            "page_id": "page-01",
            "yolo_line": "1",
            "class": "slur",
            "expected_start_meas": "1.0",
            "expected_end_meas": "2.0",
            "expected_start_note": "10",
            "expected_end_note": "20",
            "expected_connected_note": "",
            "expected_staff": "1",
        },
        {
            "page_id": "page-01",
            "yolo_line": "99",
            "class": "tie",
        },
    ]
    prediction = [_detail(1, "slur", "matched", "1.0", "2.5")]

    report, errors = evaluate_ground_truth_rows(truth, prediction)

    assert report["ground_truth_rows"] == 2
    assert report["matched_prediction_rows"] == 1
    assert report["by_class"]["slur"]["fields"]["start_meas"]["accuracy"] == 1.0
    assert report["by_class"]["slur"]["fields"]["end_meas"]["accuracy"] == 0.0
    assert report["by_class"]["slur"]["fields"]["staff"]["accuracy"] == 1.0
    assert report["missing_predictions"] == ["page-01:Y99"]
    assert errors == ["missing predictions for 1 ground-truth rows"]


def test_review_queue_filters_and_prioritizes_unreviewed_low_confidence():
    first = _detail(1, "slur", "review", "1.0", "2.0")
    first["confidence"] = "0.70"
    second = _detail(2, "slur", "review", "3.0", "4.0")
    second["confidence"] = "0.30"
    third = _detail(3, "tie", "matched", "5.0", "6.0")

    queue = build_review_queue(
        [first, second, third],
        class_name="slur",
        decisions={"page-01:Y2": {"action": "confirm"}},
    )

    assert [row["review_key"] for row in queue] == ["page-01:Y1", "page-01:Y2"]
    assert queue[1]["saved_action"] == "confirm"
    assert [row["review_key"] for row in build_review_queue(
        [first, second, third], machine_status="matched", include_matched=True
    )] == ["page-01:Y3"]


def test_render_review_focus_images_keeps_native_pixels_and_highlights_bbox():
    source = Image.new("RGB", (1000, 800), "white")
    data = io.BytesIO()
    source.save(data, format="PNG")
    row = {
        "x": "0.5",
        "y": "0.5",
        "w": "0.1",
        "h": "0.1",
        "target_x_px": "550",
        "target_y_px": "450",
        "review_target_x_px": "520",
        "review_target_y_px": "430",
        "review_note_candidates_json": json.dumps(
            [{"x_px": 300, "y_px": 300, "note_id": 9}]
        ),
    }

    full_data, crop_data = render_review_focus_images(data.getvalue(), row)

    with Image.open(io.BytesIO(full_data)) as full:
        assert full.size == (1000, 800)
        assert full.getpixel((450, 360))[0] > 200
        assert full.getpixel((450, 360))[1] < 100
        target_pixel = full.getpixel((550, 450))
        assert target_pixel[1] > 100
        assert target_pixel[2] > 100
        candidate_outline = full.getpixel((290, 300))
        assert candidate_outline[0] > 200
        assert candidate_outline[1] > 60
        assert candidate_outline[2] < 80
    with Image.open(io.BytesIO(crop_data)) as crop:
        assert crop.width < 1000
        assert crop.height < 800
        assert crop.width >= 200

    _full, _crop, geometry = render_review_focus_images(
        data.getvalue(), row, include_crop_geometry=True
    )
    assert geometry["page_width"] == 1000
    assert geometry["page_height"] == 800
    assert geometry["width"] < geometry["page_width"]
    assert geometry["height"] < geometry["page_height"]


def test_workspace_special_decisions_change_corrected_output_conservatively():
    details = [
        _detail(1, "slur", "review", "1.0", "2.0"),
        _detail(2, "tie", "review", "3.0", "4.0"),
        _detail(3, "fermataAbove", "review", "5.0", "5.0"),
    ]
    editor = build_editor_rows(details)
    editor[0]["action"] = "scan_only"
    editor[1].update(
        {
            "action": "wrong_class",
            "corrected_class_id": "57",
            "corrected_class": "slur",
        }
    )
    editor[2]["action"] = "not_a_symbol"

    final_rows, payload, accuracy, errors = apply_review_decisions(
        details, editor, reviewer="Ivy"
    )

    assert errors == []
    assert len(final_rows) == 2
    by_class = {row["class"]: row for row in final_rows}
    assert by_class["slur"]["start_meas"] == ""
    corrected_class_row = next(row for row in final_rows if row["human_corrected"] == "1")
    assert corrected_class_row["class"] == "slur"
    assert corrected_class_row["class_id"] == "57"
    assert corrected_class_row["start_note"] == ""
    assert [entry["action"] for entry in payload["entries"]] == [
        "scan_only",
        "wrong_class",
        "not_a_symbol",
    ]
    assert accuracy["reviewed_rows"] == 0


def test_review_checkpoint_round_trip_validates_current_alignment():
    details = [_detail(1, "slur", "review", "1.0", "2.0")]
    decision = build_editor_rows(details)[0]
    decision["action"] = "confirm"
    decisions = {"page-01:Y1": decision}

    payload = build_review_checkpoint(
        decisions,
        "Ivy",
        alignment_fingerprint="a" * 64,
        score_id="score-01",
        pipeline_version="pipeline-2",
    )
    restored, errors = load_review_checkpoint(
        payload,
        details,
        expected_fingerprint="a" * 64,
        expected_score_id="score-01",
        expected_pipeline_version="pipeline-2",
    )

    assert errors == []
    assert restored == decisions
    assert payload["reviewer"] == "Ivy"
    assert payload["schema_version"] == "2.0"
    assert payload["alignment_fingerprint"] == "a" * 64

    payload["entries"][0]["yolo_line"] = "99"
    restored, errors = load_review_checkpoint(
        payload,
        details,
        expected_fingerprint="a" * 64,
        expected_score_id="score-01",
        expected_pipeline_version="pipeline-2",
    )
    assert restored == {}
    assert errors == [
        "checkpoint row not found in current alignment: page-01:Y99"
    ]


def test_review_checkpoint_rejects_missing_or_different_alignment_fingerprint():
    details = [_detail(1, "slur", "review", "1.0", "2.0")]
    decision = build_editor_rows(details)[0]
    decision["action"] = "confirm"
    payload = build_review_checkpoint(
        {"page-01:Y1": decision},
        "Ivy",
        alignment_fingerprint="a" * 64,
        score_id="score-01",
        pipeline_version="pipeline-2",
    )

    restored, errors = load_review_checkpoint(
        payload,
        details,
        expected_fingerprint="b" * 64,
        expected_score_id="score-01",
        expected_pipeline_version="pipeline-2",
    )
    assert restored == {}
    assert errors == [
        "review checkpoint belongs to a different alignment input batch; "
        "use the same images, YOLO TXT, XML, BPSD CSV, notes.json, and settings"
    ]

    del payload["alignment_fingerprint"]
    restored, errors = load_review_checkpoint(
        payload,
        details,
        expected_fingerprint="a" * 64,
        expected_score_id="score-01",
        expected_pipeline_version="pipeline-2",
    )
    assert restored == {}
    assert "missing alignment_fingerprint" in errors[0]
