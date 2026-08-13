import io
import json
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


def test_web_app_loads_without_exceptions():
    app = Path(__file__).parents[1] / "bpsd_aligner" / "web.py"
    result = AppTest.from_file(str(app)).run(timeout=15)
    assert not result.exception
    assert result.title[0].value == "BPSD XML–YOLO Aligner"
    assert [tab.label for tab in result.tabs] == [
        "Run alignment",
        "Inspect CSV",
        "Advanced combine",
        "CLI guide",
    ]
    assert any("score_xml_repetitions" in item.value for item in result.info)
    assert any(
        "Clean repetition PDF" in uploader.label
        for uploader in result.file_uploader
    )
    assert any(
        checkbox.label == "Run alignment in a background worker"
        for checkbox in result.checkbox
    )


def test_web_app_can_require_an_access_token(monkeypatch):
    monkeypatch.setenv("BPSD_ALIGNER_ACCESS_TOKEN", "test-only-token")
    app = Path(__file__).parents[1] / "bpsd_aligner" / "web.py"
    result = AppTest.from_file(str(app)).run(timeout=15)

    assert not result.exception
    assert result.title[0].value == "BPSD XML–YOLO Aligner"
    assert not result.tabs
    assert any(field.label == "Access token" for field in result.text_input)
    assert any(button.label == "Open aligner" for button in result.button)


def test_completed_job_exposes_human_review_editor_and_apply_button():
    app_path = Path(__file__).parents[1] / "bpsd_aligner" / "web.py"
    app = AppTest.from_file(str(app_path))
    page_image = Image.new("RGB", (800, 1000), "white")
    page_buffer = io.BytesIO()
    page_image.save(page_buffer, format="PNG")
    app.session_state["raw_alignment_job"] = {
        "fingerprint": "a" * 64,
        "report": {
            "passed": True,
            "score_id": "score-01",
            "pipeline_version": "pipeline-2",
            "alignment_rows": 1,
            "yolo_rows_needing_review": 1,
            "warnings": [],
            "validation_errors": [],
            "yolo_status_counts": {"review": 1},
        },
        "final_bps_csv": b"class_id\n56\n",
        "yolo_aligned_csv": b"class_id\n56\n",
        "xml_events_csv": b"class\nxmlNote\n",
        "xml_nodes_csv": b"tag\nnote\n",
        "yolo_xml_timeline_csv": b"source_record_type\nyolo\nxml_event\n",
        "all_information_csv": b"source_record_type\nyolo\nxml_event\nxml_node\n",
        "combined_master_csv": b"row_origin\nyolo\n",
        "alignment_links_csv": b"link_id\nL1\n",
        "xml_spans_csv": b"span_id\nS1\n",
        "performance_expanded_timeline_csv": b"record_id\nR1\n",
        "validation_json": b"{}",
        "output_zip": b"zip",
        "overlays": {},
        "page_images": {"page-01": page_buffer.getvalue()},
        "detailed_rows": [
            {
                "page_id": "page-01",
                "txt_line": "1",
                "class_id": "56",
                "class": "slur",
                "x": "0.5",
                "y": "0.5",
                "w": "0.1",
                "h": "0.02",
                "musical_time": "0",
                "start_meas": "1.0",
                "end_meas": "2.0",
                "start_note": "10",
                "end_note": "20",
                "connected_note": "[10, 20]",
                "stem_dir": "NA",
                "xml_staff": "1",
                "status": "review",
                "target_type": "span",
            },
            {
                "page_id": "page-01",
                "txt_line": "2",
                "class_id": "25",
                "class": "fingering1",
                "x": "0.6",
                "y": "0.5",
                "w": "0.02",
                "h": "0.02",
                "musical_time": "0",
                "start_meas": "3.0",
                "end_meas": "3.0",
                "start_note": "30",
                "end_note": "30",
                "connected_note": "[30]",
                "stem_dir": "NA",
                "xml_staff": "1",
                "status": "review",
                "target_type": "note",
                "target_x_px": "480",
                "target_y_px": "500",
                "review_note_candidates_json": json.dumps(
                    [
                        {
                            "note_id": 31,
                            "start_meas": "3.5",
                            "end_meas": "3.5",
                            "connected_note": "[31]",
                            "pitch": "E4",
                            "xml_measure": 4,
                            "staff": 1,
                            "x_px": 500,
                            "y_px": 500,
                        }
                    ]
                ),
            },
        ],
    }

    result = app.run(timeout=15)

    assert not result.exception
    assert any(
        expander.label == "Human review and corrections"
        for expander in result.expander
    )
    assert any(
        button.label == "Apply reviewed decisions" for button in result.button
    )
    assert any(
        checkbox.label == "Include machine-matched rows for spot checking"
        for checkbox in result.checkbox
    )
    assert any(
        button.label == "✓ 機器答案正確" for button in result.button
    )
    assert any(
        button.label == "Apply all saved workspace decisions"
        for button in result.button
    )
    assert any(
        uploader.label == "Resume from review checkpoint"
        for uploader in result.file_uploader
    )
    assert any(
        checkbox.label == "Prepare Final CSV + review images ZIP"
        for checkbox in result.checkbox
    )
    assert not any(
        button.label == "Final CSV + review images ZIP"
        for button in result.button
    )
    next_button = next(button for button in result.button if button.label == "下一筆 →")
    result = next_button.click().run(timeout=15)
    assert not result.exception
    selector = next(
        item for item in result.selectbox if item.label == "目前項目（可直接跳到其他筆）"
    )
    assert selector.value == "page-01:Y2"
    assert any(field.label == "開始音符" for field in result.text_input)
    assert any(field.label == "結束音符" for field in result.text_input)
    assert not any(item.label == "開始音符" for item in result.selectbox)
    assert not any(item.label == "結束音符" for item in result.selectbox)

    note_selector = next(
        item
        for item in result.selectbox
        if item.label == "機器圈錯音時，請選正確的橘色編號"
    )
    result = note_selector.select("1").run(timeout=15)
    assert not result.exception
    correction_button = next(
        button
        for button in result.button
        if button.label == "✓ 儲存所選音頭更正"
    )
    result = correction_button.click().run(timeout=15)
    assert not result.exception
    decision = result.session_state["raw_alignment_job"]["review_decisions"][
        "page-01:Y2"
    ]
    assert decision["action"] == "correct"
    assert decision["start_note"] == "31"
    assert decision["start_meas"] == "3.5"

    previous_button = next(
        button for button in result.button if button.label == "← 上一筆"
    )
    result = previous_button.click().run(timeout=15)
    assert not result.exception
    selector = next(
        item for item in result.selectbox if item.label == "目前項目（可直接跳到其他筆）"
    )
    assert selector.value == "page-01:Y1"

    confirm_button = next(
        button for button in result.button if button.label == "✓ 機器答案正確"
    )
    result = confirm_button.click().run(timeout=15)
    assert not result.exception
    selector = next(
        item for item in result.selectbox if item.label == "目前項目（可直接跳到其他筆）"
    )
    assert selector.value == "page-01:Y2"
