import csv
from pathlib import Path

from PIL import Image

import alignment_review_sheets
from alignment_review_sheets import generate_review_sheets


def _write_master(path: Path, image_path: Path) -> None:
    row = {
        "bbox_id": "score-01:Y1",
        "page_id": "score-01",
        "yolo_line": "1",
        "class_id": "3",
        "class": "fingering1",
        "alignment_status": "candidate",
        "movement_scope_status": "in_bpsd_scope",
        "human_approved": "false",
        "image_path": str(image_path),
        "x": "0.5",
        "y": "0.5",
        "w": "0.1",
        "h": "0.1",
        "target_x_px": "110",
        "target_y_px": "55",
        "xml_measure": "1",
        "pitches": '["C4"]',
        "note_ids": "[7]",
        "repeat_occurrence_count": "1",
        "start_meas": "0.0",
        "staff": "1",
        "confidence": "0.9",
        "end_meas": "0.0",
        "review_status": "needs_review",
        "corrected_value_json": "",
        "comment": "",
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_generate_review_sheets_resume_reuses_complete_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "score-01.png"
    Image.new("RGB", (220, 110), "white").save(image_path)
    master = tmp_path / "master.csv"
    _write_master(master, image_path)
    output = tmp_path / "review"

    first = generate_review_sheets(master, output, progress_every=1)
    assert first["sheets"] == 1
    assert first["reused_sheets"] == 0

    def unexpected_render(*args, **kwargs):
        raise AssertionError("resume should not rerender a valid PNG/CSV pair")

    monkeypatch.setattr(alignment_review_sheets, "_render_panel", unexpected_render)
    second = generate_review_sheets(
        master,
        output,
        resume=True,
        progress_every=1,
    )

    assert second["sheets"] == 1
    assert second["reused_sheets"] == 1
