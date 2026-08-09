import json
from pathlib import Path

from PIL import Image

import render_yolo_overlays
from render_yolo_overlays import render_dataset


def test_render_dataset_writes_two_overlays_and_indexes(tmp_path: Path) -> None:
    xia = tmp_path / "xia"
    (xia / "images").mkdir(parents=True)
    (xia / "labels").mkdir()
    Image.new("RGB", (200, 100), "white").save(xia / "images" / "score-01.jpg")
    (xia / "labels" / "score-01.txt").write_text(
        "3 0.5 0.5 0.2 0.4\n", encoding="utf-8"
    )
    (xia / "notes.json").write_text(
        json.dumps({"categories": [{"id": 3, "name": "dynamicF"}]}),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    rows = render_dataset(xia, output)

    assert rows[0]["yolo_boxes"] == "1"
    assert (output / "id_only" / "score-01.png").exists()
    assert (output / "id_class" / "score-01.png").exists()
    assert (output / "overlay_index.csv").exists()
    assert "score-01" in (output / "index.html").read_text(encoding="utf-8")


def test_render_dataset_resume_reuses_complete_overlay_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    xia = tmp_path / "xia"
    (xia / "images").mkdir(parents=True)
    (xia / "labels").mkdir()
    Image.new("RGB", (200, 100), "white").save(xia / "images" / "score-01.jpg")
    (xia / "labels" / "score-01.txt").write_text(
        "3 0.5 0.5 0.2 0.4\n",
        encoding="utf-8",
    )
    (xia / "notes.json").write_text(
        json.dumps({"categories": [{"id": 3, "name": "dynamicF"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    render_dataset(xia, output)

    def unexpected_render(*args, **kwargs):
        raise AssertionError("resume should not rerender valid overlays")

    monkeypatch.setattr(render_yolo_overlays, "render_overlay", unexpected_render)
    rows = render_dataset(xia, output, resume=True, progress_every=1)

    assert rows[0]["yolo_boxes"] == "1"
