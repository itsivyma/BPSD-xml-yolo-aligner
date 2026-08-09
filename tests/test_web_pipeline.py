import csv
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from bpsd_aligner.web_pipeline import build_output_zip, run_uploaded_alignment


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1" width="100">
      <attributes>
        <divisions>4</divisions>
        <time><beats>1</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type><dynamics default-x="25"><f/></dynamics></direction-type>
        <staff>1</staff>
      </direction>
      <note default-x="25">
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>
"""


def _write_uploads(directory: Path) -> dict[str, Path]:
    image_path = directory / "page.png"
    image = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    for staff_start in (80, 190):
        for offset in range(5):
            y = staff_start + offset * 10
            draw.line((80, y, 920, y), fill="black", width=2)
    image.save(image_path)

    yolo_path = directory / "page.txt"
    yolo_path.write_text("18 0.25 0.30 0.03 0.04\n", encoding="utf-8")
    xml_path = directory / "score.xml"
    xml_path.write_text(MUSICXML, encoding="utf-8")
    bps_path = directory / "ann_score_note.csv"
    bps_path.write_text(
        "start_meas;end_meas;duration_quarterLength;pitch;pitchName;timeSig;articulation;grace\n"
        "0.000;1.000;1.000;60;C4;1/4;;0\n",
        encoding="utf-8",
    )
    categories_path = directory / "notes.json"
    categories_path.write_text(
        json.dumps({"categories": [{"id": 18, "name": "dynamicF"}]}),
        encoding="utf-8",
    )
    return {
        "image_path": image_path,
        "yolo_path": yolo_path,
        "xml_path": xml_path,
        "bps_notes_path": bps_path,
        "notes_json_path": categories_path,
    }


def test_uploaded_alignment_preserves_all_sources_and_renders_overlay(tmp_path):
    inputs = _write_uploads(tmp_path)
    progress = []

    report = run_uploaded_alignment(
        **inputs,
        output_dir=tmp_path / "output",
        page_number=1,
        score_id="synthetic-score",
        progress_callback=lambda step, total, message: progress.append(
            (step, total, message)
        ),
    )

    assert report["passed"] is True
    assert report["alignment_rows"] == 1
    assert report["input_counts"]["yolo_boxes"] == 1
    assert [item[0] for item in progress] == list(range(7))

    all_information_path = Path(report["outputs"]["all_information_csv"])
    with all_information_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    source_types = {row["source_record_type"] for row in rows}
    assert source_types == {"yolo", "xml_event", "xml_node"}
    assert len(rows) == report["all_information_rows"]
    assert sum(row["source_record_type"] == "yolo" for row in rows) == 1
    assert sum(row["source_record_type"] == "xml_event" for row in rows) == report["xml_event_rows"]
    assert sum(row["source_record_type"] == "xml_node" for row in rows) == report["xml_node_rows"]

    overlay_path = Path(report["outputs"]["all_symbols_overlay"])
    with Image.open(overlay_path) as overlay:
        overlay.verify()

    archive_path = build_output_zip(report, tmp_path / "output.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "all_information_csv.csv" in names
    assert "all_symbols_overlay.png" in names
