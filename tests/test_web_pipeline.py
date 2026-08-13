import csv
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from bpsd_aligner.web_pipeline import (
    FINAL_BPS_FIELDS,
    PIPELINE_VERSION,
    build_batch_information_outputs,
    build_final_bps_rows,
    build_output_zip,
    build_performance_expanded_timeline,
    build_xml_spans,
    prepare_score_sources,
    run_uploaded_alignment,
)
from bpsd_aligner.job_store import request_job_cancellation, write_job_manifest
from bpsd_aligner.web_worker import run_background_job
from pipeline_checkpoint import atomic_write_json


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
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
    yolo_path.write_text(
        "18 0.25 0.30 0.03 0.04\n"
        "115 0.50 0.15 0.12 0.03\n",
        encoding="utf-8",
    )
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
        json.dumps(
            {
                "categories": [
                    {"id": 18, "name": "dynamicF"},
                    {"id": 115, "name": "termDolce"},
                ]
            }
        ),
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
        clean_image_path=inputs["image_path"],
        output_dir=tmp_path / "output",
        page_number=1,
        score_id="synthetic-score",
        progress_callback=lambda step, total, message: progress.append(
            (step, total, message)
        ),
    )

    assert report["passed"] is True
    assert report["alignment_rows"] == 2
    assert report["input_counts"]["yolo_boxes"] == 2
    assert report["input_counts"]["clean_reference_pages"] == 1
    assert report["clean_reference"]["requested"] is True
    assert report["clean_reference"]["used"] is True
    assert [item[0] for item in progress] == list(range(7))

    yolo_aligned_path = Path(report["outputs"]["yolo_aligned_csv"])
    with yolo_aligned_path.open(newline="", encoding="utf-8") as file:
        yolo_aligned = list(csv.DictReader(file))
    assert len(yolo_aligned) == report["alignment_rows"]
    assert [float(row["start_meas"]) for row in yolo_aligned] == sorted(
        float(row["start_meas"]) for row in yolo_aligned
    )
    with Path(report["outputs"]["review_queue_csv"]).open(
        newline="", encoding="utf-8"
    ) as file:
        review_queue = list(csv.DictReader(file))
    assert len(review_queue) == report["yolo_rows_needing_review"]
    assert all(row["alignment_status"] != "matched" for row in review_queue)

    with Path(report["outputs"]["final_bps_csv"]).open(
        newline="", encoding="utf-8"
    ) as file:
        final_reader = csv.DictReader(file)
        final_rows = list(final_reader)
        assert final_reader.fieldnames == FINAL_BPS_FIELDS
    assert len(final_rows) == report["alignment_rows"]
    final_dolce = next(row for row in final_rows if row["class"] == "termDolce")
    assert final_dolce["start_meas"] == ""
    assert final_dolce["end_meas"] == ""
    assert final_dolce["start_note"] == ""
    assert final_dolce["human_corrected"] == "0"
    assert all(value != "NA" for row in final_rows for value in row.values())

    xml_events_path = Path(report["outputs"]["xml_events_csv"])
    with xml_events_path.open(newline="", encoding="utf-8") as file:
        standalone_xml_events = list(csv.DictReader(file))
    assert len(standalone_xml_events) == report["xml_event_rows"]
    assert all(row["class"] for row in standalone_xml_events)

    timeline_path = Path(report["outputs"]["yolo_xml_timeline_csv"])
    with timeline_path.open(newline="", encoding="utf-8") as file:
        timeline_rows = list(csv.DictReader(file))
    assert len(timeline_rows) == report["timeline_rows"]
    assert {row["source_record_type"] for row in timeline_rows} == {
        "yolo",
        "xml_event",
    }
    assert [float(row["start_meas"]) for row in timeline_rows] == sorted(
        float(row["start_meas"]) for row in timeline_rows
    )

    all_information_path = Path(report["outputs"]["all_information_csv"])
    with all_information_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    source_types = {row["source_record_type"] for row in rows}
    assert source_types == {"yolo", "xml_event", "xml_node"}
    assert len(rows) == report["all_information_rows"]
    yolo_rows = [row for row in rows if row["source_record_type"] == "yolo"]
    assert len(yolo_rows) == 2
    assert all(row["start_meas"] and row["end_meas"] for row in yolo_rows)
    assert next(row for row in yolo_rows if row["class"] == "termDolce")["review_status"] == "needs_review"
    assert sum(row["source_record_type"] == "xml_event" for row in rows) == report["xml_event_rows"]
    assert sum(row["source_record_type"] == "xml_node" for row in rows) == report["xml_node_rows"]
    xml_event_rows = [row for row in rows if row["source_record_type"] == "xml_event"]
    assert all(row["class"] for row in xml_event_rows)
    assert all(row["class"] for row in rows)
    timed_rows = [row for row in rows if row["start_meas"] not in {"", "NA"}]
    assert [float(row["start_meas"]) for row in timed_rows] == sorted(
        float(row["start_meas"]) for row in timed_rows
    )

    overlay_path = Path(report["outputs"]["all_symbols_overlay"])
    with Image.open(overlay_path) as overlay:
        overlay.verify()
    review_overlay_path = Path(report["outputs"]["review_overlay"])
    with Image.open(review_overlay_path) as overlay:
        overlay.verify()
    assert report["class_overlay_count"] == 2
    class_overlay_keys = {
        name
        for name in report["outputs"]
        if name.startswith("class_") and name.endswith("_overlay")
    }
    assert class_overlay_keys == {
        "class_dynamicF_overlay",
        "class_termDolce_overlay",
    }
    for key in class_overlay_keys:
        with Image.open(report["outputs"][key]) as overlay:
            overlay.verify()

    archive_path = build_output_zip(report, tmp_path / "output.zip")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "all_information.csv" in names
    assert "page_all_symbols.png" in names
    assert "page_needs_review.png" in names
    assert "page_class_dynamicF.png" in names
    assert "page_class_termDolce.png" in names
    assert "yolo_aligned.csv" in names
    assert "bps_omr_final.csv" in names
    assert "review_queue.csv" in names
    assert "xml_events.csv" in names
    assert "yolo_xml_timeline.csv" in names

    yolo_fields, yolo_rows = _read_csv_with_fields(yolo_aligned_path)
    _event_fields, event_rows = _read_csv_with_fields(xml_events_path)
    xml_nodes_path = Path(report["outputs"]["xml_nodes_csv"])
    _node_fields, node_rows = _read_csv_with_fields(xml_nodes_path)
    second_page_rows = [
        {
            **row,
            "page_id": "page-02",
            "scan_page": "2",
            "bbox_id": f"page-02:{row['bbox_id']}",
        }
        for row in yolo_rows
    ]
    batch = build_batch_information_outputs(
        yolo_rows=[*yolo_rows, *second_page_rows],
        xml_events=event_rows,
        xml_nodes=node_rows,
        output_dir=tmp_path / "batch",
    )
    assert yolo_fields
    assert batch["passed"] is True
    assert batch["yolo_rows"] == 4
    assert batch["xml_event_rows"] == len(event_rows)
    assert batch["xml_node_rows"] == len(node_rows)
    assert Path(batch["outputs"]["xml_spans_csv"]).is_file()
    assert Path(
        batch["outputs"]["performance_expanded_timeline_csv"]
    ).is_file()
    with Path(batch["outputs"]["yolo_xml_timeline_csv"]).open(
        newline="", encoding="utf-8-sig"
    ) as file:
        batch_timeline = list(csv.DictReader(file))
    assert sum(
        row["source_record_type"] == "xml_event" for row in batch_timeline
    ) == len(event_rows)


def test_shared_score_checkpoint_is_bound_to_source_hashes(tmp_path):
    inputs = _write_uploads(tmp_path)
    output = tmp_path / "shared"
    first = prepare_score_sources(
        xml_path=inputs["xml_path"],
        bps_notes_path=inputs["bps_notes_path"],
        output_dir=output,
        score_id="hash-test",
    )
    inputs["bps_notes_path"].write_text(
        inputs["bps_notes_path"].read_text(encoding="utf-8")
        + "1.000;2.000;1.000;62;D4;1/4;;0\n",
        encoding="utf-8",
    )
    second = prepare_score_sources(
        xml_path=inputs["xml_path"],
        bps_notes_path=inputs["bps_notes_path"],
        output_dir=output,
        score_id="hash-test",
    )

    assert first["bps_notes_sha256"] != second["bps_notes_sha256"]
    assert second["bps_note_count"] == 2


def test_background_worker_completes_and_resumes_page_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_uploads(source_dir)
    fingerprint = "9" * 64
    job_dir = tmp_path / fingerprint
    input_dir = job_dir / "inputs"
    page_dir = input_dir / "pages" / "page-01"
    page_dir.mkdir(parents=True)
    second_page_dir = input_dir / "pages" / "page-02"
    second_page_dir.mkdir(parents=True)
    copied = {}
    for name, source_key in (
        ("score.xml", "xml_path"),
        ("ann_score_note.csv", "bps_notes_path"),
        ("notes.json", "notes_json_path"),
    ):
        copied[source_key] = shutil.copy(source[source_key], input_dir / name)
    copied["image_path"] = shutil.copy(source["image_path"], page_dir / "page.png")
    copied["yolo_path"] = shutil.copy(source["yolo_path"], page_dir / "page.txt")
    second_image = shutil.copy(source["image_path"], second_page_dir / "page.png")
    second_yolo = shutil.copy(source["yolo_path"], second_page_dir / "page.txt")
    write_job_manifest(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version=PIPELINE_VERSION,
        inputs=[],
    )
    request_path = job_dir / "job_request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": "1.0",
            "pipeline_version": PIPELINE_VERSION,
            "fingerprint": fingerprint,
            "job_dir": str(job_dir),
            "score_id": "background-test",
            "infer_fingerings": True,
            "xml_path": str(copied["xml_path"]),
            "bps_notes_path": str(copied["bps_notes_path"]),
            "notes_json_path": str(copied["notes_json_path"]),
            "unfolded_xml_path": "",
            "clean_pdf_path": "",
            "resume_archive": "",
            "pages": [
                {
                    "page_id": "page-01",
                    "page_number": 1,
                    "image_path": str(copied["image_path"]),
                    "yolo_path": str(copied["yolo_path"]),
                },
                {
                    "page_id": "page-02",
                    "page_number": 1,
                    "image_path": str(second_image),
                    "yolo_path": str(second_yolo),
                },
            ],
        },
    )

    result_path = run_background_job(request_path)
    result = json.loads(result_path.read_text())
    checkpoint = job_dir / "checkpoints" / "page-01.json"
    second_checkpoint = job_dir / "checkpoints" / "page-02.json"
    first_checkpoint_mtime = checkpoint.stat().st_mtime_ns
    second_checkpoint_mtime = second_checkpoint.stat().st_mtime_ns
    assert result["report"]["page_count"] == 2
    assert Path(result["final_bps_csv"]).is_file()
    assert json.loads((job_dir / "job_status.json").read_text())["state"] == "completed"

    run_background_job(request_path)
    assert checkpoint.stat().st_mtime_ns == first_checkpoint_mtime
    assert second_checkpoint.stat().st_mtime_ns == second_checkpoint_mtime

    request_job_cancellation(job_dir)
    run_background_job(request_path)
    assert json.loads((job_dir / "job_status.json").read_text())["state"] == "cancelled"


def _read_csv_with_fields(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def test_cross_page_xml_spans_and_repeat_expansion_are_explicit(tmp_path: Path):
    events = [
        {
            "score_id": "S",
            "xml_event_id": "E1",
            "event_type": "notation",
            "event_subtype": "slur",
            "class": "slur",
            "voice": "1",
            "staff": "1",
            "page": "1",
            "xml_measure": "4",
            "start_meas": "4.0",
            "end_meas": "4.0",
            "xml_attributes_json": '{"number":"1","type":"start"}',
            "anchor_xml_event_ids_json": "[]",
            "repeat_json": '[{"start_meas":4.0,"end_meas":4.0,"repeat_occurrence":1}]',
        },
        {
            "score_id": "S",
            "xml_event_id": "E2",
            "event_type": "notation",
            "event_subtype": "slur",
            "class": "slur",
            "voice": "1",
            "staff": "2",
            "page": "2",
            "xml_measure": "5",
            "start_meas": "5.0",
            "end_meas": "5.0",
            "xml_attributes_json": '{"number":"1","type":"stop"}',
            "anchor_xml_event_ids_json": "[]",
            "repeat_json": '[{"start_meas":8.0,"end_meas":8.0,"repeat_occurrence":2}]',
        },
    ]
    spans = build_xml_spans(events, tmp_path / "spans.csv")
    assert len(spans) == 1
    assert spans[0]["cross_page"] is True
    assert spans[0]["start_page"] == "1"
    assert spans[0]["end_page"] == "2"

    expanded = build_performance_expanded_timeline(
        [
            {
                "bbox_id": "Y1",
                "start_meas": "1",
                "end_meas": "1",
                "repeat_occurrences_json": (
                    '[{"start_meas":1,"end_meas":1},'
                    '{"start_meas":9,"end_meas":9}]'
                ),
            }
        ],
        events,
        tmp_path / "performance.csv",
    )
    assert [row["start_meas"] for row in expanded] == [1, 4.0, 8.0, 9]


def test_tie_spans_pair_by_pitch_and_deduplicate_tie_and_tied_markers(tmp_path: Path):
    events = []
    for event_id, pitch, note_id, time in (
        ("N1", "60", "10", "1"),
        ("N2", "64", "11", "1"),
        ("N3", "60", "12", "2"),
        ("N4", "64", "13", "2"),
    ):
        events.append(
            {
                "score_id": "S",
                "xml_event_id": event_id,
                "event_type": "note",
                "midi_pitch": pitch,
                "start_note": note_id,
                "end_note": note_id,
                "start_meas": time,
                "end_meas": time,
            }
        )
    for subtype in ("tie", "tied"):
        for event_id, anchor, marker in (
            (f"{subtype}-1", "N1", "start"),
            (f"{subtype}-2", "N2", "start"),
            (f"{subtype}-3", "N3", "stop"),
            (f"{subtype}-4", "N4", "stop"),
        ):
            events.append(
                {
                    "score_id": "S",
                    "xml_event_id": event_id,
                    "event_type": "notation",
                    "event_subtype": subtype,
                    "class": "tie",
                    "voice": "1",
                    "staff": "1",
                    "page": "1",
                    "xml_attributes_json": json.dumps({"type": marker}),
                    "anchor_xml_event_ids_json": json.dumps([anchor]),
                }
            )

    spans = build_xml_spans(events, tmp_path / "tie-spans.csv")
    assert len(spans) == 2
    assert {(row["start_note"], row["end_note"]) for row in spans} == {
        ("10", "12"),
        ("11", "13"),
    }


def test_final_bps_rows_keep_human_corrections_and_clear_machine_candidates():
    base = {
        "class_id": "25",
        "x": "0.5",
        "y": "0.5",
        "w": "0.2",
        "h": "0.01",
        "class": "dynamicCrescendoLong",
        "musical_time": "0",
        "start_meas": "43.0",
        "end_meas": "44.0",
        "start_note": "NA",
        "end_note": "NA",
        "connected_note": "NA",
        "stem_dir": "NA",
        "alignment_status": "ambiguous",
        "match_source": "geometric_span_time_estimate",
        "human_approved": "false",
    }

    machine, corrected = build_final_bps_rows(
        [base, {**base, "human_approved": "true", "match_source": "human_review"}]
    )

    assert machine["start_meas"] == ""
    assert machine["end_meas"] == ""
    assert machine["human_corrected"] == "0"
    assert corrected["start_meas"] == "43.0"
    assert corrected["end_meas"] == "44.0"
    assert corrected["start_note"] == ""
    assert corrected["human_corrected"] == "1"
