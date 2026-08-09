import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from xml_export import BPS_FIELDS, EVENT_FIELDS, export_dataset, export_score


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>1</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="below">
        <direction-type><dynamics><f/></dynamics></direction-type>
        <voice>2</voice><staff>2</staff>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><stem>up</stem><staff>1</staff>
        <tie type="start"/>
        <notations>
          <tied type="start"/>
          <slur type="start" number="1"/>
          <articulations><staccato placement="above"/></articulations>
        </notations>
      </note>
    </measure>
    <measure number="2">
      <print new-page="yes"/>
      <note><rest/><duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff></note>
      <barline location="right"><repeat direction="backward"/></barline>
    </measure>
  </part>
</score-partwise>
"""


def _write_inputs(tmp_path: Path) -> tuple[dict, Path]:
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(MUSICXML, encoding="utf-8")
    bps_path = tmp_path / "sample.csv"
    bps_path.write_text(
        "start_meas;end_meas;duration_quarterLength;pitch;pitchName;timeSig;articulation;grace\n"
        "001.001;002.000;001.000;60;C4;1/4;staccato;0\n",
        encoding="utf-8",
    )
    repeat_dir = tmp_path / "repeat"
    repeat_dir.mkdir()
    repeat_path = repeat_dir / "sample_repeat_mapping.csv"
    with repeat_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "unfolded_measure_index", "unfolded_measure",
                "written_measure_index", "written_measure",
                "is_repeated_measure", "repeat_occurrence",
                "repeat_occurrence_count", "repeat_group_id", "mapping_status",
            ],
        )
        writer.writeheader()
        for index in (1, 2):
            writer.writerow(
                {
                    "unfolded_measure_index": index,
                    "unfolded_measure": index,
                    "written_measure_index": index,
                    "written_measure": index,
                    "is_repeated_measure": "False",
                    "repeat_occurrence": 1,
                    "repeat_occurrence_count": 1,
                    "repeat_group_id": "",
                    "mapping_status": "matched_fingerprint",
                }
            )
    return {
        "score_id": "sample",
        "xml_path": str(xml_path),
        "bps_notes_path": str(bps_path),
    }, repeat_dir


def test_export_score_is_lossless_and_bps_oriented(tmp_path):
    score, repeat_dir = _write_inputs(tmp_path)
    nodes, events = export_score(score, repeat_dir)

    source_nodes = sum(1 for _ in ET.parse(score["xml_path"]).getroot().iter())
    assert len(nodes) == source_nodes
    assert len({row["xml_node_id"] for row in nodes}) == len(nodes)
    node_ids = {row["xml_node_id"] for row in nodes}
    assert all(row["xml_node_id"] in node_ids for row in events)
    assert EVENT_FIELDS[: len(BPS_FIELDS)] == BPS_FIELDS

    measures = [row for row in nodes if row["tag"] == "measure"]
    assert [(row["measure_index"], row["page"], row["system"]) for row in measures] == [
        (1, 1, 1),
        (2, 2, 1),
    ]

    event_pairs = {(row["event_type"], row["event_subtype"]) for row in events}
    assert ("attribute", "time_signature") in event_pairs
    assert ("attribute", "key") in event_pairs
    assert ("attribute", "clef") in event_pairs
    assert ("direction", "dynamic") in event_pairs
    assert ("notation", "staccato") in event_pairs
    assert ("notation", "slur") in event_pairs
    assert ("notation", "tied") in event_pairs
    assert ("notation", "tie") in event_pairs
    assert ("barline", "barline") in event_pairs

    dynamic = next(row for row in events if row["dynamic"] == "f")
    assert dynamic["staff"] == "2"
    assert dynamic["voice"] == "2"
    note = next(row for row in events if row["event_type"] == "note")
    assert note["start_note"] == 0
    assert note["validation_status"] == "bps_note_matched"
    note_occurrence = json.loads(note["repeat_json"])[0]
    assert note_occurrence["note_id"] == 0
    assert note_occurrence["note_match_status"] == "rounded_tolerance"
    barline = next(row for row in events if row["event_type"] == "barline")
    assert json.loads(barline["repeat_json"])[0]["mapping_status"] == "matched_fingerprint"
    assert json.loads(barline["event_payload_json"])[0]["tag"] == "repeat"


def test_export_dataset_checkpoint_resume(tmp_path):
    score, repeat_dir = _write_inputs(tmp_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["score_id", "xml_path", "bps_notes_path"])
        writer.writeheader()
        writer.writerow(score)

    output_dir = tmp_path / "output"
    first = export_dataset(manifest, repeat_dir, output_dir, resume=True)
    second = export_dataset(manifest, repeat_dir, output_dir, resume=True)

    assert first["passed"] is True
    assert first["score_reports"][0]["resumed"] is False
    assert second["score_reports"][0]["resumed"] is True
    assert (output_dir / "checkpoints" / "sample.json").is_file()
    assert (output_dir / "xml_nodes.csv").is_file()
    assert (output_dir / "xml_events.csv").is_file()
