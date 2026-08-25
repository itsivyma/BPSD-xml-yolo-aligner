import csv
import json

import pytest
from PIL import Image, ImageDraw

from bps_xml_alignment import (
    OUTPUT_FIELDS,
    StaffGeometry,
    SystemGeometry,
    align_barlines_from_reference,
    attach_review_note_candidates,
    attach_bps_note_ids,
    attach_repeat_occurrences,
    build_slur_candidates,
    build_tie_candidates,
    conservative_all_symbol_rows,
    detect_barlines,
    detect_systems,
    load_categories,
    load_yolo,
    match_dynamics,
    match_cross_page_spans,
    match_direction_symbols,
    match_fingerings,
    match_point_notations,
    match_small_beams,
    match_small_flags,
    match_small_note_symbols,
    match_small_stems,
    match_text_directions,
    match_tuplets,
    match_wavy_line_spans,
    match_xml_spans,
    estimate_all_symbol_times,
    note_pixel_position,
    parse_musicxml_page,
    snap_notehead_x,
    unresolved_fingering_rows,
    write_csv,
)
from defusedxml.common import EntitiesForbidden


def _one_system() -> list[SystemGeometry]:
    return [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
            ),
            lower=StaffGeometry(
                center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
            ),
            x_left=100,
            x_right=900,
        )
    ]


def test_dynamic_count_agreement_alone_cannot_produce_full_confidence():
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "dynamicF",
            "x": 0.10,
            "y": 0.20,
            "w": 0.02,
            "h": 0.02,
        }
    ]
    events = [
        {
            "class": "dynamicF",
            "system": 1,
            "x_norm": 0.90,
            "bps_time": 2.0,
            "xml_measure": 3,
            "xml_symbol": "f",
            "staff": 1,
            "repeat_occurrences": [],
        }
    ]

    rows, _unused = match_dynamics(boxes, events, _one_system(), 400)

    assert rows[0]["count_agreement"] == "1.000"
    assert rows[0]["geometry_score"] == "0.000"
    assert rows[0]["confidence"] != "1.000"
    assert rows[0]["status"] == "review"
    assert rows[0]["confidence_calibrated"] == "false"


def test_repeat_disagreement_prevents_dynamic_auto_acceptance():
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "dynamicF",
            "x": 0.50,
            "y": 0.20,
            "w": 0.02,
            "h": 0.02,
        }
    ]
    events = [
        {
            "class": "dynamicF",
            "system": 1,
            "x_norm": 0.50,
            "bps_time": 2.0,
            "xml_measure": 3,
            "xml_symbol": "f",
            "staff": 1,
            "repeat_occurrences": [
                {"mapping_status": "structural_unfolded_disagreement"}
            ],
        }
    ]

    rows, _unused = match_dynamics(boxes, events, _one_system(), 400)

    assert rows[0]["match_score"] == "1.000"
    assert rows[0]["repeat_mapping_status"] == (
        "structural_unfolded_disagreement"
    )
    assert rows[0]["status"] == "review"
    assert rows[0]["xml_time_confirmed"] == "false"


def test_unsupported_navigation_prevents_dynamic_auto_acceptance():
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "dynamicF",
            "x": 0.50,
            "y": 0.20,
            "w": 0.02,
            "h": 0.02,
        }
    ]
    events = [
        {
            "class": "dynamicF",
            "system": 1,
            "x_norm": 0.50,
            "bps_time": 2.0,
            "xml_measure": 3,
            "xml_symbol": "f",
            "staff": 1,
            "repeat_occurrences": [
                {"mapping_status": "unsupported_navigation"}
            ],
        }
    ]

    rows, _unused = match_dynamics(boxes, events, _one_system(), 400)

    assert rows[0]["repeat_mapping_status"] == "unsupported_navigation"
    assert rows[0]["status"] == "review"
    assert rows[0]["xml_time_confirmed"] == "false"


def test_attach_review_note_candidates_orders_nearby_notes_and_keeps_metadata():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
            ),
            lower=StaffGeometry(
                center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
            ),
            x_left=100,
            x_right=900,
        )
    ]
    rows = [{"system": "1", "x": "0.30", "y": "0.20"}]
    notes = [
        {
            "note_id": 20,
            "system": 1,
            "staff": 1,
            "x_norm": 0.75,
            "bps_time": 2.0,
            "xml_measure": 3,
            "pitch_name": "G4",
            "diatonic": 32,
            "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 10,
            "system": 1,
            "staff": 1,
            "x_norm": 0.25,
            "bps_time": 1.5,
            "xml_measure": 2,
            "printed_measure": 1,
            "pitch_name": "E4",
            "diatonic": 30,
            "clef": {"sign": "G", "line": 2},
        },
    ]

    attach_review_note_candidates(rows, notes, systems, 1000, 400)
    candidates = json.loads(rows[0]["review_note_candidates_json"])

    assert [candidate["note_id"] for candidate in candidates] == [10, 20]
    assert candidates[0]["start_meas"] == "1.500"
    assert candidates[0]["pitch"] == "E4"
    assert candidates[0]["staff"] == 1
    assert candidates[0]["xml_measure"] == 2
    assert candidates[0]["printed_measure"] == 1
    assert candidates[0]["measure_note_order"] == 1


def test_span_review_candidates_include_all_page_systems_and_xml_only_notes():
    systems = [
        SystemGeometry(
            number=number,
            upper=StaffGeometry(
                center=center, line_spacing=10, lines=[center - 20, center - 10, center, center + 10, center + 20]
            ),
            lower=StaffGeometry(
                center=center + 100,
                line_spacing=10,
                lines=[center + 80, center + 90, center + 100, center + 110, center + 120],
            ),
            x_left=100,
            x_right=900,
        )
        for number, center in ((1, 100), (2, 300))
    ]
    rows = [
        {
            "system": "1",
            "target_type": "span",
            "x": "0.5",
            "y": "0.25",
        }
    ]
    notes = [
        {
            "note_id": 10,
            "xml_note_sequence": 1,
            "system": 1,
            "staff": 1,
            "x_norm": 0.5,
            "bps_time": 1.0,
            "xml_measure": 2,
            "pitch_name": "E4",
            "diatonic": 30,
            "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": None,
            "xml_note_sequence": 2,
            "system": 2,
            "staff": 1,
            "x_norm": 0.25,
            "bps_time": None,
            "xml_measure": 8,
            "pitch_name": "G4",
            "diatonic": 32,
            "clef": {"sign": "G", "line": 2},
        },
    ]

    attach_review_note_candidates(rows, notes, systems, 1000, 600, limit=1)
    candidates = json.loads(rows[0]["review_note_candidates_json"])

    assert len(candidates) == 2
    assert {candidate["xml_measure"] for candidate in candidates} == {2, 8}
    xml_only = next(candidate for candidate in candidates if candidate["note_id"] is None)
    assert xml_only["connected_note"] == "[]"


def test_note_pixel_position_uses_measure_local_piecewise_mapping():
    system = SystemGeometry(
        number=1,
        upper=StaffGeometry(
            center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
        ),
        lower=StaffGeometry(
            center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
        ),
        x_left=100,
        x_right=900,
    )
    note = {
        "system": 1,
        "system_measure_index": 2,
        "measure_x_norm": 0.25,
        "x_norm": 0.5,
        "staff": 1,
        "diatonic": 34,
        "clef": {"sign": "G", "line": 2},
    }

    x, _y = note_pixel_position(
        note,
        system,
        measure_x_maps={(1, 2): {"left_x": 600, "right_x": 800}},
    )

    assert x == 650


def test_parse_musicxml_keeps_tied_start_and_stop(tmp_path):
    xml_path = tmp_path / "score.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1" width="100">
    <attributes><divisions>1</divisions></attributes>
    <note default-x="10"><pitch><step>C</step><octave>4</octave></pitch>
      <duration>1</duration><voice>1</voice><staff>1</staff>
      <notations><tied type="start"/></notations></note>
    <note default-x="50"><pitch><step>C</step><octave>4</octave></pitch>
      <duration>1</duration><voice>1</voice><staff>1</staff>
      <notations><tied type="stop"/></notations></note>
  </measure></part>
</score-partwise>
""",
        encoding="utf-8",
    )

    page = parse_musicxml_page(xml_path, 1)

    assert page["notes"][0]["tie_marks"] == [{"type": "start"}]
    assert page["notes"][1]["tie_marks"] == [{"type": "stop"}]


def test_parse_musicxml_keeps_grace_beams_and_direction_endpoints(tmp_path):
    xml_path = tmp_path / "directions.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"><part-name>Piano</part-name>
</score-part></part-list><part id="P1"><measure number="1" width="100">
  <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
  <direction><direction-type><words default-x="12">a tempo</words><metronome default-x="14"><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome><octave-shift type="down" number="1" size="8" default-x="10"/><wedge type="crescendo" number="1" default-x="10"/></direction-type><staff>1</staff></direction>
  <direction><direction-type><pedal type="start" number="1" default-x="10"/></direction-type><staff>2</staff></direction>
  <note default-x="20"><grace/><pitch><step>E</step><octave>4</octave></pitch><voice>1</voice><type>eighth</type><staff>1</staff><stem>up</stem><beam number="1">begin</beam></note>
  <note default-x="30"><grace/><pitch><step>F</step><octave>4</octave></pitch><voice>1</voice><type>eighth</type><staff>1</staff><stem>up</stem><beam number="1">end</beam></note>
  <note default-x="40"><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
  <direction><direction-type><octave-shift type="stop" number="1" default-x="90"/><wedge type="stop" number="1" default-x="90"/></direction-type><staff>1</staff></direction>
  <direction><direction-type><pedal type="stop" number="1" default-x="90"/></direction-type><staff>2</staff></direction>
</measure></part></score-partwise>""",
        encoding="utf-8",
    )

    page = parse_musicxml_page(xml_path, 1)

    assert [note["is_grace"] for note in page["notes"]] == [True, True, False]
    assert page["notes"][0]["beam_values"] == ["begin"]
    assert page["notes"][1]["beam_values"] == ["end"]
    assert [(item["kind"], item["type"]) for item in page["direction_markers"]] == [
        ("octave", "down"),
        ("wedge", "crescendo"),
        ("pedal", "start"),
        ("octave", "stop"),
        ("wedge", "stop"),
        ("pedal", "stop"),
    ]
    assert [(item["kind"], item["text"]) for item in page["text_directions"]] == [
        ("words", "a tempo"),
        ("metronome", "quarter=120"),
    ]


def test_small_note_stem_and_beam_matchers_keep_xml_note_semantics():
    systems = _one_system()
    base = {
        "system": 1,
        "staff": 1,
        "voice": "1",
        "xml_measure": 2,
        "printed_measure": 2,
        "xml_measure_index": 2,
        "system_measure_index": 0,
        "clef": {"sign": "G", "line": 2},
        "is_grace": True,
        "note_type": "eighth",
        "stem": "up",
        "accidental": "",
        "occurrences": [],
    }
    notes = [
        {
            **base,
            "xml_note_sequence": 1,
            "xml_chord_sequence": 1,
            "x_norm": 0.30,
            "bps_time": 1.0,
            "end_bps_time": 1.125,
            "note_id": 10,
            "midi": 64,
            "pitch_name": "E4",
            "diatonic": 30,
            "beam_values": ["begin"],
        },
        {
            **base,
            "xml_note_sequence": 2,
            "xml_chord_sequence": 2,
            "x_norm": 0.50,
            "bps_time": 1.125,
            "end_bps_time": 1.250,
            "note_id": 11,
            "midi": 65,
            "pitch_name": "F4",
            "diatonic": 31,
            "beam_values": ["end"],
        },
    ]
    first_x, first_y = note_pixel_position(notes[0], systems[0])
    second_x, _second_y = note_pixel_position(notes[1], systems[0])
    boxes = [
        {
            "txt_line": 1,
            "class_id": 62,
            "class": "noteheadBlackOnLineSmall",
            "x": first_x / 1000,
            "y": first_y / 400,
            "w": 0.01,
            "h": 0.02,
        },
        {
            "txt_line": 2,
            "class_id": 88,
            "class": "stemSmall",
            "x": first_x / 1000,
            "y": first_y / 400,
            "w": 0.005,
            "h": 0.08,
        },
        {
            "txt_line": 3,
            "class_id": 20,
            "class": "beamSmall",
            "x": ((first_x + second_x) / 2) / 1000,
            "y": first_y / 400,
            "w": abs(second_x - first_x) / 1000,
            "h": 0.01,
        },
    ]

    notehead = match_small_note_symbols(boxes, notes, systems, 1000, 400)[0]
    stem = match_small_stems(boxes, notes, systems, 1000, 400)[0]
    beam = match_small_beams(boxes, notes, systems, 1000, 400)[0]

    assert notehead["status"] == "matched"
    assert notehead["connected_note"] == "[10]"
    assert notehead["end_meas"] == "1.125"
    assert stem["stem_dir"] == 1
    assert stem["end_meas"] == "1.125"
    assert beam["connected_note"] == "[10, 11]"
    assert beam["end_meas"] == "1.250"


def test_small_flag_matcher_uses_unbeamed_grace_duration_and_stem():
    systems = _one_system()
    note = {
        "xml_note_sequence": 1,
        "xml_chord_sequence": 1,
        "system": 1,
        "staff": 1,
        "voice": "1",
        "xml_measure": 2,
        "xml_measure_index": 2,
        "system_measure_index": 0,
        "x_norm": 0.4,
        "bps_time": 1.0,
        "end_bps_time": 1.125,
        "note_id": 10,
        "midi": 64,
        "pitch_name": "E4",
        "diatonic": 30,
        "clef": {"sign": "G", "line": 2},
        "is_grace": True,
        "note_type": "eighth",
        "stem": "up",
        "beam_values": [],
        "occurrences": [],
    }
    note_x, note_y = note_pixel_position(note, systems[0])
    box = {
        "txt_line": 1,
        "class_id": 55,
        "class": "flag8thUpSmall",
        "x": (note_x + 9) / 1000,
        "y": note_y / 400,
        "w": 0.01,
        "h": 0.04,
    }

    row = match_small_flags([box], [note], systems, 1000, 400)[0]

    assert row["status"] == "matched"
    assert row["start_note"] == 10
    assert row["connected_note"] == "[10]"
    assert row["end_meas"] == "1.125"


def test_direction_matcher_uses_octave_span_and_pedal_endpoints():
    systems = _one_system()
    markers = [
        {
            "kind": "octave", "type": "down", "number": "1",
            "system": 1, "staff": 1, "xml_measure": 2,
            "xml_measure_index": 2, "system_measure_index": 0,
            "x_norm": 0.2, "bps_time": 1.0, "occurrences": [],
        },
        {
            "kind": "octave", "type": "stop", "number": "1",
            "system": 1, "staff": 1, "xml_measure": 3,
            "xml_measure_index": 3, "system_measure_index": 1,
            "x_norm": 0.8, "bps_time": 2.0, "occurrences": [],
        },
        {
            "kind": "pedal", "type": "stop", "number": "1",
            "system": 1, "staff": 2, "xml_measure": 3,
            "xml_measure_index": 3, "system_measure_index": 1,
            "x_norm": 0.8, "bps_time": 2.0, "occurrences": [],
        },
        {
            "kind": "wedge", "type": "crescendo", "number": "1",
            "system": 1, "staff": 1, "xml_measure": 2,
            "xml_measure_index": 2, "system_measure_index": 0,
            "x_norm": 0.2, "bps_time": 1.0, "occurrences": [],
        },
        {
            "kind": "wedge", "type": "stop", "number": "1",
            "system": 1, "staff": 1, "xml_measure": 3,
            "xml_measure_index": 3, "system_measure_index": 1,
            "x_norm": 0.8, "bps_time": 2.0, "occurrences": [],
        },
    ]
    notes = [
        {
            "xml_note_sequence": 1, "xml_chord_sequence": 1,
            "system": 1, "staff": 1, "voice": "1", "xml_measure": 2,
            "x_norm": 0.3, "bps_time": 1.2, "note_id": 10,
            "midi": 64, "pitch_name": "E4", "diatonic": 30,
            "clef": {"sign": "G", "line": 2}, "occurrences": [],
        },
        {
            "xml_note_sequence": 2, "xml_chord_sequence": 2,
            "system": 1, "staff": 1, "voice": "1", "xml_measure": 3,
            "x_norm": 0.7, "bps_time": 1.8, "note_id": 11,
            "midi": 67, "pitch_name": "G4", "diatonic": 32,
            "clef": {"sign": "G", "line": 2}, "occurrences": [],
        },
    ]
    boxes = [
        {
            "txt_line": 1, "class_id": 85, "class": "ottavaBracket",
            "x": 0.5, "y": 0.15, "w": 0.48, "h": 0.03,
        },
        {
            "txt_line": 2, "class_id": 59, "class": "keyboardPedalUp",
            "x": 0.74, "y": 0.75, "w": 0.02, "h": 0.03,
        },
        {
            "txt_line": 3, "class_id": 24,
            "class": "dynamicCrescendoHairpin",
            "x": 0.5, "y": 0.4, "w": 0.48, "h": 0.03,
        },
    ]

    rows = match_direction_symbols(boxes, notes, markers, systems, 1000, 400)
    by_class = {row["class"]: row for row in rows}

    assert by_class["ottavaBracket"]["connected_note"] == "[10, 11]"
    assert by_class["ottavaBracket"]["start_meas"] == "1.000"
    assert by_class["ottavaBracket"]["end_meas"] == "2.000"
    assert by_class["keyboardPedalUp"]["start_meas"] == "2.000"
    assert by_class["keyboardPedalUp"]["start_note"] == ""
    assert by_class["dynamicCrescendoHairpin"]["start_meas"] == "1.000"
    assert by_class["dynamicCrescendoHairpin"]["end_meas"] == "2.000"
    assert by_class["dynamicCrescendoHairpin"]["connected_note"] == "NA"


def test_text_and_wavy_line_matchers_use_explicit_xml_evidence():
    systems = _one_system()
    text_events = [
        {
            "kind": "words",
            "text": "a tempo",
            "system": 1,
            "staff": 1,
            "xml_measure": 2,
            "xml_measure_index": 2,
            "system_measure_index": 0,
            "x_norm": 0.5,
            "bps_time": 1.0,
            "repeat_occurrences": [],
        }
    ]
    tempo_box = {
        "txt_line": 1,
        "class_id": 90,
        "class": "tempoATempo",
        "x": 0.5,
        "y": 0.1,
        "w": 0.1,
        "h": 0.03,
    }
    text_row = match_text_directions(
        [tempo_box], text_events, systems, 1000, 400
    )[0]

    note_base = {
        "system": 1,
        "staff": 1,
        "voice": "1",
        "xml_measure": 2,
        "xml_measure_index": 2,
        "system_measure_index": 0,
        "clef": {"sign": "G", "line": 2},
        "occurrences": [],
    }
    notes = [
        {
            **note_base,
            "xml_note_sequence": 1,
            "xml_chord_sequence": 1,
            "x_norm": 0.3,
            "bps_time": 1.0,
            "note_id": 10,
            "midi": 64,
            "pitch_name": "E4",
            "diatonic": 30,
            "wavy_line_marks": [{"type": "start", "number": "1"}],
        },
        {
            **note_base,
            "xml_note_sequence": 2,
            "xml_chord_sequence": 2,
            "x_norm": 0.7,
            "bps_time": 1.5,
            "note_id": 11,
            "midi": 67,
            "pitch_name": "G4",
            "diatonic": 32,
            "wavy_line_marks": [{"type": "stop", "number": "1"}],
        },
    ]
    first_x, _first_y = note_pixel_position(notes[0], systems[0])
    last_x, _last_y = note_pixel_position(notes[1], systems[0])
    wiggle_box = {
        "txt_line": 2,
        "class_id": 84,
        "class": "ornamentWiggleTrill",
        "x": ((first_x + last_x) / 2) / 1000,
        "y": 0.15,
        "w": abs(last_x - first_x) / 1000,
        "h": 0.02,
    }
    wiggle_row = match_wavy_line_spans(
        [wiggle_box], notes, systems, 1000, 400
    )[0]

    assert text_row["status"] == "matched"
    assert text_row["start_meas"] == "1.000"
    assert text_row["start_note"] == ""
    assert wiggle_row["status"] == "matched"
    assert wiggle_row["start_note"] == 10
    assert wiggle_row["end_note"] == 11
    assert wiggle_row["start_meas"] == "1.000"
    assert wiggle_row["end_meas"] == "1.500"


def test_parse_musicxml_accepts_official_doctype_without_resolving_it(tmp_path):
    xml_path = tmp_path / "doctype.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1" width="100">
    <attributes><divisions>1</divisions></attributes>
    <note default-x="10"><pitch><step>C</step><octave>4</octave></pitch>
      <duration>1</duration><voice>1</voice><staff>1</staff></note>
  </measure></part>
</score-partwise>
""",
        encoding="utf-8",
    )

    assert len(parse_musicxml_page(xml_path, 1)["notes"]) == 1


def test_parse_musicxml_uses_scanned_system_measure_anchors(tmp_path):
    xml_path = tmp_path / "layout-mismatch.xml"
    measures = []
    for number in range(1, 7):
        print_tag = (
            '<print new-page="yes"/>'
            if number in {1, 3}
            else '<print new-system="yes"/>'
            if number in {2, 5}
            else ""
        )
        measures.append(
            f'''<measure number="{number}" width="100">{print_tag}
            <attributes><divisions>1</divisions></attributes>
            <note default-x="50"><pitch><step>C</step><octave>4</octave></pitch>
              <duration>1</duration><voice>1</voice><staff>1</staff></note>
            </measure>'''
        )
    xml_path.write_text(
        '''<?xml version="1.0"?><score-partwise><part-list>
        <score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
        <part id="P1">'''
        + "".join(measures)
        + "</part></score-partwise>",
        encoding="utf-8",
    )

    xml_layout = parse_musicxml_page(xml_path, page_number=2)
    scan_layout = parse_musicxml_page(
        xml_path,
        page_number=2,
        system_start_measures=[2, 4],
        page_end_measure=5,
    )

    assert [measure["measure"] for measure in xml_layout["measures"]] == [3, 4, 5, 6]
    assert [measure["measure"] for measure in scan_layout["measures"]] == [3, 4, 5, 6]
    assert [measure["printed_measure"] for measure in scan_layout["measures"]] == [
        2,
        3,
        4,
        5,
    ]
    assert [measure["system"] for measure in scan_layout["measures"]] == [1, 1, 2, 2]
    assert scan_layout["layout_source"] == "scan_printed_measure_anchors"
    assert scan_layout["notes"][0]["xml_measure"] == 3
    assert scan_layout["notes"][0]["printed_measure"] == 2
    assert scan_layout["notes"][0]["page"] == 2


def test_scanned_measure_anchor_keeps_pickup_xml_measure_unlabelled(tmp_path):
    xml_path = tmp_path / "pickup.xml"
    xml_path.write_text(
        '''<?xml version="1.0"?><score-partwise><part-list>
        <score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
        <part id="P1">
          <measure number="1" width="100"><print new-page="yes"/>
            <attributes><divisions>4</divisions>
              <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note default-x="50"><pitch><step>C</step><octave>4</octave></pitch>
              <duration>4</duration><voice>1</voice><staff>1</staff></note>
          </measure>
          <measure number="2" width="100">
            <note default-x="50"><pitch><step>D</step><octave>4</octave></pitch>
              <duration>16</duration><voice>1</voice><staff>1</staff></note>
          </measure>
        </part></score-partwise>''',
        encoding="utf-8",
    )

    page = parse_musicxml_page(
        xml_path,
        page_number=1,
        system_start_measures=[1],
        page_end_measure=1,
    )

    assert [measure["measure"] for measure in page["measures"]] == [1, 2]
    assert [measure["printed_measure"] for measure in page["measures"]] == [0, 1]
    assert page["notes"][1]["xml_measure"] == 2
    assert page["notes"][1]["printed_measure"] == 1
    assert page["notes"][1]["bps_time"] == 1.0


def test_parse_musicxml_still_rejects_entity_expansion(tmp_path):
    xml_path = tmp_path / "entity.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE score-partwise [<!ENTITY unsafe "expanded">]>
<score-partwise><part-list><score-part id="P1"><part-name>&unsafe;</part-name>
</score-part></part-list></score-partwise>""",
        encoding="utf-8",
    )

    with pytest.raises(EntitiesForbidden):
        parse_musicxml_page(xml_path, 1)


def test_build_tie_candidates_pairs_same_staff_voice_and_pitch():
    notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 2.0,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 3,
            "note_id": 12,
            "tie_marks": [{"type": "start"}],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 2.5,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 4,
            "note_id": 12,
            "tie_marks": [{"type": "stop"}],
        },
    ]
    bps_notes = [
        {"note_id": 12, "bps_time": 2.0, "end_time": 3.0, "midi": 60},
    ]

    candidates, issues = build_tie_candidates(notes, bps_notes)

    assert issues == []
    assert candidates[0]["pitch"] == "C4"
    assert candidates[0]["start_meas"] == "2.000"
    assert candidates[0]["end_meas"] == "2.500"
    assert candidates[0]["start_note_candidate"] == 12
    assert candidates[0]["end_note_candidate"] == 12
    assert candidates[0]["status"] == "time_confirmed"


def test_build_tie_candidates_requires_review_for_ambiguous_unison_endpoint():
    notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 2.0,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 3,
            "note_id": 12,
            "note_id_ambiguous": True,
            "tie_marks": [{"type": "start"}],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 2.5,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 4,
            "note_id": 12,
            "tie_marks": [{"type": "stop"}],
        },
    ]
    bps_notes = [
        {"note_id": 12, "bps_time": 2.0, "end_time": 3.0, "midi": 60},
    ]

    candidates, issues = build_tie_candidates(notes, bps_notes)

    assert issues == []
    assert candidates[0]["note_id_ambiguous"] is True
    assert candidates[0]["status"] == "review"


def test_build_tie_candidates_allows_unique_cross_voice_pair():
    notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 4.667,
            "midi": 48,
            "pitch_name": "C3",
            "staff": 2,
            "voice": "3",
            "system": 2,
            "xml_measure": 5,
            "note_id": 20,
            "tie_marks": [{"type": "start"}],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 5.0,
            "midi": 48,
            "pitch_name": "C3",
            "staff": 2,
            "voice": "1",
            "system": 2,
            "xml_measure": 6,
            "note_id": 21,
            "tie_marks": [{"type": "stop"}],
        },
    ]
    bps_notes = [
        {"note_id": 20, "bps_time": 4.667, "end_time": 5.0, "midi": 48},
        {"note_id": 21, "bps_time": 5.0, "end_time": 5.5, "midi": 48},
    ]

    candidates, issues = build_tie_candidates(notes, bps_notes)

    assert issues == []
    assert candidates[0]["pitch"] == "C3"
    assert candidates[0]["start_note_candidate"] == 20
    assert candidates[0]["end_note_candidate"] == 21


def test_parse_musicxml_snaps_dynamic_to_following_note_onset(tmp_path):
    xml_path = tmp_path / "score.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="29" width="120">
      <attributes>
        <divisions>256</divisions>
        <time><beats>3</beats><beat-type>4</beat-type></time>
      </attributes>
      <direction>
        <direction-type><dynamics default-x="41"><f/><mf/><sfz/></dynamics></direction-type>
        <offset sound="no">243</offset>
        <staff>1</staff>
      </direction>
      <note default-x="15">
        <pitch><step>E</step><octave>5</octave></pitch>
        <duration>512</duration><voice>1</voice><staff>1</staff>
      </note>
      <note default-x="71">
        <pitch><step>F</step><octave>5</octave></pitch>
        <duration>128</duration><voice>1</voice><staff>1</staff>
      </note>
      <note default-x="95">
        <rest/><duration>128</duration><voice>1</voice><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>
""",
        encoding="utf-8",
    )

    page = parse_musicxml_page(xml_path, page_number=1)

    assert page["dynamics"][0]["direction_onset"] == 243
    assert page["dynamics"][0]["onset"] == 512
    assert page["dynamics"][0]["onset_source"] == "following_note"
    assert round(page["dynamics"][0]["bps_time"], 3) == 28.667
    assert [event["class"] for event in page["dynamics"]] == [
        "dynamicF",
        "dynamicM",
        "dynamicF",
        "dynamicS",
        "dynamicF",
        "dynamicZ",
    ]


def test_parse_musicxml_applies_mid_measure_clef_by_onset(tmp_path):
    xml_path = tmp_path / "clef-change.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1" width="110">
      <attributes>
        <divisions>256</divisions>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef number="2"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note default-x="15">
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>512</duration><voice>1</voice><staff>1</staff>
      </note>
      <attributes>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <backup><duration>512</duration></backup>
      <note default-x="15">
        <pitch><step>F</step><octave>4</octave></pitch>
        <duration>512</duration><voice>2</voice><staff>2</staff>
      </note>
      <note default-x="79">
        <pitch><step>F</step><octave>4</octave></pitch>
        <duration>256</duration><voice>2</voice><staff>2</staff>
      </note>
    </measure>
  </part>
</score-partwise>
""",
        encoding="utf-8",
    )

    page = parse_musicxml_page(xml_path, page_number=1)
    staff_two = [note for note in page["notes"] if note["staff"] == 2]

    assert staff_two[0]["onset"] == 0
    assert staff_two[0]["clef"]["sign"] == "G"
    assert staff_two[1]["onset"] == 512
    assert staff_two[1]["clef"]["sign"] == "F"


def test_load_yolo_keeps_original_line_number(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(
        "18 0.2 0.3 0.1 0.1\n"
        "\n"
        "25 0.4 0.5 0.02 0.03\n",
        encoding="utf-8",
    )

    boxes = load_yolo(path)

    assert [box["txt_line"] for box in boxes] == [1, 3]
    assert [box["class"] for box in boxes] == ["dynamicF", "fingering1"]


def test_load_yolo_rejects_non_finite_geometry(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 nan 0.5 0.1 0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite geometry"):
        load_yolo(path)


def test_attach_bps_note_ids_matches_time_and_pitch():
    xml_notes = [
        {"bps_time": 1.0, "midi": 60, "staff": 1, "x_norm": 0.2},
        {"bps_time": 1.0, "midi": 64, "staff": 1, "x_norm": 0.2},
    ]
    bps_notes = [
        {"note_id": 10, "bps_time": 1.0, "midi": 60},
        {"note_id": 11, "bps_time": 1.0, "midi": 64},
    ]

    attach_bps_note_ids(xml_notes, bps_notes)

    assert [note["note_id"] for note in xml_notes] == [10, 11]


def test_attach_bps_note_ids_reuses_tied_note_span():
    xml_notes = [
        {"bps_time": 2.0, "midi": 60, "staff": 1, "x_norm": 0.2},
    ]
    bps_notes = [
        {
            "note_id": 12,
            "bps_time": 1.5,
            "end_time": 2.5,
            "midi": 60,
        },
    ]

    attach_bps_note_ids(xml_notes, bps_notes)

    assert xml_notes[0]["note_id"] == 12


def test_attach_bps_note_ids_marks_same_time_same_pitch_unison_ambiguous():
    xml_notes = [
        {"bps_time": 1.0, "midi": 60, "staff": 1, "x_norm": 0.2},
        {"bps_time": 1.0, "midi": 60, "staff": 2, "x_norm": 0.2},
    ]
    bps_notes = [
        {"note_id": 10, "bps_time": 1.0, "midi": 60},
        {"note_id": 11, "bps_time": 1.0, "midi": 60},
    ]

    attach_bps_note_ids(xml_notes, bps_notes)

    assert [note["note_id"] for note in xml_notes] == [10, 11]
    assert all(note["note_id_ambiguous"] for note in xml_notes)


def test_parse_musicxml_page_supports_default_namespace(tmp_path):
    xml_path = tmp_path / "namespaced.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<score-partwise xmlns="http://www.musicxml.org/ns/musicxml">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1" width="100">
    <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
    <note default-x="10"><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><staff>1</staff>
      <notations><slur type="start" number="1"/></notations>
    </note>
  </measure></part>
</score-partwise>""",
        encoding="utf-8",
    )

    page = parse_musicxml_page(xml_path, page_number=1)

    assert len(page["notes"]) == 1
    assert page["notes"][0]["pitch_name"] == "C4"
    assert page["notes"][0]["slur_marks"] == [
        {"type": "start", "number": "1", "orientation": ""}
    ]


def test_build_slur_candidates_pairs_endpoints_and_keeps_bps_time():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 1.0,
            "midi": 67,
            "pitch_name": "G4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 9,
            "slur_marks": [
                {
                    "type": "start",
                    "number": "1",
                    "orientation": "over",
                }
            ],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 1.5,
            "midi": 66,
            "pitch_name": "F#4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 12,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "over",
                }
            ],
        },
    ]
    bps_notes = [
        {
            "note_id": 9,
            "bps_time": 1.0,
            "end_time": 1.5,
            "midi": 67,
        },
        {
            "note_id": 12,
            "bps_time": 1.5,
            "end_time": 1.667,
            "midi": 66,
        },
    ]

    candidates, issues = build_slur_candidates(xml_notes, bps_notes)

    assert issues == []
    assert len(candidates) == 1
    assert candidates[0]["start_meas"] == "1.000"
    assert candidates[0]["end_meas"] == "1.500"
    assert candidates[0]["start_pitch"] == "G4"
    assert candidates[0]["end_pitch"] == "F#4"
    assert candidates[0]["status"] == "time_confirmed"


def test_build_slur_candidates_requires_review_for_ambiguous_unison_endpoint():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 1.0,
            "midi": 67,
            "pitch_name": "G4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 9,
            "note_id_ambiguous": True,
            "slur_marks": [{"type": "start", "number": "1"}],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 1.5,
            "midi": 66,
            "pitch_name": "F#4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 2,
            "note_id": 12,
            "slur_marks": [{"type": "stop", "number": "1"}],
        },
    ]
    bps_notes = [
        {"note_id": 9, "bps_time": 1.0, "end_time": 1.5, "midi": 67},
        {"note_id": 12, "bps_time": 1.5, "end_time": 1.667, "midi": 66},
    ]

    candidates, issues = build_slur_candidates(xml_notes, bps_notes)

    assert issues == []
    assert candidates[0]["note_id_ambiguous"] is True
    assert candidates[0]["status"] == "review"


def test_build_slur_candidates_reports_unpaired_endpoints():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 2.0,
            "midi": 60,
            "pitch_name": "C4",
            "staff": 1,
            "voice": "1",
            "system": 1,
            "xml_measure": 3,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "",
                }
            ],
        }
    ]

    candidates, issues = build_slur_candidates(xml_notes, [])

    assert candidates == []
    assert issues[0]["issue"] == "stop_without_start"


def test_build_slur_candidates_allows_cross_staff_slur():
    xml_notes = [
        {
            "xml_note_sequence": 0,
            "bps_time": 15.0,
            "midi": 64,
            "pitch_name": "E4",
            "staff": 1,
            "voice": "1",
            "system": 2,
            "xml_measure": 16,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "start",
                    "number": "1",
                    "orientation": "under",
                }
            ],
        },
        {
            "xml_note_sequence": 1,
            "bps_time": 15.667,
            "midi": 52,
            "pitch_name": "E3",
            "staff": 2,
            "voice": "1",
            "system": 2,
            "xml_measure": 16,
            "note_id": None,
            "slur_marks": [
                {
                    "type": "stop",
                    "number": "1",
                    "orientation": "under",
                }
            ],
        },
    ]

    candidates, issues = build_slur_candidates(xml_notes, [])

    assert issues == []
    assert len(candidates) == 1
    assert candidates[0]["start_staff"] == 1
    assert candidates[0]["end_staff"] == 2


def _timed_note(
    sequence,
    time,
    pitch,
    midi,
    x_norm,
    *,
    marks=None,
    ties=None,
    fermata=False,
    xml_measure=2,
    system=1,
):
    return {
        "xml_note_sequence": sequence,
        "xml_chord_sequence": sequence,
        "bps_time": time,
        "midi": midi,
        "pitch_name": pitch,
        "diatonic": 28 + sequence,
        "staff": 1,
        "voice": "1",
        "system": system,
        "xml_measure": xml_measure,
        "xml_measure_index": xml_measure,
        "system_measure_index": 0,
        "measure_x_norm": x_norm,
        "x_norm": x_norm,
        "clef": {"sign": "G", "line": 2},
        "note_id": sequence,
        "occurrences": [],
        "slur_marks": marks or [],
        "tie_marks": ties or [],
        "articulation_marks": ["staccato"] if sequence == 0 else [],
        "ornament_marks": [],
        "fermata_marks": [{"placement": "above"}] if fermata else [],
        "tuplet_marks": [],
        "stem": "up",
    }


def test_direct_notations_and_spans_receive_start_and_end_times():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]),
            lower=StaffGeometry(center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]),
            x_left=100,
            x_right=900,
        )
    ]
    notes = [
        _timed_note(0, 1.0, "G4", 67, 0.15, marks=[{"type": "start", "number": "1", "orientation": "over"}]),
        _timed_note(1, 1.5, "F#4", 66, 0.40, marks=[{"type": "stop", "number": "1", "orientation": "over"}], fermata=True),
        _timed_note(2, 2.0, "C4", 60, 0.60, ties=[{"type": "start"}]),
        _timed_note(3, 2.5, "C4", 60, 0.82, ties=[{"type": "stop"}]),
    ]
    bps_notes = [
        {"note_id": index, "bps_time": note["bps_time"], "end_time": note["bps_time"] + 0.25, "midi": note["midi"]}
        for index, note in enumerate(notes)
    ]
    boxes = [
        {"txt_line": 1, "class_id": 16, "class": "articStaccatoAbove", "x": 0.27, "y": 0.12, "w": 0.02, "h": 0.02},
        {"txt_line": 2, "class_id": 37, "class": "fermataAbove", "x": 0.42, "y": 0.11, "w": 0.04, "h": 0.03},
        {"txt_line": 3, "class_id": 87, "class": "slur", "x": 0.32, "y": 0.14, "w": 0.22, "h": 0.04},
        {"txt_line": 4, "class_id": 145, "class": "tie", "x": 0.67, "y": 0.20, "w": 0.20, "h": 0.03},
    ]

    point_rows = match_point_notations(boxes, notes, [], systems, 1000, 500)
    span_rows = match_xml_spans(boxes, notes, bps_notes, systems, 1000, 500)
    by_class = {row["class"]: row for row in point_rows + span_rows}

    assert by_class["articStaccatoAbove"]["start_meas"] == "1.000"
    assert by_class["articStaccatoAbove"]["end_meas"] == "1.000"
    assert by_class["fermataAbove"]["start_meas"] == "1.500"
    assert by_class["slur"]["start_meas"] == "1.000"
    assert by_class["slur"]["end_meas"] == "1.500"
    assert by_class["tie"]["start_meas"] == "2.000"
    assert by_class["tie"]["end_meas"] == "2.500"


def test_tuplet_ends_at_last_note_onset_and_requires_complete_note_ids():
    notes = [
        _timed_note(index, 10.0 + index * 0.25, pitch, midi, 0.2 + index * 0.1)
        for index, (pitch, midi) in enumerate(
            [("G4", 67), ("A4", 69), ("B4", 71)]
        )
    ]
    for index, note in enumerate(notes):
        note["actual_notes"] = 3
        note["end_bps_time"] = 10.25 + index * 0.25
        note["tuplet_marks"] = (
            [{"type": "start", "number": "1"}] if index == 0 else []
        )
    box = {
        "txt_line": 1,
        "class_id": 160,
        "class": "tuplet3",
        "x": 0.34,
        "y": 0.20,
        "w": 0.08,
        "h": 0.03,
    }

    complete = match_tuplets([box], notes, _one_system(), 1000, 500)[0]
    assert complete["status"] == "matched"
    assert complete["xml_time_confirmed"] == "true"
    assert complete["end_meas"] == "10.500"

    notes[1]["note_id"] = None
    incomplete = match_tuplets([box], notes, _one_system(), 1000, 500)[0]
    assert incomplete["status"] == "review"
    assert incomplete["xml_time_confirmed"] == "false"


def test_slur_connected_notes_include_both_complete_endpoint_chords():
    notes = [
        _timed_note(0, 1.0, "E5", 76, 0.20),
        _timed_note(1, 1.0, "G5", 79, 0.20),
        _timed_note(
            2,
            1.0,
            "C5",
            72,
            0.20,
            marks=[{"type": "start", "number": "1", "orientation": "over"}],
        ),
        _timed_note(
            3,
            2.0,
            "D5",
            74,
            0.80,
            marks=[{"type": "stop", "number": "1", "orientation": "over"}],
        ),
        _timed_note(4, 2.0, "F5", 77, 0.80),
        _timed_note(5, 2.0, "A5", 81, 0.80),
    ]
    for note in notes[:3]:
        note["xml_chord_sequence"] = 10
    for note in notes[3:]:
        note["xml_chord_sequence"] = 11
    bps_notes = [
        {
            "note_id": note["note_id"],
            "bps_time": note["bps_time"],
            "end_time": note["bps_time"] + 0.25,
            "midi": note["midi"],
        }
        for note in notes
    ]
    box = {
        "txt_line": 1,
        "class_id": 56,
        "class": "slur",
        "x": 0.50,
        "y": 0.20,
        "w": 0.48,
        "h": 0.04,
    }

    row = match_xml_spans(
        [box], notes, bps_notes, _one_system(), 1000, 500
    )[0]

    assert row["start_note"] == 2
    assert row["end_note"] == 3
    assert row["connected_note"] == "[0, 1, 2, 3, 4, 5]"
    assert json.loads(row["pitches"]) == ["E5", "G5", "C5", "D5", "F5", "A5"]


def test_tie_with_equally_good_xml_targets_stays_in_review():
    notes = [
        _timed_note(0, 2.0, "C4", 60, 0.40, ties=[{"type": "start"}]),
        _timed_note(1, 2.0, "C4", 60, 0.40, ties=[{"type": "start"}]),
        _timed_note(2, 2.5, "C4", 60, 0.60, ties=[{"type": "stop"}]),
        _timed_note(3, 2.5, "C4", 60, 0.60, ties=[{"type": "stop"}]),
    ]
    for index, note in enumerate(notes):
        note["voice"] = "1" if index in {0, 2} else "2"
        note["diatonic"] = 30
    bps_notes = [
        {
            "note_id": note["note_id"],
            "bps_time": note["bps_time"],
            "end_time": note["bps_time"] + 0.25,
            "midi": note["midi"],
        }
        for note in notes
    ]
    box = {
        "txt_line": 1,
        "class_id": 145,
        "class": "tie",
        "x": 0.50,
        "y": 0.20,
        "w": 0.16,
        "h": 0.03,
    }

    row = match_xml_spans(
        [box], notes, bps_notes, _one_system(), 1000, 500
    )[0]

    assert row["status"] == "review"
    assert float(row["candidate_margin"]) < 0.08


def test_cross_page_span_keeps_complete_endpoints_but_requires_review():
    note = _timed_note(0, 10.0, "G4", 67, 0.70, xml_measure=10)
    note["end_bps_time"] = 10.25
    span = {
        "span_id": "S:SPAN1",
        "class": "slur",
        "span_type": "slur",
        "start_meas": "10.000",
        "end_meas": "11.000",
        "start_note": "0",
        "end_note": "9",
        "connected_note": "[0, 9]",
        "start_page": "1",
        "end_page": "2",
        "start_xml_measure": "10",
        "end_xml_measure": "11",
        "start_staff": "1",
        "end_staff": "1",
        "cross_page": "true",
        "status": "paired",
    }
    box = {
        "txt_line": 1,
        "class_id": 56,
        "class": "slur",
        "x": 0.82,
        "y": 0.15,
        "w": 0.14,
        "h": 0.03,
    }

    row = match_cross_page_spans(
        [box], [note], [span], 1, _one_system(), 1000, 500
    )[0]

    assert row["status"] == "review"
    assert row["xml_time_confirmed"] == "true"
    assert row["start_meas"] == "10.000"
    assert row["end_meas"] == "11.000"
    assert row["start_note"] == "0"
    assert row["end_note"] == "9"
    assert row["cross_page_span_id"] == "S:SPAN1"
    assert row["start_xml_page"] == "1"
    assert row["end_xml_page"] == "2"
    assert row["match_source"] == "whole_score_cross_page_span_candidate"


def test_cross_system_slur_segments_share_complete_xml_endpoints():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]),
            lower=StaffGeometry(center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]),
            x_left=100,
            x_right=900,
        ),
        SystemGeometry(
            number=2,
            upper=StaffGeometry(center=450, line_spacing=10, lines=[430, 440, 450, 460, 470]),
            lower=StaffGeometry(center=600, line_spacing=10, lines=[580, 590, 600, 610, 620]),
            x_left=100,
            x_right=900,
        ),
    ]
    notes = [
        _timed_note(
            0,
            10.0,
            "C4",
            60,
            0.70,
            marks=[{"type": "start", "number": "1", "orientation": "over"}],
            xml_measure=10,
            system=1,
        ),
        _timed_note(
            1,
            11.0,
            "D4",
            62,
            0.30,
            marks=[{"type": "stop", "number": "1", "orientation": "over"}],
            xml_measure=11,
            system=2,
        ),
    ]
    bps_notes = [
        {
            "note_id": index,
            "bps_time": note["bps_time"],
            "end_time": note["bps_time"] + 0.25,
            "midi": note["midi"],
        }
        for index, note in enumerate(notes)
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 56,
            "class": "slur",
            "x": 0.78,
            "y": 0.12,
            "w": 0.24,
            "h": 0.04,
        },
        {
            "txt_line": 2,
            "class_id": 56,
            "class": "slur",
            "x": 0.22,
            "y": 0.56,
            "w": 0.24,
            "h": 0.04,
        },
    ]

    rows = match_xml_spans(boxes, notes, bps_notes, systems, 1000, 800)

    assert len(rows) == 2
    by_line = {row["txt_line"]: row for row in rows}
    assert by_line[1]["start_meas"] == "10.000"
    assert by_line[1]["end_meas"] == "11.000"
    assert by_line[2]["start_meas"] == "10.000"
    assert by_line[2]["end_meas"] == "11.000"
    assert by_line[1]["match_source"] == "musicxml_slur_start_endpoint_candidate"
    assert by_line[2]["match_source"] == "musicxml_slur_end_endpoint_candidate"
    assert {row["status"] for row in rows} == {"review"}


def test_term_receives_reviewable_position_and_outside_timeline_flag():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]),
            lower=StaffGeometry(center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]),
            x_left=100,
            x_right=900,
        )
    ]
    note = _timed_note(0, 3.25, "G4", 67, 0.5)
    box = {"txt_line": 1, "class_id": 115, "class": "termDolce", "x": 0.5, "y": 0.15, "w": 0.12, "h": 0.03}

    row = estimate_all_symbol_times([box], [note], [], systems, 1000, 500)[0]

    assert row["start_meas"] == "3.250"
    assert row["end_meas"] == "3.250"
    assert row["status"] == "review"
    assert row["match_source"] == "geometric_nearest_anchor_time_estimate"
    assert row["musical_time"] == 1


def test_detect_systems_finds_paired_staves():
    image = Image.new("L", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    for staff_start in (80, 180, 300, 400):
        for offset in range(5):
            y = staff_start + offset * 10
            draw.line((80, y, 920, y), fill="black", width=2)

    systems = detect_systems(image.convert("RGB"))

    assert len(systems) == 2
    assert systems[0].upper.center < systems[0].lower.center
    assert systems[1].upper.center < systems[1].lower.center


def test_detect_systems_rejects_dense_short_patterns_and_footer_text():
    image = Image.new("L", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    true_staves = (80, 190, 330, 440)
    for staff_start in true_staves:
        for offset in range(5):
            y = staff_start + offset * 12
            draw.line((80, y, 920, y), fill="black", width=2)

    # Short repeated strokes imitate dense chord beams but do not span enough
    # of the page to be accepted as a staff.
    for offset in range(5):
        y = 270 + offset * 10
        for x in range(120, 820, 90):
            draw.line((x, y, x + 22, y), fill="black", width=3)

    # Distributed footer-like text has high total ink on several rows, but no
    # long continuous horizontal line.
    for offset in range(5):
        y = 570 + offset * 10
        for x in range(100, 900, 35):
            draw.line((x, y, x + 8, y), fill="black", width=2)

    systems = detect_systems(image.convert("RGB"))

    assert len(systems) == 2
    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(
            [system.upper.center for system in systems],
            [104, 354],
        )
    )
    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(
            [system.lower.center for system in systems],
            [214, 464],
        )
    )


def test_detect_systems_keeps_staff_extent_when_left_side_is_partly_obscured():
    image = Image.new("L", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    upper_lines = [80, 90, 100, 110, 120]
    lower_lines = [230, 240, 250, 260, 270]
    # Only three rows remain clean across the full system; the other seven are
    # visible only on the right, as happens under dense notation and damage.
    for y in upper_lines[:3]:
        draw.line((80, y, 920, y), fill="black", width=2)
    for y in [*upper_lines[3:], *lower_lines]:
        draw.line((500, y, 920, y), fill="black", width=2)

    systems = detect_systems(image.convert("RGB"))

    assert len(systems) == 1
    assert systems[0].x_left < 120
    assert systems[0].x_right > 880


def test_detect_barlines_uses_continuous_vertical_ink():
    image = Image.new("L", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    upper_lines = [80, 90, 100, 110, 120]
    lower_lines = [230, 240, 250, 260, 270]
    for y in upper_lines + lower_lines:
        draw.line((100, y, 900, y), fill="black", width=2)
    for x in (100, 300, 600, 900):
        draw.line((x, 80, x, 270), fill="black", width=3)

    # A note-like pair of vertical segments has substantial ink but does not
    # continuously connect the two staves, so it must not become a barline.
    draw.line((450, 80, 450, 145), fill="black", width=4)
    draw.line((450, 205, 450, 270), fill="black", width=4)

    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=upper_lines,
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=lower_lines,
            ),
            x_left=100,
            x_right=900,
        )
    ]

    boundaries = detect_barlines(
        image.convert("RGB"),
        systems,
        expected_boundary_counts=[4],
    )

    assert boundaries == [[100, 300, 600, 900]]


def test_align_barlines_from_reference_marks_occluded_line_for_review():
    reference = Image.new("L", (1000, 400), "white")
    target = Image.new("L", (1000, 400), "white")
    reference_draw = ImageDraw.Draw(reference)
    target_draw = ImageDraw.Draw(target)
    upper_lines = [80, 90, 100, 110, 120]
    lower_lines = [230, 240, 250, 260, 270]
    for draw in (reference_draw, target_draw):
        for y in upper_lines + lower_lines:
            draw.line((100, y, 900, y), fill="black", width=2)
    for x in (100, 300, 600, 900):
        reference_draw.line((x, 80, x, 270), fill="black", width=3)
        target_draw.line((x, 80, x, 270), fill="black", width=3)

    # Simulate a barline interrupted by a printed symbol in the target scan.
    target_draw.rectangle((598, 145, 602, 195), fill="white")
    geometry = SystemGeometry(
        number=1,
        upper=StaffGeometry(
            center=100,
            line_spacing=10,
            lines=upper_lines,
        ),
        lower=StaffGeometry(
            center=250,
            line_spacing=10,
            lines=lower_lines,
        ),
        x_left=100,
        x_right=900,
    )

    aligned = align_barlines_from_reference(
        target.convert("RGB"),
        [geometry],
        [geometry],
        [[100, 300, 600, 900]],
    )

    assert [item["x"] for item in aligned[0]] == [100, 300, 600, 900]
    assert aligned[0][1]["status"] == "detected"
    assert aligned[0][2]["status"] == "review_occluded"


def test_snap_notehead_x_ignores_staff_line_and_finds_dense_oval():
    image = Image.new("L", (800, 240), "white")
    draw = ImageDraw.Draw(image)
    lines = [80, 90, 100, 110, 120]
    for y in lines:
        draw.line((50, y, 750, y), fill="black", width=2)
    draw.ellipse((522, 95, 538, 105), fill="black")
    staff = StaffGeometry(
        center=100,
        line_spacing=10,
        lines=lines,
    )

    snapped = snap_notehead_x(
        image.convert("RGB"),
        predicted_x=510,
        predicted_y=100,
        staff=staff,
        search_radius=30,
    )

    assert 528 <= snapped["x"] <= 532


def test_stacked_fingerings_use_distinct_chord_notes():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "fingering5",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 2,
            "class_id": 27,
            "class": "fingering3",
            "x": 0.5,
            "y": 0.20,
            "w": 0.01,
            "h": 0.01,
        },
    ]
    notes = [
        {
            "note_id": 10,
            "system": 1,
            "staff": 1,
            "x_norm": 0.5,
            "bps_time": 2.0,
            "xml_measure": 3,
            "pitch_name": "F5",
            "diatonic": 38,
            "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 11,
            "system": 1,
            "staff": 1,
            "x_norm": 0.5,
            "bps_time": 2.0,
            "xml_measure": 3,
            "pitch_name": "B4",
            "diatonic": 34,
            "clef": {"sign": "G", "line": 2},
        },
    ]

    rows = match_fingerings(
        boxes,
        notes,
        systems,
        image_width=1000,
        image_height=400,
    )

    assert len(rows) == 2
    assert {row["start_note"] for row in rows} == {10, 11}


def test_independent_fingerings_cannot_reuse_the_same_note():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
            ),
            lower=StaffGeometry(
                center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1, "class_id": 25, "class": "fingering1",
            "x": 0.49, "y": 0.25, "w": 0.01, "h": 0.01,
        },
        {
            "txt_line": 2, "class_id": 26, "class": "fingering2",
            "x": 0.50, "y": 0.25, "w": 0.01, "h": 0.01,
        },
    ]
    notes = [
        {
            "note_id": 10, "system": 1, "staff": 1, "x_norm": 0.5,
            "bps_time": 1.0, "xml_measure": 2, "pitch_name": "B4",
            "diatonic": 34, "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 11, "system": 1, "staff": 1, "x_norm": 0.54,
            "bps_time": 1.5, "xml_measure": 2, "pitch_name": "B4",
            "diatonic": 34, "clef": {"sign": "G", "line": 2},
        },
    ]

    rows = match_fingerings(boxes, notes, systems, 1000, 400)

    assert len(rows) == 2
    assert {row["start_note"] for row in rows} == {10, 11}


def test_interstaff_fingering_prefers_staff_side_over_nearest_ledger_note():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
            ),
            lower=StaffGeometry(
                center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [{
        "txt_line": 1, "class_id": 26, "class": "fingering2",
        "x": 0.5, "y": 0.45, "w": 0.01, "h": 0.01,
    }]
    notes = [
        {
            "note_id": 10, "system": 1, "staff": 1, "x_norm": 0.5,
            "bps_time": 1.0, "xml_measure": 2, "pitch_name": "C3",
            "diatonic": 21, "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 11, "system": 1, "staff": 2, "x_norm": 0.5,
            "bps_time": 1.0, "xml_measure": 2, "pitch_name": "C4",
            "diatonic": 28, "clef": {"sign": "F", "line": 4},
        },
    ]

    rows = match_fingerings(boxes, notes, systems, 1000, 400)

    assert len(rows) == 1
    assert rows[0]["start_note"] == 11
    assert rows[0]["xml_staff"] == 2


def test_single_fingering_on_multinote_chord_is_not_autoaccepted():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]
            ),
            lower=StaffGeometry(
                center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [{
        "txt_line": 1, "class_id": 25, "class": "fingering1",
        "x": 0.5, "y": 0.18, "w": 0.01, "h": 0.01,
    }]
    notes = [
        {
            "note_id": 10, "system": 1, "staff": 1, "x_norm": 0.5,
            "bps_time": 1.0, "xml_measure": 1, "pitch_name": "E5",
            "diatonic": 37, "clef": {"sign": "G", "line": 2},
        },
        {
            "note_id": 11, "system": 1, "staff": 1, "x_norm": 0.5,
            "bps_time": 1.0, "xml_measure": 1, "pitch_name": "C5",
            "diatonic": 35, "clef": {"sign": "G", "line": 2},
        },
    ]

    rows = match_fingerings(boxes, notes, systems, 1000, 400)

    assert rows[0]["status"] == "review"
    assert float(rows[0]["confidence"]) < 0.70
    assert rows[0]["match_source"].endswith("ambiguous_chord")


def test_long_crescendo_preserves_written_measure_start_and_end():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(center=100, line_spacing=10, lines=[80, 90, 100, 110, 120]),
            lower=StaffGeometry(center=250, line_spacing=10, lines=[230, 240, 250, 260, 270]),
            x_left=100,
            x_right=900,
        )
    ]
    notes = [
        _timed_note(0, 42.0, "C4", 60, 0.15, xml_measure=43),
        _timed_note(1, 43.0, "D4", 62, 0.85, xml_measure=44),
    ]
    box = {
        "txt_line": 1,
        "class_id": 25,
        "class": "dynamicCrescendoLong",
        "x": 0.5,
        "y": 0.15,
        "w": 0.7,
        "h": 0.03,
    }

    rows = estimate_all_symbol_times([box], notes, [], systems, 1000, 500)

    assert rows[0]["start_xml_measure"] == 43
    assert rows[0]["end_xml_measure"] == 44
    assert rows[0]["start_meas"] == "42.000"
    assert rows[0]["end_meas"] == "43.000"


def test_unresolved_fingering_semantics_are_blank():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 29,
            "class": "fingering5",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
    ]

    rows = unresolved_fingering_rows(boxes, systems, image_height=400)

    assert rows[0]["class"] == "fingering5"
    assert rows[0]["musical_time"] == 0
    assert rows[0]["start_meas"] == ""
    assert rows[0]["start_note"] == ""
    assert rows[0]["connected_note"] == ""
    assert rows[0]["status"] == "unresolved"


def test_official_csv_has_only_bps_omr_fields(tmp_path):
    path = tmp_path / "output.csv"
    row = {
        "class_id": 18,
        "x": "0.2",
        "y": "0.3",
        "w": "0.01",
        "h": "0.02",
        "class": "dynamicF",
        "musical_time": 0,
        "start_meas": "0.667",
        "end_meas": "0.667",
        "start_note": "NA",
        "end_note": "NA",
        "connected_note": "NA",
        "stem_dir": "NA",
        "xml_measure": 1,
        "status": "matched",
        "confidence": "1.000",
    }

    write_csv(path, [row])

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        output_rows = list(reader)

    assert reader.fieldnames == OUTPUT_FIELDS
    assert output_rows[0]["class"] == "dynamicF"
    assert "xml_measure" not in output_rows[0]
    assert "status" not in output_rows[0]


def test_load_categories_uses_notes_json_names(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(
        '{"categories":[{"id":56,"name":"slur"}]}',
        encoding="utf-8",
    )

    assert load_categories(path) == {56: "slur"}


def test_all_symbol_policy_leaves_undocumented_flags_blank():
    systems = [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(
                center=100,
                line_spacing=10,
                lines=[80, 90, 100, 110, 120],
            ),
            lower=StaffGeometry(
                center=250,
                line_spacing=10,
                lines=[230, 240, 250, 260, 270],
            ),
            x_left=100,
            x_right=900,
        )
    ]
    boxes = [
        {
            "txt_line": 1,
            "class_id": 23,
            "class": "fermataAbove",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 2,
            "class_id": 62,
            "class": "tempoInTempo",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
        {
            "txt_line": 3,
            "class_id": 107,
            "class": "tie",
            "x": 0.5,
            "y": 0.15,
            "w": 0.01,
            "h": 0.01,
        },
    ]

    rows = conservative_all_symbol_rows(
        boxes,
        systems,
        image_height=400,
    )

    assert rows[0]["musical_time"] == 0
    assert rows[1]["musical_time"] == 1
    assert rows[1]["start_note"] == "NA"
    assert rows[2]["musical_time"] == 0
    assert rows[2]["start_note"] == ""
    assert all(row["stem_dir"] == "NA" for row in rows)


def test_attach_repeat_occurrences_preserves_both_bps_note_ids():
    xml_notes = [
        {
            "xml_measure": 2,
            "xml_measure_index": 2,
            "bps_time": 1.5,
            "xml_note_sequence": 7,
            "staff": 1,
            "midi": 60,
            "x_norm": 0.4,
        }
    ]
    mapping = [
        {
            "written_measure": "2",
            "written_measure_index": "2",
            "unfolded_measure_index": "2",
            "repeat_occurrence": "1",
            "repeat_occurrence_count": "2",
            "repeat_group_id": "R01",
            "mapping_status": "matched_fingerprint",
        },
        {
            "written_measure": "2",
            "written_measure_index": "2",
            "unfolded_measure_index": "5",
            "repeat_occurrence": "2",
            "repeat_occurrence_count": "2",
            "repeat_group_id": "R01",
            "mapping_status": "matched_fingerprint",
        },
    ]
    bps_notes = [
        {"note_id": 10, "bps_time": 1.5, "end_time": 1.75, "midi": 60},
        {"note_id": 20, "bps_time": 4.5, "end_time": 4.75, "midi": 60},
    ]

    expanded = attach_repeat_occurrences(xml_notes, mapping, bps_notes)

    assert [note["note_id"] for note in expanded] == [10, 20]
    assert [item["note_id"] for item in xml_notes[0]["occurrences"]] == [10, 20]
    assert xml_notes[0]["note_id"] == 10


def test_attach_repeat_occurrences_uses_measure_index_and_timeline_offset():
    xml_notes = [{
        "xml_measure": 49,
        "xml_measure_index": 49,
        "timeline_offset": 1,
        "measure_within": 0.5,
        "bps_time": 49.5,
        "xml_note_sequence": 8,
        "staff": 2,
        "midi": 63,
        "x_norm": 0.7,
    }]
    mapping = [{
        "written_measure": "49",
        "written_measure_index": "49",
        "unfolded_measure_index": "49",
        "repeat_occurrence": "1",
        "repeat_occurrence_count": "1",
        "repeat_group_id": "",
        "mapping_status": "matched_fingerprint",
    }]
    bps_notes = [{
        "note_id": 588,
        "bps_time": 49.5,
        "end_time": 49.833,
        "midi": 63,
    }]

    expanded = attach_repeat_occurrences(xml_notes, mapping, bps_notes)

    assert expanded[0]["bps_time"] == 49.5
    assert expanded[0]["note_id"] == 588
    assert xml_notes[0]["note_id"] == 588
