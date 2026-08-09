import csv
import io

from PIL import Image, ImageDraw

from bps_xml_alignment import (
    OUTPUT_FIELDS,
    StaffGeometry,
    SystemGeometry,
    align_barlines_from_reference,
    attach_bps_note_ids,
    attach_repeat_occurrences,
    build_slur_candidates,
    build_tie_candidates,
    conservative_all_symbol_rows,
    detect_barlines,
    detect_systems,
    load_categories,
    load_yolo,
    match_fingerings,
    match_point_notations,
    match_xml_spans,
    estimate_all_symbol_times,
    note_pixel_position,
    parse_musicxml_page,
    snap_notehead_x,
    unresolved_fingering_rows,
    write_csv,
)


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
        <direction-type><dynamics default-x="41"><f/></dynamics></direction-type>
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
    sequence, time, pitch, midi, x_norm, *, marks=None, ties=None, fermata=False
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
        "system": 1,
        "xml_measure": 2,
        "xml_measure_index": 2,
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


def test_unknown_class_receives_reviewable_geometry_time_estimate():
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

    assert rows[0]["musical_time"] == ""
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
