"""Local BPS-OMR alignment prototype.

The prototype aligns selected YOLO glyph boxes on a scanned score page with
MusicXML/BPSD semantics:

* dynamicF, dynamicP, dynamicS are expanded from MusicXML dynamics and matched
  in reading order inside each detected piano system.
* fingering1..5 are not present in the source MusicXML.  They are associated
  with the nearest rendered MusicXML/BPSD note and are explicitly marked as
  inferred.

The Streamlit application calls :func:`run_alignment`; CLI and web therefore
share the same alignment logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from bpsd_aligner.bps_omr_schema import musical_time_for_class
from bpsd_aligner.candidate_scoring import (
    greedy_pairs as _greedy_pairs,
    mutual_geometry_pairs as _mutual_geometry_pairs,
)
from bpsd_aligner.geometry import (
    StaffGeometry,
    SystemGeometry,
    align_barlines_from_reference,
    assign_system,
    detect_barlines,
    detect_systems,
)
from bpsd_aligner.musicxml import (
    child as _xml_child,
    children as _xml_children,
    child_text as _xml_child_text,
    local_name as _shared_local_name,
    parse_musicxml,
)
from bpsd_aligner.overlay import (
    render_alignment_overlay,
    write_alignment_overlay,
)
from bpsd_aligner.span_semantics import endpoint_note_ids, index_chord_members
from bpsd_aligner.schema import BPS_OMR_FIELDS
from bpsd_aligner.thresholds import auto_accept_threshold
from bpsd_aligner.repeat_mapping import repeat_mapping_is_safe


DYNAMIC_CLASS_BY_GLYPH = {
    "f": (18, "dynamicF"),
    "m": (31, "dynamicM"),
    "p": (20, "dynamicP"),
    "r": (33, "dynamicR"),
    "s": (21, "dynamicS"),
    "z": (36, "dynamicZ"),
}

FINGERING_CLASSES = {
    25: "fingering1",
    26: "fingering2",
    27: "fingering3",
    28: "fingering4",
    29: "fingering5",
}

DYNAMIC_CLASS_NAMES = {
    class_name for _class_id, class_name in DYNAMIC_CLASS_BY_GLYPH.values()
}
FINGERING_CLASS_NAMES = {f"fingering{digit}" for digit in range(1, 6)}
FINGERING_DY_WEIGHT = 0.18
FINGERING_WRONG_STAFF_PENALTY_RATIO = 0.02
FINGERING_STRICT_STAFF = False
FINGERING_AUTO_ACCEPT_THRESHOLD = 0.95

POINT_NOTATION_RULES = {
    "articStaccatissimo": ("articulation", "staccatissimo"),
    "articStaccato": ("articulation", "staccato"),
    "articAccent": ("articulation", "accent"),
    "articMarcato": ("articulation", "strong-accent"),
    "fermata": ("fermata", "fermata"),
    "ornamentTrill": ("ornament", "trill-mark"),
    "ornamentShortTrill": ("ornament", "trill-mark"),
    "ornamentTurnInverted": ("ornament", "inverted-turn"),
    "ornamentTurn": ("ornament", "turn"),
}
GEOMETRIC_SPAN_CLASSES = {"slur", "tie", "ottavaBracket"}

TARGET_CLASSES = {
    **{class_id: name for class_id, name in DYNAMIC_CLASS_BY_GLYPH.values()},
    **FINGERING_CLASSES,
}

OUTPUT_FIELDS = BPS_OMR_FIELDS

DETAILED_OUTPUT_FIELDS = [
    *OUTPUT_FIELDS,
    "txt_line",
    "system",
    "xml_measure",
    "start_xml_measure",
    "end_xml_measure",
    "xml_symbol",
    "xml_staff",
    "target_type",
    "note_ids",
    "pitches",
    "repeat_occurrences_json",
    "repeat_occurrence_count",
    "repeat_group_id",
    "repeat_mapping_status",
    "match_source",
    "confidence",
    "match_score",
    "confidence_calibrated",
    "geometry_score",
    "candidate_margin",
    "count_agreement",
    "xml_time_confirmed",
    "status",
    "target_x_px",
    "target_y_px",
    "end_target_x_px",
    "end_target_y_px",
    "cross_page_span_id",
    "start_xml_page",
    "end_xml_page",
    "review_candidate_set_id",
    "review_note_candidates_json",
]

SLUR_CANDIDATE_FIELDS = [
    "candidate_id",
    "start_meas",
    "end_meas",
    "start_pitch",
    "end_pitch",
    "start_xml_measure",
    "end_xml_measure",
    "start_system",
    "end_system",
    "start_staff",
    "end_staff",
    "start_voice",
    "end_voice",
    "start_note_candidate",
    "end_note_candidate",
    "start_note_match",
    "end_note_match",
    "note_id_ambiguous",
    "orientation",
    "status",
]

TIE_CANDIDATE_FIELDS = [
    "candidate_id",
    "start_meas",
    "end_meas",
    "pitch",
    "start_xml_measure",
    "end_xml_measure",
    "system",
    "staff",
    "voice",
    "start_note_candidate",
    "end_note_candidate",
    "start_note_match",
    "end_note_match",
    "note_id_ambiguous",
    "status",
]

def _local_name(tag: str) -> str:
    return _shared_local_name(tag)


def _float_text(element: ET.Element | None, default: float = 0.0) -> float:
    if element is None or element.text is None:
        return default
    return float(element.text)


def _int_text(element: ET.Element | None, default: int = 0) -> int:
    return int(round(_float_text(element, default)))


def load_categories(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {}
    for item in data.get("categories", []):
        class_id = int(item["id"])
        class_name = str(item["name"]).strip()
        if not class_name:
            raise ValueError(f"Empty class name for class_id {class_id}")
        categories[class_id] = class_name
    if not categories:
        raise ValueError("notes.json does not contain categories")
    return categories


def load_yolo(
    path: Path,
    categories: dict[int, str] | None = None,
) -> list[dict]:
    boxes = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"YOLO line {line_number} does not have 5 fields")
        class_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"YOLO line {line_number} contains non-finite geometry")
        boxes.append(
            {
                "txt_line": line_number,
                "class_id": class_id,
                "class": (
                    categories.get(class_id, "")
                    if categories is not None
                    else TARGET_CLASSES.get(class_id, "")
                ),
                "x": x,
                "y": y,
                "w": width,
                "h": height,
            }
        )
    return boxes



def _pitch_midi(pitch: ET.Element) -> tuple[int, str, int]:
    step = (_xml_child_text(pitch, "step", "C") or "C").strip()
    alter = _int_text(_xml_child(pitch, "alter"), 0)
    octave = _int_text(_xml_child(pitch, "octave"), 4)
    semitone = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }[step]
    midi = (octave + 1) * 12 + semitone + alter
    accidental = "#" if alter == 1 else "b" if alter == -1 else ""
    return midi, f"{step}{accidental}{octave}", octave * 7 + "CDEFGAB".index(step)


def _measure_number(measure: ET.Element, fallback: int) -> int:
    raw = measure.attrib.get("number", str(fallback))
    digits = "".join(character for character in raw if character.isdigit())
    return int(digits) if digits else fallback


def parse_musicxml_page(
    xml_path: Path,
    page_number: int = 1,
    *,
    system_start_measures: list[int] | None = None,
    page_end_measure: int | None = None,
) -> dict:
    """Parse score events for one scanned page.

    MusicXML page and system breaks are edition-specific and can differ from
    the scanned score.  When ``system_start_measures`` is supplied, those
    printed measure numbers define the systems on the scanned page while the
    note semantics still come from the corresponding MusicXML measures.
    """

    root = parse_musicxml(xml_path)
    part = next(
        element for element in root.iter() if _local_name(element.tag) == "part"
    )

    divisions = 1
    beats = 4
    beat_type = 4
    current_page = 0
    current_system = 0
    clefs = {
        1: {"sign": "G", "line": 2},
        2: {"sign": "F", "line": 4},
    }
    xml_note_sequence = 0
    xml_chord_sequence = 0
    current_chord_sequence = -1

    raw_measures = []
    for measure_index, measure in enumerate(
        [child for child in part if _local_name(child.tag) == "measure"],
        start=1,
    ):
        measure_start_clefs = {
            staff: dict(clef)
            for staff, clef in clefs.items()
        }
        print_element = next(
            (
                child
                for child in measure
                if _local_name(child.tag) == "print"
            ),
            None,
        )
        if print_element is not None and print_element.attrib.get("new-page") == "yes":
            current_page += 1
            current_system = 1
        elif (
            print_element is not None
            and print_element.attrib.get("new-system") == "yes"
        ):
            current_system += 1
        elif current_page == 0:
            current_page = 1
            current_system = 1

        attributes = next(
            (
                child
                for child in measure
                if _local_name(child.tag) == "attributes"
            ),
            None,
        )
        if attributes is not None:
            if _xml_child(attributes, "divisions") is not None:
                divisions = _int_text(_xml_child(attributes, "divisions"), divisions)
            time_element = _xml_child(attributes, "time")
            if time_element is not None:
                beats = _int_text(_xml_child(time_element, "beats"), beats)
                beat_type = _int_text(_xml_child(time_element, "beat-type"), beat_type)
        nominal_duration = divisions * beats * 4 / beat_type
        cursor = 0.0
        max_cursor = 0.0
        last_note_onset = 0.0
        last_note_x = 0.0
        notes = []
        rests = []
        dynamics = []
        text_directions = []
        direction_markers = []
        clef_events: dict[int, list[tuple[float, dict]]] = defaultdict(list)

        for child in measure:
            name = _local_name(child.tag)

            if name == "attributes":
                for clef in _xml_children(child, "clef"):
                    staff_number = int(clef.attrib.get("number", "1"))
                    clef_value = {
                        "sign": _xml_child_text(clef, "sign", "G") or "G",
                        "line": _int_text(_xml_child(clef, "line"), 2),
                    }
                    clefs[staff_number] = clef_value
                    clef_events[staff_number].append(
                        (cursor, dict(clef_value))
                    )

            elif name == "backup":
                cursor -= _float_text(_xml_child(child, "duration"))

            elif name == "forward":
                cursor += _float_text(_xml_child(child, "duration"))
                max_cursor = max(max_cursor, cursor)

            elif name == "direction":
                offset = _float_text(_xml_child(child, "offset"))
                onset = cursor + offset
                staff = _int_text(_xml_child(child, "staff"), 1)
                for direction_type in _xml_children(child, "direction-type"):
                    dynamics_element = _xml_child(direction_type, "dynamics")
                    if dynamics_element is not None:
                        default_x = float(
                            dynamics_element.attrib.get("default-x", "0")
                        )
                        for dynamic_element in list(dynamics_element):
                            symbol = _local_name(dynamic_element.tag)
                            glyphs = [
                                glyph.lower()
                                for glyph in symbol
                                if glyph.lower() in DYNAMIC_CLASS_BY_GLYPH
                            ]
                            for component_index, glyph in enumerate(glyphs):
                                class_id, class_name = DYNAMIC_CLASS_BY_GLYPH[glyph]
                                dynamics.append(
                                    {
                                        "class_id": class_id,
                                        "class": class_name,
                                        "glyph": glyph,
                                        "xml_symbol": symbol,
                                        "component_index": component_index,
                                        "onset": onset,
                                        "direction_onset": onset,
                                        "staff": staff,
                                        "anchor_measure_x": default_x,
                                        "measure_x": default_x + component_index * 4,
                                    }
                                )

                    for words in _xml_children(direction_type, "words"):
                        text = (words.text or "").strip()
                        if text:
                            text_directions.append(
                                {
                                    "kind": "words",
                                    "text": text,
                                    "onset": onset,
                                    "staff": staff,
                                    "measure_x": float(
                                        words.attrib.get(
                                            "default-x",
                                            child.attrib.get("default-x", "0"),
                                        )
                                    ),
                                }
                            )

                    for metronome in _xml_children(direction_type, "metronome"):
                        beat_unit = _xml_child_text(metronome, "beat-unit").strip()
                        per_minute = _xml_child_text(metronome, "per-minute").strip()
                        text_directions.append(
                            {
                                "kind": "metronome",
                                "text": "=".join(
                                    value for value in (beat_unit, per_minute) if value
                                ),
                                "onset": onset,
                                "staff": staff,
                                "measure_x": float(
                                    metronome.attrib.get(
                                        "default-x",
                                        child.attrib.get("default-x", "0"),
                                    )
                                ),
                            }
                        )

                    for marker_name, marker_kind in (
                        ("octave-shift", "octave"),
                        ("pedal", "pedal"),
                        ("wedge", "wedge"),
                    ):
                        for marker in _xml_children(direction_type, marker_name):
                            direction_markers.append(
                                {
                                    "kind": marker_kind,
                                    "type": marker.attrib.get("type", "").lower(),
                                    "number": marker.attrib.get("number", "1"),
                                    "size": marker.attrib.get("size", ""),
                                    "line": marker.attrib.get("line", ""),
                                    "sign": marker.attrib.get("sign", ""),
                                    "onset": onset,
                                    "staff": staff,
                                    "measure_x": float(
                                        marker.attrib.get(
                                            "default-x",
                                            child.attrib.get("default-x", "0"),
                                        )
                                    ),
                                }
                            )

            elif name == "note":
                is_chord = _xml_child(child, "chord") is not None
                grace_element = _xml_child(child, "grace")
                if not is_chord:
                    current_chord_sequence = xml_chord_sequence
                    xml_chord_sequence += 1
                duration = _float_text(_xml_child(child, "duration"))
                onset = last_note_onset if is_chord else cursor
                if not is_chord:
                    last_note_onset = onset
                default_x = float(child.attrib.get("default-x", last_note_x))
                if not is_chord:
                    last_note_x = default_x
                staff = _int_text(_xml_child(child, "staff"), 1)
                voice = (_xml_child_text(child, "voice", "1") or "1").strip()
                note_clef = dict(
                    measure_start_clefs.get(
                        staff,
                        measure_start_clefs.get(1, clefs[1]),
                    )
                )
                for event_onset, event_clef in sorted(
                    clef_events.get(staff, []),
                    key=lambda event: event[0],
                ):
                    if event_onset <= onset:
                        note_clef = dict(event_clef)
                pitch = _xml_child(child, "pitch")
                if pitch is not None:
                    midi, pitch_name, diatonic = _pitch_midi(pitch)
                    slur_marks = []
                    tie_marks = []
                    articulation_marks = []
                    ornament_marks = []
                    wavy_line_marks = []
                    fermata_marks = []
                    tuplet_marks = []
                    notations = _xml_child(child, "notations")
                    if notations is not None:
                        for notation in notations:
                            notation_name = _local_name(notation.tag)
                            if notation_name == "slur":
                                slur_marks.append(
                                    {
                                        "type": notation.attrib.get("type", ""),
                                        "number": notation.attrib.get("number", "1"),
                                        "orientation": notation.attrib.get(
                                            "orientation",
                                            "",
                                        ),
                                    }
                                )
                            elif notation_name == "tied":
                                tie_marks.append(
                                    {
                                        "type": notation.attrib.get("type", ""),
                                    }
                                )
                            elif notation_name == "articulations":
                                articulation_marks.extend(
                                    _local_name(mark.tag)
                                    for mark in notation
                                )
                            elif notation_name == "ornaments":
                                for mark in notation:
                                    mark_name = _local_name(mark.tag)
                                    ornament_marks.append(mark_name)
                                    if mark_name == "wavy-line":
                                        wavy_line_marks.append(
                                            {
                                                "type": mark.attrib.get("type", ""),
                                                "number": mark.attrib.get("number", "1"),
                                                "placement": mark.attrib.get(
                                                    "placement", ""
                                                ),
                                            }
                                        )
                            elif notation_name == "fermata":
                                fermata_marks.append(
                                    {
                                        "type": (notation.text or "normal").strip(),
                                        "placement": notation.attrib.get(
                                            "placement",
                                            "",
                                        ),
                                    }
                                )
                            elif notation_name == "tuplet":
                                tuplet_marks.append(
                                    {
                                        "type": notation.attrib.get("type", ""),
                                        "number": notation.attrib.get("number", "1"),
                                        "placement": notation.attrib.get(
                                            "placement",
                                            "",
                                        ),
                                    }
                                )
                    time_modification = _xml_child(child, "time-modification")
                    actual_notes = (
                        _int_text(_xml_child(time_modification, "actual-notes"), 0)
                        if time_modification is not None
                        else 0
                    )
                    normal_notes = (
                        _int_text(_xml_child(time_modification, "normal-notes"), 0)
                        if time_modification is not None
                        else 0
                    )
                    if not tie_marks:
                        tie_marks = [
                            {"type": tie.attrib.get("type", "")}
                            for tie in _xml_children(child, "tie")
                        ]
                    notes.append(
                        {
                            "xml_note_sequence": xml_note_sequence,
                            "xml_chord_sequence": current_chord_sequence,
                            "onset": onset,
                            "duration": duration,
                            "staff": staff,
                            "voice": voice,
                            "midi": midi,
                            "pitch_name": pitch_name,
                            "diatonic": diatonic,
                            "measure_x": default_x,
                            "clef": note_clef,
                            "stem": _xml_child_text(child, "stem").strip(),
                            "slur_marks": slur_marks,
                            "tie_marks": tie_marks,
                            "articulation_marks": articulation_marks,
                            "ornament_marks": ornament_marks,
                            "wavy_line_marks": wavy_line_marks,
                            "fermata_marks": fermata_marks,
                            "tuplet_marks": tuplet_marks,
                            "actual_notes": actual_notes,
                            "normal_notes": normal_notes,
                            "accidental": _xml_child_text(child, "accidental").strip(),
                            "note_type": _xml_child_text(child, "type").strip(),
                            "beam_values": [
                                (beam.text or "").strip()
                                for beam in _xml_children(child, "beam")
                            ],
                            "is_grace": grace_element is not None,
                            "grace_slash": (
                                grace_element is not None
                                and grace_element.attrib.get("slash", "no") == "yes"
                            ),
                        }
                    )
                    xml_note_sequence += 1
                elif _xml_child(child, "rest") is not None:
                    fermata_marks = []
                    notations = _xml_child(child, "notations")
                    if notations is not None:
                        for notation in notations:
                            if _local_name(notation.tag) == "fermata":
                                fermata_marks.append(
                                    {
                                        "type": (notation.text or "normal").strip(),
                                        "placement": notation.attrib.get(
                                            "placement",
                                            "",
                                        ),
                                    }
                                )
                    rests.append(
                        {
                            "xml_chord_sequence": current_chord_sequence,
                            "onset": onset,
                            "duration": duration,
                            "staff": staff,
                            "voice": voice,
                            "measure_x": default_x,
                            "fermata_marks": fermata_marks,
                        }
                    )
                if not is_chord:
                    cursor += duration
                    max_cursor = max(max_cursor, cursor)

        for dynamic in dynamics:
            following_notes = sorted(
                (
                    note
                    for note in notes
                    if note["staff"] == dynamic["staff"]
                    and note["measure_x"] >= dynamic["anchor_measure_x"]
                ),
                key=lambda note: (
                    note["measure_x"],
                    note["onset"],
                    note["xml_note_sequence"],
                ),
            )
            if following_notes:
                dynamic["onset"] = following_notes[0]["onset"]
                dynamic["onset_source"] = "following_note"
            else:
                dynamic["onset_source"] = "direction_offset_fallback"

        raw_measures.append(
            {
                "page": current_page,
                "system": current_system,
                "measure": _measure_number(measure, measure_index),
                "measure_index": measure_index,
                "width": float(measure.attrib.get("width", "0")),
                "nominal_duration": nominal_duration,
                "actual_duration": max_cursor,
                "notes": notes,
                "rests": rests,
                "dynamics": dynamics,
                "text_directions": text_directions,
                "direction_markers": direction_markers,
            }
        )

    xml_layout_measures = [
        measure for measure in raw_measures if measure["page"] == page_number
    ]
    first_measure = raw_measures[0]
    score_has_pickup = (
        first_measure["actual_duration"] < first_measure["nominal_duration"]
    )
    anchors = list(system_start_measures or [])
    if anchors:
        if any(not isinstance(value, int) or value < 1 for value in anchors):
            raise ValueError("Scan system-start measures must be positive integers")
        if anchors != sorted(set(anchors)):
            raise ValueError(
                "Scan system-start measures must be strictly increasing"
            )
        if page_end_measure is None:
            if not xml_layout_measures:
                raise ValueError(f"MusicXML page {page_number} does not exist")
            last_xml_system = max(
                int(measure["system"]) for measure in xml_layout_measures
            )
            last_system_measures = [
                measure
                for measure in xml_layout_measures
                if int(measure["system"]) == last_xml_system
            ]
            pickup_count = sum(
                measure["measure_index"] == 1 and score_has_pickup
                for measure in last_system_measures
            )
            page_end_measure = (
                anchors[-1] + len(last_system_measures) - pickup_count - 1
            )
        if page_end_measure < anchors[-1]:
            raise ValueError(
                "The scanned page end measure cannot precede its last system start"
            )

        xml_system_numbers = sorted(
            {int(measure["system"]) for measure in xml_layout_measures}
        )
        if len(xml_system_numbers) != len(anchors):
            raise ValueError(
                f"MusicXML page {page_number} has {len(xml_system_numbers)} systems, "
                f"but {len(anchors)} printed system anchors were supplied"
            )

        page_measures = []
        for system_index, (xml_system, printed_start) in enumerate(
            zip(xml_system_numbers, anchors, strict=True), start=1
        ):
            printed_end = (
                anchors[system_index] - 1
                if system_index < len(anchors)
                else page_end_measure
            )
            xml_system_measures = [
                measure
                for measure in xml_layout_measures
                if int(measure["system"]) == xml_system
            ]
            pickup_count = sum(
                measure["measure_index"] == 1 and score_has_pickup
                for measure in xml_system_measures
            )
            printed_count = printed_end - printed_start + 1
            if len(xml_system_measures) != printed_count + pickup_count:
                raise ValueError(
                    f"Scanned system {system_index} spans printed measures "
                    f"{printed_start}-{printed_end} ({printed_count} measures), "
                    f"but MusicXML system {xml_system} contains "
                    f"{len(xml_system_measures) - pickup_count} numbered measures"
                )
            system_measures = []
            for local_index, measure in enumerate(xml_system_measures):
                system_measures.append(
                    {
                        **measure,
                        "page": page_number,
                        "system": system_index,
                        "printed_measure": (
                            printed_start + local_index - pickup_count
                        ),
                    }
                )
            page_measures.extend(system_measures)
        layout_source = "scan_printed_measure_anchors"
    else:
        page_measures = [
            {**measure, "printed_measure": measure["measure"]}
            for measure in xml_layout_measures
        ]
        layout_source = "musicxml_page_layout"
    if not page_measures:
        raise ValueError(f"MusicXML page {page_number} does not exist")
    # BPSD starts a complete first measure at 1.000, while an anacrusis uses
    # measure 0. This must follow document order, not the displayed measure
    # number, which may be non-numeric or independently renumbered.
    timeline_offset = 0 if score_has_pickup else 1

    system_offsets: dict[int, float] = defaultdict(float)
    system_widths: dict[int, float] = defaultdict(float)
    system_measure_counts: dict[int, int] = defaultdict(int)
    for measure in page_measures:
        measure["system_x"] = system_offsets[measure["system"]]
        measure["system_measure_index"] = system_measure_counts[
            measure["system"]
        ]
        system_measure_counts[measure["system"]] += 1
        system_offsets[measure["system"]] += measure["width"]
        system_widths[measure["system"]] += measure["width"]

    notes = []
    rests = []
    dynamics = []
    text_directions = []
    direction_markers = []
    for measure in page_measures:
        nominal = measure["nominal_duration"]
        pickup_shift = 0.0
        if measure["measure_index"] == 1 and measure["actual_duration"] < nominal:
            pickup_shift = nominal - measure["actual_duration"]

        for raw_note in measure["notes"]:
            within = (pickup_shift + raw_note["onset"]) / nominal
            event = dict(raw_note)
            event.update(
                {
                    "page": page_number,
                    "system": measure["system"],
                    "xml_measure": measure["measure"],
                    "printed_measure": measure["printed_measure"],
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
                    "duration_measures": (
                        raw_note["duration"] / nominal if nominal > 0 else 0.0
                    ),
                    "system_measure_index": measure["system_measure_index"],
                    "measure_x_norm": (
                        raw_note["measure_x"] / measure["width"]
                        if measure["width"] > 0
                        else 0.0
                    ),
                    "x_norm": (
                        measure["system_x"] + raw_note["measure_x"]
                    )
                    / system_widths[measure["system"]],
                }
            )
            notes.append(event)

        for raw_rest in measure["rests"]:
            within = (pickup_shift + raw_rest["onset"]) / nominal
            event = dict(raw_rest)
            event.update(
                {
                    "page": page_number,
                    "system": measure["system"],
                    "xml_measure": measure["measure"],
                    "printed_measure": measure["printed_measure"],
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
                    "system_measure_index": measure["system_measure_index"],
                    "measure_x_norm": (
                        raw_rest["measure_x"] / measure["width"]
                        if measure["width"] > 0
                        else 0.0
                    ),
                    "x_norm": (
                        measure["system_x"] + raw_rest["measure_x"]
                    )
                    / system_widths[measure["system"]],
                }
            )
            rests.append(event)

        for raw_dynamic in measure["dynamics"]:
            within = (pickup_shift + raw_dynamic["onset"]) / nominal
            event = dict(raw_dynamic)
            event.update(
                {
                    "page": page_number,
                    "system": measure["system"],
                    "xml_measure": measure["measure"],
                    "printed_measure": measure["printed_measure"],
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
                    "x_norm": (
                        measure["system_x"] + raw_dynamic["measure_x"]
                    )
                    / system_widths[measure["system"]],
                }
            )
            dynamics.append(event)

        for raw_direction in measure["text_directions"]:
            within = (pickup_shift + raw_direction["onset"]) / nominal
            event = dict(raw_direction)
            event.update(
                {
                    "page": page_number,
                    "system": measure["system"],
                    "xml_measure": measure["measure"],
                    "printed_measure": measure["printed_measure"],
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
                    "system_measure_index": measure["system_measure_index"],
                    "measure_x_norm": (
                        raw_direction["measure_x"] / measure["width"]
                        if measure["width"] > 0
                        else 0.0
                    ),
                    "x_norm": (
                        measure["system_x"] + raw_direction["measure_x"]
                    )
                    / system_widths[measure["system"]],
                }
            )
            text_directions.append(event)

        for raw_marker in measure["direction_markers"]:
            within = (pickup_shift + raw_marker["onset"]) / nominal
            event = dict(raw_marker)
            event.update(
                {
                    "page": page_number,
                    "system": measure["system"],
                    "xml_measure": measure["measure"],
                    "printed_measure": measure["printed_measure"],
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
                    "system_measure_index": measure["system_measure_index"],
                    "measure_x_norm": (
                        raw_marker["measure_x"] / measure["width"]
                        if measure["width"] > 0
                        else 0.0
                    ),
                    "x_norm": (
                        measure["system_x"] + raw_marker["measure_x"]
                    )
                    / system_widths[measure["system"]],
                }
            )
            direction_markers.append(event)

    return {
        "measures": page_measures,
        "notes": notes,
        "rests": rests,
        "dynamics": dynamics,
        "text_directions": text_directions,
        "direction_markers": direction_markers,
        "system_widths": dict(system_widths),
        "layout_source": layout_source,
        "system_start_measures": anchors,
        "page_end_measure": (
            page_end_measure if anchors else page_measures[-1]["printed_measure"]
        ),
    }


def load_bps_notes(path: Path) -> list[dict]:
    notes = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        for note_id, row in enumerate(reader):
            notes.append(
                {
                    "note_id": note_id,
                    "bps_time": float(row["start_meas"]),
                    "end_time": float(row["end_meas"]),
                    "midi": int(row["pitch"]),
                    "pitch_name": row["pitchName"],
                }
            )
    return notes


def attach_bps_note_ids(xml_notes: list[dict], bps_notes: list[dict]) -> None:
    by_time_pitch: dict[tuple[float, int], deque[dict]] = defaultdict(deque)
    for note in bps_notes:
        by_time_pitch[(round(note["bps_time"], 3), note["midi"])].append(
            note
        )

    xml_key_counts: dict[tuple[float, int], int] = defaultdict(int)
    for note in xml_notes:
        xml_key_counts[(round(note["bps_time"], 3), note["midi"])] += 1

    for note in sorted(
        xml_notes,
        key=lambda item: (
            item["bps_time"],
            item["staff"],
            item["midi"],
            item["x_norm"],
        ),
    ):
        key = (round(note["bps_time"], 3), note["midi"])
        candidates = by_time_pitch[key]
        ambiguous = len(candidates) > 1 or xml_key_counts[key] > 1
        # Keep a stable provisional one-to-one ID for repeat traversal and
        # geometry. Ambiguity is carried separately so strict output can
        # abstain without changing the candidate set or musical time.
        matched = candidates.popleft() if candidates else None
        note["note_id"] = matched["note_id"] if matched else None
        note["note_id_ambiguous"] = ambiguous
        note["end_bps_time"] = (
            matched.get("end_time", matched["bps_time"]) if matched else None
        )

    # BPSD merges notes connected by ties into one longer note row.  MusicXML
    # still contains the continuation note at the following measure position.
    # Reuse the BPSD row whose time span contains that continuation.
    for note in xml_notes:
        if note["note_id"] is not None:
            continue
        spanning = [
            bps_note
            for bps_note in bps_notes
            if bps_note["midi"] == note["midi"]
            and bps_note["bps_time"]
            <= note["bps_time"]
            <= bps_note.get("end_time", bps_note["bps_time"])
        ]
        if spanning:
            best = min(
                spanning,
                key=lambda bps_note: (
                    abs(bps_note["bps_time"] - note["bps_time"]),
                    bps_note["note_id"],
                ),
            )
            note["note_id"] = best["note_id"]
            note["end_bps_time"] = best.get("end_time", best["bps_time"])
            if len(spanning) > 1:
                note["note_id_ambiguous"] = True


def attach_repeat_occurrences(
    xml_notes: list[dict],
    repeat_mapping_rows: list[dict],
    bps_notes: list[dict],
) -> list[dict]:
    """Attach every unfolded BPSD occurrence to written-score XML notes.

    Geometry remains attached to the single printed note.  The returned list
    contains one copy per performance occurrence, while each original note is
    enriched with a lossless ``occurrences`` list and retains its first
    occurrence in the legacy scalar fields.
    """

    by_written_measure_index: dict[int, list[dict]] = defaultdict(list)
    for row in repeat_mapping_rows:
        if row.get("mapping_status") == "unresolved":
            continue
        try:
            written_measure_index = int(row["written_measure_index"])
        except (KeyError, TypeError, ValueError):
            continue
        by_written_measure_index[written_measure_index].append(row)

    expanded = []
    originals_by_sequence: dict[int, dict] = {}
    for note in xml_notes:
        written_measure = int(note["xml_measure"])
        written_measure_index = int(
            note.get("xml_measure_index", written_measure)
        )
        mappings = sorted(
            by_written_measure_index.get(written_measure_index, []),
            key=lambda row: int(row["unfolded_measure_index"]),
        )
        if not mappings:
            note["occurrences"] = []
            note["note_id"] = None
            continue
        timeline_offset = int(note.get("timeline_offset", 0))
        within_measure = float(
            note.get(
                "measure_within",
                note["bps_time"]
                - (written_measure_index - 1 + timeline_offset),
            )
        )
        sequence = int(note["xml_note_sequence"])
        originals_by_sequence[sequence] = note
        for mapping in mappings:
            occurrence = dict(note)
            occurrence["written_bps_time"] = note["bps_time"]
            occurrence["unfolded_measure_index"] = int(
                mapping["unfolded_measure_index"]
            )
            occurrence["repeat_occurrence"] = int(mapping["repeat_occurrence"])
            occurrence["repeat_occurrence_count"] = int(
                mapping["repeat_occurrence_count"]
            )
            occurrence["repeat_group_id"] = mapping.get("repeat_group_id", "")
            occurrence["repeat_mapping_status"] = mapping["mapping_status"]
            occurrence["written_measure"] = mapping.get(
                "written_measure", note.get("xml_measure", "")
            )
            occurrence["performance_measure"] = mapping.get(
                "performance_measure", mapping["unfolded_measure_index"]
            )
            occurrence["is_repeated_measure"] = str(
                mapping.get("is_repeated_measure", "")
            ).lower() in {"true", "1"}
            occurrence["repeat_status"] = mapping.get("repeat_status", "none")
            occurrence["volta_numbers"] = mapping.get("volta_numbers", "[]")
            occurrence["repeat_source"] = mapping.get(
                "repeat_source", "repetition_musicxml"
            )
            occurrence["bps_time"] = (
                occurrence["unfolded_measure_index"]
                - 1
                + timeline_offset
                + within_measure
            )
            expanded.append(occurrence)

    attach_bps_note_ids(expanded, bps_notes)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for occurrence in expanded:
        grouped[int(occurrence["xml_note_sequence"])].append(occurrence)
    for sequence, original in originals_by_sequence.items():
        occurrences = sorted(grouped[sequence], key=lambda item: item["bps_time"])
        original["occurrences"] = [
            {
                "repeat_occurrence": item["repeat_occurrence"],
                "repeat_occurrence_count": item["repeat_occurrence_count"],
                "repeat_group_id": item["repeat_group_id"],
                "unfolded_measure_index": item["unfolded_measure_index"],
                "bps_time": item["bps_time"],
                "end_bps_time": item.get("end_bps_time"),
                "note_id": item.get("note_id"),
                "mapping_status": item["repeat_mapping_status"],
                "written_measure": item.get("written_measure", ""),
                "performance_measure": item.get("performance_measure", ""),
                "is_repeated_measure": item.get("is_repeated_measure", False),
                "repeat_status": item.get("repeat_status", "none"),
                "volta_numbers": item.get("volta_numbers", "[]"),
                "repeat_source": item.get("repeat_source", ""),
            }
            for item in occurrences
        ]
        first = occurrences[0]
        original["written_bps_time"] = first["written_bps_time"]
        original["bps_time"] = first["bps_time"]
        original["note_id"] = first.get("note_id")
        original["end_bps_time"] = first.get("end_bps_time")
        original["repeat_occurrence_count"] = len(occurrences)
        original["repeat_group_id"] = first["repeat_group_id"]
        original["written_measure"] = first.get("written_measure", "")
        original["is_repeated_measure"] = first.get("is_repeated_measure", False)
        original["repeat_status"] = first.get("repeat_status", "none")
        original["volta_numbers"] = first.get("volta_numbers", "[]")
        original["repeat_source"] = first.get("repeat_source", "")

    return expanded


def attach_timeline_repeat_occurrences(
    events: list[dict], repeat_mapping_rows: list[dict]
) -> None:
    """Attach unfolded measure/time occurrences to non-note XML events."""

    by_written_measure_index: dict[int, list[dict]] = defaultdict(list)
    for row in repeat_mapping_rows:
        if row.get("mapping_status") == "unresolved":
            continue
        try:
            by_written_measure_index[int(row["written_measure_index"])].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    for event in events:
        written_measure = int(event["xml_measure"])
        written_measure_index = int(
            event.get("xml_measure_index", written_measure)
        )
        timeline_offset = int(event.get("timeline_offset", 0))
        within = float(
            event.get(
                "measure_within",
                event["bps_time"]
                - (written_measure_index - 1 + timeline_offset),
            )
        )
        mappings = sorted(
            by_written_measure_index.get(written_measure_index, []),
            key=lambda row: int(row["unfolded_measure_index"]),
        )
        occurrences = [
            {
                "repeat_occurrence": int(row["repeat_occurrence"]),
                "repeat_occurrence_count": int(row["repeat_occurrence_count"]),
                "repeat_group_id": row.get("repeat_group_id", ""),
                "unfolded_measure_index": int(row["unfolded_measure_index"]),
                "bps_time": (
                    int(row["unfolded_measure_index"])
                    - 1
                    + timeline_offset
                    + within
                ),
                "mapping_status": row["mapping_status"],
                "written_measure": row.get("written_measure", ""),
                "performance_measure": row.get(
                    "performance_measure", row["unfolded_measure_index"]
                ),
                "is_repeated_measure": str(
                    row.get("is_repeated_measure", "")
                ).lower() in {"true", "1"},
                "repeat_status": row.get("repeat_status", "none"),
                "volta_numbers": row.get("volta_numbers", "[]"),
                "repeat_source": row.get("repeat_source", ""),
            }
            for row in mappings
        ]
        event["repeat_occurrences"] = occurrences
        if occurrences:
            event["written_bps_time"] = event["bps_time"]
            event["bps_time"] = occurrences[0]["bps_time"]
            event["repeat_occurrence_count"] = len(occurrences)
            event["repeat_group_id"] = occurrences[0]["repeat_group_id"]
            event["written_measure"] = occurrences[0].get("written_measure", "")
            event["is_repeated_measure"] = occurrences[0].get(
                "is_repeated_measure", False
            )
            event["repeat_status"] = occurrences[0].get("repeat_status", "none")
            event["volta_numbers"] = occurrences[0].get("volta_numbers", "[]")
            event["repeat_source"] = occurrences[0].get("repeat_source", "")


def _note_match_status(note: dict, bps_by_id: dict[int, dict]) -> str:
    note_id = note.get("note_id")
    if note_id is None or note_id not in bps_by_id:
        return "unresolved"

    bps_note = bps_by_id[note_id]
    if bps_note["midi"] != note["midi"]:
        return "pitch_mismatch"

    xml_time = round(note["bps_time"], 3)
    start_time = round(bps_note["bps_time"], 3)
    end_time = round(bps_note["end_time"], 3)
    if xml_time == start_time:
        return "exact"
    if start_time <= xml_time <= end_time:
        return "within_tied_span"
    return "time_mismatch"


def build_slur_candidates(
    xml_notes: list[dict],
    bps_notes: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pair MusicXML slur endpoints without matching them to YOLO boxes."""

    # Slurs may legitimately cross from one staff to the other in piano
    # notation.  Staff is therefore recorded on each endpoint but is not part
    # of the pairing key.
    open_slurs: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    pairs = []
    issues = []

    for note in sorted(
        xml_notes,
        key=lambda item: item["xml_note_sequence"],
    ):
        for mark in note.get("slur_marks", []):
            key = (note["voice"], mark["number"])
            mark_type = mark["type"]
            if mark_type == "start":
                open_slurs[key].append((note, mark))
            elif mark_type == "stop":
                if not open_slurs[key]:
                    issues.append(
                        {
                            "issue": "stop_without_start",
                            "staff": note["staff"],
                            "voice": note["voice"],
                            "number": mark["number"],
                            "xml_measure": note["xml_measure"],
                            "pitch": note["pitch_name"],
                        }
                    )
                    continue
                start_note, start_mark = open_slurs[key].pop()
                pairs.append((start_note, note, start_mark))
            else:
                issues.append(
                    {
                        "issue": "unsupported_slur_type",
                        "type": mark_type,
                        "xml_measure": note["xml_measure"],
                        "pitch": note["pitch_name"],
                    }
                )

    for (voice, number), starts in sorted(open_slurs.items()):
        for note, _mark in starts:
            issues.append(
                {
                    "issue": "start_without_stop",
                    "staff": note["staff"],
                    "voice": voice,
                    "number": number,
                    "xml_measure": note["xml_measure"],
                    "pitch": note["pitch_name"],
                }
            )

    pairs.sort(
        key=lambda pair: (
            pair[0]["bps_time"],
            pair[0]["system"],
            pair[0]["staff"],
            pair[0]["xml_note_sequence"],
        )
    )
    bps_by_id = {note["note_id"]: note for note in bps_notes}
    candidates = []
    for index, (start_note, end_note, start_mark) in enumerate(pairs, start=1):
        start_match = _note_match_status(start_note, bps_by_id)
        end_match = _note_match_status(end_note, bps_by_id)
        note_id_ambiguous = bool(
            start_note.get("note_id_ambiguous")
            or end_note.get("note_id_ambiguous")
        )
        status = (
            "time_confirmed"
            if not note_id_ambiguous
            and start_match in {"exact", "within_tied_span"}
            and end_match in {"exact", "within_tied_span"}
            else "review"
        )
        candidates.append(
            {
                "candidate_id": f"S{index:02d}",
                "start_meas": f"{start_note['bps_time']:.3f}",
                "end_meas": f"{end_note['bps_time']:.3f}",
                "start_pitch": start_note["pitch_name"],
                "end_pitch": end_note["pitch_name"],
                "start_xml_measure": start_note["xml_measure"],
                "end_xml_measure": end_note["xml_measure"],
                "start_system": start_note["system"],
                "end_system": end_note["system"],
                "start_staff": start_note["staff"],
                "end_staff": end_note["staff"],
                "start_voice": start_note["voice"],
                "end_voice": end_note["voice"],
                "start_note_candidate": (
                    ""
                    if start_note.get("note_id") is None
                    else start_note["note_id"]
                ),
                "end_note_candidate": (
                    ""
                    if end_note.get("note_id") is None
                    else end_note["note_id"]
                ),
                "start_note_match": start_match,
                "end_note_match": end_match,
                "note_id_ambiguous": note_id_ambiguous,
                "orientation": start_mark.get("orientation", ""),
                "status": status,
            }
        )

    return candidates, issues


def write_slur_candidates_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SLUR_CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_tie_candidates(
    xml_notes: list[dict],
    bps_notes: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pair tie endpoints by staff and pitch, preferring the same voice."""

    open_ties: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pairs = []
    issues = []
    for note in sorted(xml_notes, key=lambda item: item["xml_note_sequence"]):
        key = (note["staff"], note["midi"])
        for mark in note.get("tie_marks", []):
            mark_type = mark["type"]
            if mark_type == "start":
                open_ties[key].append(note)
            elif mark_type == "stop":
                if not open_ties[key]:
                    issues.append(
                        {
                            "issue": "tie_stop_without_start",
                            "staff": note["staff"],
                            "voice": note["voice"],
                            "xml_measure": note["xml_measure"],
                            "pitch": note["pitch_name"],
                        }
                    )
                    continue
                same_voice = [
                    index
                    for index, start in enumerate(open_ties[key])
                    if start["voice"] == note["voice"]
                ]
                if same_voice:
                    start_index = same_voice[-1]
                elif len(open_ties[key]) == 1:
                    start_index = 0
                else:
                    issues.append(
                        {
                            "issue": "ambiguous_cross_voice_tie_stop",
                            "staff": note["staff"],
                            "voice": note["voice"],
                            "xml_measure": note["xml_measure"],
                            "pitch": note["pitch_name"],
                        }
                    )
                    continue
                pairs.append((open_ties[key].pop(start_index), note))
            else:
                issues.append(
                    {
                        "issue": "unsupported_tie_type",
                        "type": mark_type,
                        "xml_measure": note["xml_measure"],
                        "pitch": note["pitch_name"],
                    }
                )

    for (staff, _midi), starts in sorted(open_ties.items()):
        for note in starts:
            issues.append(
                {
                    "issue": "tie_start_without_stop",
                    "staff": staff,
                    "voice": note["voice"],
                    "xml_measure": note["xml_measure"],
                    "pitch": note["pitch_name"],
                }
            )

    pairs.sort(key=lambda pair: pair[0]["xml_note_sequence"])
    bps_by_id = {note["note_id"]: note for note in bps_notes}
    candidates = []
    for index, (start_note, end_note) in enumerate(pairs, start=1):
        start_match = _note_match_status(start_note, bps_by_id)
        end_match = _note_match_status(end_note, bps_by_id)
        note_id_ambiguous = bool(
            start_note.get("note_id_ambiguous")
            or end_note.get("note_id_ambiguous")
        )
        status = (
            "time_confirmed"
            if not note_id_ambiguous
            and start_match in {"exact", "within_tied_span"}
            and end_match in {"exact", "within_tied_span"}
            else "review"
        )
        candidates.append(
            {
                "candidate_id": f"T{index:02d}",
                "start_meas": f"{start_note['bps_time']:.3f}",
                "end_meas": f"{end_note['bps_time']:.3f}",
                "pitch": start_note["pitch_name"],
                "start_xml_measure": start_note["xml_measure"],
                "end_xml_measure": end_note["xml_measure"],
                "system": start_note["system"],
                "staff": start_note["staff"],
                "voice": start_note["voice"],
                "start_note_candidate": start_note.get("note_id", ""),
                "end_note_candidate": end_note.get("note_id", ""),
                "start_note_match": start_match,
                "end_note_match": end_match,
                "note_id_ambiguous": note_id_ambiguous,
                "status": status,
            }
        )
    return candidates, issues


def write_tie_candidates_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TIE_CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _clef_middle_diatonic(clef: dict) -> int:
    sign = clef.get("sign", "G")
    line = int(clef.get("line", 2))
    reference = {
        "G": 4 * 7 + 4,
        "F": 3 * 7 + 3,
        "C": 4 * 7,
    }.get(sign, 4 * 7 + 4)
    return reference + (3 - line) * 2


def note_pixel_position(
    note: dict,
    system: SystemGeometry,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> tuple[float, float]:
    measure_map = (
        measure_x_maps.get(
            (note["system"], note.get("system_measure_index", -1))
        )
        if measure_x_maps
        else None
    )
    if measure_map is not None and "measure_x_norm" in note:
        within = min(1.0, max(0.0, float(note["measure_x_norm"])))
        x = measure_map["left_x"] + within * (
            measure_map["right_x"] - measure_map["left_x"]
        )
    else:
        x = system.x_left + note["x_norm"] * (
            system.x_right - system.x_left
        )
    staff_geometry = system.upper if note["staff"] == 1 else system.lower
    middle_diatonic = _clef_middle_diatonic(note["clef"])
    y = staff_geometry.center - (
        note["diatonic"] - middle_diatonic
    ) * staff_geometry.line_spacing / 2
    return x, y


def attach_review_note_candidates(
    rows: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
    note_x_overrides: dict[int, float] | None = None,
    *,
    limit: int = 64,
) -> None:
    """Attach human-readable XML note choices to every YOLO row.

    Span symbols receive the complete page note index so either endpoint can be
    selected across systems. Point symbols retain a bounded local index.
    """

    systems_by_number = {system.number: system for system in systems}
    notes_by_system: dict[int, list[dict]] = defaultdict(list)
    for note in xml_notes:
        if note.get("system") in systems_by_number:
            notes_by_system[note["system"]].append(note)
    note_x_overrides = note_x_overrides or {}

    for row in rows:
        try:
            system_number = int(row.get("system", ""))
            box_x = float(row.get("x", 0.5)) * image_width
            box_y = float(row.get("y", 0.5)) * image_height
        except (TypeError, ValueError):
            row["review_note_candidates_json"] = "[]"
            continue
        system = systems_by_number.get(system_number)
        if system is None:
            row["review_note_candidates_json"] = "[]"
            continue

        candidates = []
        seen = set()
        candidate_notes = (
            xml_notes
            if str(row.get("target_type", "")) == "span"
            else notes_by_system.get(system_number, [])
        )
        for note in candidate_notes:
            note_system = systems_by_number.get(note.get("system"))
            if note_system is None:
                continue
            note_id = note.get("note_id")
            time = note.get("bps_time")
            key = (
                note.get("xml_note_sequence"),
                str(note_id),
                round(float(time), 6) if time is not None else None,
                note.get("staff"),
                note.get("pitch_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            note_x, note_y = note_pixel_position(
                note, note_system, measure_x_maps=measure_x_maps
            )
            sequence = note.get("xml_note_sequence")
            if sequence in note_x_overrides:
                note_x = note_x_overrides[sequence]
            dx = abs(box_x - note_x)
            dy = abs(box_y - note_y)
            candidates.append(
                {
                    "note_id": note_id,
                    "start_meas": f"{float(time):.3f}" if time is not None else "",
                    "end_meas": f"{float(time):.3f}" if time is not None else "",
                    "connected_note": json.dumps(
                        [note_id] if note_id is not None else []
                    ),
                    "pitch": note.get("pitch_name", ""),
                    "xml_measure": note.get("xml_measure", ""),
                    "printed_measure": note.get(
                        "printed_measure", note.get("xml_measure", "")
                    ),
                    "staff": note.get("staff", ""),
                    "xml_note_sequence": note.get("xml_note_sequence", ""),
                    "x_px": round(note_x, 1),
                    "y_px": round(note_y, 1),
                    "distance_px": round(dx + 0.75 * dy, 1),
                }
            )
        measure_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for candidate in candidates:
            measure_groups[
                (str(candidate["printed_measure"]), str(candidate["staff"]))
            ].append(candidate)
        for measure_candidates in measure_groups.values():
            measure_candidates.sort(
                key=lambda candidate: (
                    candidate["x_px"],
                    candidate["y_px"],
                    str(candidate["note_id"]),
                )
            )
            for order, candidate in enumerate(measure_candidates, start=1):
                candidate["measure_note_order"] = order

        candidates.sort(
            key=lambda candidate: (
                candidate["distance_px"],
                candidate["x_px"],
                candidate["y_px"],
            )
        )
        selected_candidates = (
            candidates
            if str(row.get("target_type", "")) == "span"
            else candidates[:limit]
        )
        row["review_note_candidates_json"] = json.dumps(
            selected_candidates,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def build_measure_local_x_maps(
    image: Image.Image,
    systems: list[SystemGeometry],
    xml_page: dict,
) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    """Align XML measure boundaries to the scan for piecewise note x mapping."""

    measures_by_system: dict[int, list[dict]] = defaultdict(list)
    for measure in xml_page.get("measures", []):
        measures_by_system[measure["system"]].append(measure)
    ordered_systems = sorted(systems, key=lambda item: item.number)
    if any(system.number not in measures_by_system for system in ordered_systems):
        return {}, [{"status": "fallback_system_count_mismatch"}]

    reference_systems = []
    reference_boundaries = []
    for system in ordered_systems:
        measures = measures_by_system[system.number]
        total_width = sum(measure["width"] for measure in measures)
        if total_width <= 0:
            return {}, [
                {
                    "system": system.number,
                    "status": "fallback_missing_measure_width",
                }
            ]
        cumulative = [0.0]
        for measure in measures:
            cumulative.append(cumulative[-1] + measure["width"])
        reference_systems.append(
            SystemGeometry(
                number=system.number,
                upper=system.upper,
                lower=system.lower,
                x_left=0.0,
                x_right=total_width,
            )
        )
        reference_boundaries.append(cumulative)

    try:
        aligned = align_barlines_from_reference(
            image,
            ordered_systems,
            reference_systems,
            reference_boundaries,
        )
    except (ValueError, IndexError) as error:
        return {}, [
            {
                "status": "fallback_barline_alignment_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        ]

    maps: dict[tuple[int, int], dict] = {}
    diagnostics = []
    for system, boundaries in zip(ordered_systems, aligned):
        for measure_index, (left, right) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            if right["x"] <= left["x"]:
                diagnostics.append(
                    {
                        "system": system.number,
                        "measure_index": measure_index,
                        "status": "fallback_nonincreasing_boundaries",
                    }
                )
                continue
            maps[(system.number, measure_index)] = {
                "left_x": float(left["x"]),
                "right_x": float(right["x"]),
                "left_status": left["status"],
                "right_status": right["status"],
            }
            diagnostics.append(
                {
                    "system": system.number,
                    "measure_index": measure_index,
                    "status": "mapped",
                    "left_x": left["x"],
                    "right_x": right["x"],
                    "left_boundary_status": left["status"],
                    "right_boundary_status": right["status"],
                }
            )
    return maps, diagnostics


def build_clean_reference_geometry(
    image: Image.Image,
    systems: list[SystemGeometry],
    clean_image: Image.Image,
    xml_page: dict,
) -> tuple[dict[tuple[int, int], dict], dict[int, float], list[dict]]:
    """Use a clean repetition-layout page as scan geometry reference.

    The clean page supplies system and barline positions.  Noteheads are first
    snapped on that page, transferred measure-locally to the scan, then snapped
    once more on the scan.  Any structural mismatch returns an empty result so
    callers can safely fall back to MusicXML-width geometry.
    """

    measures_by_system: dict[int, list[dict]] = defaultdict(list)
    for measure in xml_page.get("measures", []):
        measures_by_system[measure["system"]].append(measure)
    ordered_systems = sorted(systems, key=lambda item: item.number)
    try:
        clean_systems = sorted(
            detect_systems(clean_image), key=lambda item: item.number
        )
    except ValueError as error:
        return {}, {}, [
            {
                "source": "clean_repetition_pdf",
                "status": "fallback_clean_system_detection_failed",
                "error": str(error),
            }
        ]
    if len(clean_systems) != len(ordered_systems):
        return {}, {}, [
            {
                "source": "clean_repetition_pdf",
                "status": "fallback_clean_system_count_mismatch",
                "scan_systems": len(ordered_systems),
                "clean_systems": len(clean_systems),
            }
        ]
    if any(system.number not in measures_by_system for system in ordered_systems):
        return {}, {}, [
            {
                "source": "clean_repetition_pdf",
                "status": "fallback_xml_system_count_mismatch",
            }
        ]

    expected_counts = [
        len(measures_by_system[system.number]) + 1 for system in ordered_systems
    ]
    try:
        clean_boundaries = detect_barlines(
            clean_image, clean_systems, expected_counts
        )
        aligned_boundaries = align_barlines_from_reference(
            image, ordered_systems, clean_systems, clean_boundaries
        )
    except (ValueError, IndexError) as error:
        return {}, {}, [
            {
                "source": "clean_repetition_pdf",
                "status": "fallback_clean_barline_alignment_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        ]

    scan_maps: dict[tuple[int, int], dict] = {}
    clean_maps: dict[tuple[int, int], dict] = {}
    diagnostics: list[dict] = []
    for scan_system, clean_system, scan_bounds, clean_bounds in zip(
        ordered_systems, clean_systems, aligned_boundaries, clean_boundaries
    ):
        for measure_index in range(len(clean_bounds) - 1):
            scan_left = scan_bounds[measure_index]
            scan_right = scan_bounds[measure_index + 1]
            clean_left = clean_bounds[measure_index]
            clean_right = clean_bounds[measure_index + 1]
            if scan_right["x"] <= scan_left["x"] or clean_right <= clean_left:
                return {}, {}, [
                    {
                        "source": "clean_repetition_pdf",
                        "system": scan_system.number,
                        "measure_index": measure_index,
                        "status": "fallback_nonincreasing_boundaries",
                    }
                ]
            key = (scan_system.number, measure_index)
            scan_maps[key] = {
                "left_x": float(scan_left["x"]),
                "right_x": float(scan_right["x"]),
                "left_status": scan_left["status"],
                "right_status": scan_right["status"],
            }
            clean_maps[key] = {
                "left_x": float(clean_left),
                "right_x": float(clean_right),
            }
            diagnostics.append(
                {
                    "source": "clean_repetition_pdf",
                    "system": scan_system.number,
                    "measure_index": measure_index,
                    "status": "mapped",
                    "left_x": scan_left["x"],
                    "right_x": scan_right["x"],
                    "left_boundary_status": scan_left["status"],
                    "right_boundary_status": scan_right["status"],
                }
            )

    scan_systems_by_number = {system.number: system for system in ordered_systems}
    clean_systems_by_number = {system.number: system for system in clean_systems}
    note_x_overrides: dict[int, float] = {}
    for note in xml_page.get("notes", []):
        sequence = note.get("xml_note_sequence")
        key = (note.get("system"), note.get("system_measure_index"))
        if sequence is None or key not in clean_maps or key not in scan_maps:
            continue
        clean_system = clean_systems_by_number.get(note["system"])
        scan_system = scan_systems_by_number.get(note["system"])
        if clean_system is None or scan_system is None:
            continue
        clean_x, clean_y = note_pixel_position(note, clean_system, clean_maps)
        clean_staff = clean_system.upper if note.get("staff", 1) == 1 else clean_system.lower
        clean_snap = snap_notehead_x(
            clean_image,
            clean_x,
            clean_y,
            clean_staff,
            search_radius=max(12, round(clean_staff.line_spacing * 2.4)),
        )
        if clean_snap["ink_count"] < max(6, round(clean_staff.line_spacing * 0.6)):
            continue
        clean_map = clean_maps[key]
        within = (clean_snap["x"] - clean_map["left_x"]) / max(
            clean_map["right_x"] - clean_map["left_x"], 1
        )
        if not -0.08 <= within <= 1.08:
            continue
        scan_map = scan_maps[key]
        scan_x = scan_map["left_x"] + min(1.0, max(0.0, within)) * (
            scan_map["right_x"] - scan_map["left_x"]
        )
        _rough_x, scan_y = note_pixel_position(note, scan_system, scan_maps)
        scan_staff = scan_system.upper if note.get("staff", 1) == 1 else scan_system.lower
        scan_snap = snap_notehead_x(
            image,
            scan_x,
            scan_y,
            scan_staff,
            search_radius=max(12, round(scan_staff.line_spacing * 2.4)),
        )
        if scan_snap["ink_count"] >= max(6, round(scan_staff.line_spacing * 0.6)):
            note_x_overrides[int(sequence)] = float(scan_snap["x"])

    diagnostics.append(
        {
            "source": "clean_repetition_pdf",
            "status": "noteheads_snapped",
            "note_x_overrides": len(note_x_overrides),
        }
    )
    return scan_maps, note_x_overrides, diagnostics


def snap_notehead_x(
    image: Image.Image,
    predicted_x: float,
    predicted_y: float,
    staff: StaffGeometry,
    search_radius: int,
) -> dict:
    """Snap a rough x position to dense notehead ink at a known pitch y."""

    gray = np.asarray(image.convert("L"))
    dark = gray < 170
    without_staff = dark.copy()
    for line in staff.lines:
        y = round(line)
        without_staff[
            max(0, y - 1) : min(without_staff.shape[0], y + 2),
            :,
        ] = False

    ellipse_radius_x = max(4, round(staff.line_spacing * 0.72))
    ellipse_radius_y = max(3, round(staff.line_spacing * 0.48))
    center_y = round(predicted_y)
    scored = []
    for center_x in range(
        round(predicted_x - search_radius),
        round(predicted_x + search_radius) + 1,
    ):
        y0 = max(0, center_y - ellipse_radius_y)
        y1 = min(without_staff.shape[0], center_y + ellipse_radius_y + 1)
        x0 = max(0, center_x - ellipse_radius_x)
        x1 = min(without_staff.shape[1], center_x + ellipse_radius_x + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        ellipse = (
            ((xx - center_x) / ellipse_radius_x) ** 2
            + ((yy - center_y) / ellipse_radius_y) ** 2
            <= 1
        )
        ink_count = int((without_staff[y0:y1, x0:x1] & ellipse).sum())
        score = ink_count - 0.08 * abs(center_x - predicted_x)
        scored.append(
            (
                score,
                ink_count,
                -abs(center_x - predicted_x),
                center_x,
            )
        )

    score, ink_count, _distance, center_x = max(scored)
    return {
        "x": center_x,
        "y": predicted_y,
        "ink_count": ink_count,
        "score": score,
    }


def _base_output_row(box: dict, system: int) -> dict:
    musical_time = musical_time_for_class(box.get("class"))
    return {
        "class_id": box["class_id"],
        "x": f"{box['x']:.6f}",
        "y": f"{box['y']:.6f}",
        "w": f"{box['w']:.6f}",
        "h": f"{box['h']:.6f}",
        "class": box["class"],
        "musical_time": "" if musical_time is None else musical_time,
        "start_meas": "NA",
        "end_meas": "NA",
        "start_note": "NA",
        "end_note": "NA",
        "connected_note": "NA",
        "stem_dir": "NA",
        "txt_line": box["txt_line"],
        "system": system,
        "xml_measure": "NA",
        "xml_symbol": "NA",
        "xml_staff": "NA",
        "target_type": "",
        "note_ids": "",
        "pitches": "",
        "repeat_occurrences_json": "",
        "repeat_occurrence_count": "",
        "repeat_group_id": "",
        "repeat_mapping_status": "",
        "match_source": "NA",
        "confidence": "0.000",
        "match_score": "0.000",
        "confidence_calibrated": "false",
        "geometry_score": "",
        "candidate_margin": "",
        "count_agreement": "",
        "xml_time_confirmed": "",
        "status": "unmatched",
        "target_x_px": "NA",
        "target_y_px": "NA",
        "end_target_x_px": "NA",
        "end_target_y_px": "NA",
    }


def _repeat_mapping_status(occurrences: list[dict]) -> str:
    statuses = {
        str(item.get("mapping_status", "")).strip()
        for item in occurrences
        if str(item.get("mapping_status", "")).strip()
    }
    return "+".join(sorted(statuses))


def finalize_match_diagnostics(rows: list[dict]) -> None:
    """Normalize heuristic score metadata and enforce repeat safety gates."""

    for row in rows:
        row["match_score"] = row.get("match_score") or row.get("confidence", "")
        row["confidence_calibrated"] = "false"
        if not row.get("repeat_mapping_status"):
            try:
                occurrences = json.loads(row.get("repeat_occurrences_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                occurrences = []
            if isinstance(occurrences, list):
                row["repeat_mapping_status"] = _repeat_mapping_status(
                    [item for item in occurrences if isinstance(item, dict)]
                )
        if not repeat_mapping_is_safe(row.get("repeat_mapping_status")) and row.get(
            "status"
        ) in {"matched", "inferred"}:
            row["status"] = "review"
            row["xml_time_confirmed"] = "false"


def match_dynamics(
    boxes: list[dict],
    xml_dynamics: list[dict],
    systems: list[SystemGeometry],
    image_height: int,
) -> tuple[list[dict], list[dict]]:
    output = []
    unused_xml = []
    by_system_class_boxes: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_system_class_xml: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for box in boxes:
        if box["class"] not in DYNAMIC_CLASS_NAMES:
            continue
        system = assign_system(box, systems, image_height)
        by_system_class_boxes[(system, box["class"])].append(box)

    for event in xml_dynamics:
        if event["class"] in DYNAMIC_CLASS_NAMES:
            by_system_class_xml[(event["system"], event["class"])].append(event)

    all_keys = sorted(set(by_system_class_boxes) | set(by_system_class_xml))
    for key in all_keys:
        yolo_items = sorted(by_system_class_boxes[key], key=lambda item: item["x"])
        xml_items = sorted(
            by_system_class_xml[key],
            key=lambda item: (
                item["x_norm"],
                item["bps_time"],
                item.get("component_index", 0),
            ),
        )
        pair_count = min(len(yolo_items), len(xml_items))

        count_agreement = (
            1.0
            if len(yolo_items) == len(xml_items)
            else pair_count / max(len(yolo_items), len(xml_items), 1)
        )

        for box, event in zip(yolo_items[:pair_count], xml_items[:pair_count]):
            x_error = abs(float(box["x"]) - float(event["x_norm"]))
            geometry_score = math.exp(-0.5 * (x_error / 0.08) ** 2)
            alternative_scores = sorted(
                (
                    math.exp(
                        -0.5
                        * (
                            abs(float(box["x"]) - float(candidate["x_norm"]))
                            / 0.08
                        )
                        ** 2
                    )
                    for candidate in xml_items
                ),
                reverse=True,
            )
            second_score = alternative_scores[1] if len(alternative_scores) > 1 else 0.0
            margin = max(0.0, geometry_score - second_score)
            box_best = min(
                xml_items,
                key=lambda candidate: abs(
                    float(box["x"]) - float(candidate["x_norm"])
                ),
            ) is event
            mutual_best = min(
                yolo_items,
                key=lambda candidate: abs(
                    float(candidate["x"]) - float(event["x_norm"])
                ),
            ) is box
            match_score = 0.75 * geometry_score + 0.25 * count_agreement
            occurrences = event.get("repeat_occurrences", [])
            repeat_mapping_status = _repeat_mapping_status(occurrences)
            repeat_safe = repeat_mapping_is_safe(repeat_mapping_status)
            row = _base_output_row(box, key[0])
            row.update(
                {
                    "start_meas": f"{event['bps_time']:.3f}",
                    "end_meas": f"{event['bps_time']:.3f}",
                    "xml_measure": event["xml_measure"],
                    "xml_symbol": event["xml_symbol"],
                    "xml_staff": event["staff"],
                    "target_type": "measure_position",
                    "repeat_occurrences_json": json.dumps(
                        occurrences,
                        ensure_ascii=False,
                    ),
                    "repeat_occurrence_count": event.get(
                        "repeat_occurrence_count", 1
                    ),
                    "repeat_group_id": event.get("repeat_group_id", ""),
                    "repeat_mapping_status": repeat_mapping_status,
                    "match_source": "musicxml_dynamic",
                    "confidence": f"{match_score:.3f}",
                    "match_score": f"{match_score:.3f}",
                    "confidence_calibrated": "false",
                    "geometry_score": f"{geometry_score:.3f}",
                    "candidate_margin": f"{margin:.3f}",
                    "count_agreement": f"{count_agreement:.3f}",
                    "xml_time_confirmed": "true" if repeat_safe else "false",
                    "status": (
                        "matched"
                        if match_score
                        >= auto_accept_threshold(box["class"], 0.85)
                        and margin >= 0.10
                        and box_best
                        and mutual_best
                        and repeat_safe
                        else "review"
                    ),
                }
            )
            output.append(row)

        for box in yolo_items[pair_count:]:
            output.append(_base_output_row(box, key[0]))
        unused_xml.extend(xml_items[pair_count:])

    return output, unused_xml


def match_fingerings(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    output = []
    systems_by_number = {system.number: system for system in systems}
    notes_by_system: dict[int, list[dict]] = defaultdict(list)
    for note in xml_notes:
        if note.get("note_id") is not None:
            notes_by_system[note["system"]].append(note)

    fingering_boxes = []
    for box in boxes:
        if box["class"] not in FINGERING_CLASS_NAMES:
            continue
        prepared = dict(box)
        prepared["_system"] = assign_system(box, systems, image_height)
        prepared["_x_px"] = box["x"] * image_width
        prepared["_y_px"] = box["y"] * image_height
        fingering_boxes.append(prepared)

    # Connected components group vertically stacked fingering digits that have
    # almost the same x coordinate.  The group is then assigned jointly to
    # distinct notes in one chord/onset.
    x_tolerance = image_width * 0.007
    y_tolerance = max(image_height * 0.045, 24)
    remaining = set(range(len(fingering_boxes)))
    box_groups = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            current_box = fingering_boxes[current]
            neighbors = []
            for other in remaining:
                other_box = fingering_boxes[other]
                if (
                    other_box["_system"] == current_box["_system"]
                    and abs(other_box["_x_px"] - current_box["_x_px"])
                    <= x_tolerance
                    and abs(other_box["_y_px"] - current_box["_y_px"])
                    <= y_tolerance
                ):
                    neighbors.append(other)
            for neighbor in neighbors:
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        box_groups.append(
            [fingering_boxes[index] for index in sorted(component)]
        )

    # Process groups deterministically in reading order.  Once a BPS note is
    # used by a fingering group it cannot be claimed by a later independent
    # group.  Stacked digits in the same group still claim distinct notes in
    # one chord jointly.
    box_groups.sort(
        key=lambda group: (
            group[0]["_system"],
            sum(box["_x_px"] for box in group) / len(group),
            sum(box["_y_px"] for box in group) / len(group),
        )
    )
    used_note_ids: set[int] = set()

    for box_group in box_groups:
        system_number = box_group[0]["_system"]
        system = systems_by_number[system_number]
        chord_groups: dict[tuple, list[tuple[dict, float, float]]] = defaultdict(list)
        for note in notes_by_system[system_number]:
            note_x, note_y = note_pixel_position(
                note, system, measure_x_maps=measure_x_maps
            )
            chord_key = (
                round(note["bps_time"], 6),
                note["staff"],
                round(note["x_norm"], 4),
            )
            chord_groups[chord_key].append((note, note_x, note_y))

        group_candidates = []
        boxes_by_y = sorted(box_group, key=lambda item: item["_y_px"])
        group_size = len(boxes_by_y)
        mean_box_x = sum(box["_x_px"] for box in boxes_by_y) / group_size
        mean_box_y = sum(box["_y_px"] for box in boxes_by_y) / group_size
        inter_staff_boundary = (
            max(system.upper.lines) + min(system.lower.lines)
        ) / 2
        preferred_staff = 1 if mean_box_y <= inter_staff_boundary else 2

        for chord_notes in chord_groups.values():
            if len(chord_notes) < group_size:
                continue
            chord_x = sum(item[1] for item in chord_notes) / len(chord_notes)
            dx = abs(mean_box_x - chord_x)
            if dx > image_width * 0.12:
                continue

            for note_subset in combinations(chord_notes, group_size):
                subset_note_ids = {
                    note_item[0]["note_id"] for note_item in note_subset
                }
                if subset_note_ids & used_note_ids:
                    continue
                notes_by_y = sorted(note_subset, key=lambda item: item[2])
                pairs = list(zip(boxes_by_y, notes_by_y))
                mean_dy = sum(
                    abs(box["_y_px"] - note_item[2])
                    for box, note_item in pairs
                ) / group_size
                candidate_staff = notes_by_y[0][0]["staff"]
                if FINGERING_STRICT_STAFF and candidate_staff != preferred_staff:
                    continue
                staff_penalty = (
                    image_width * FINGERING_WRONG_STAFF_PENALTY_RATIO
                    if candidate_staff != preferred_staff
                    else 0.0
                )
                score = dx + FINGERING_DY_WEIGHT * mean_dy + staff_penalty
                group_candidates.append(
                    (
                        score,
                        dx,
                        mean_dy,
                        staff_penalty,
                        len(chord_notes),
                        pairs,
                    )
                )

        group_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[5][0][1][0]["bps_time"],
            )
        )

        if not group_candidates:
            for box in box_group:
                output.append(_base_output_row(box, system_number))
            continue
        best = group_candidates[0]
        second_score = (
            group_candidates[1][0]
            if len(group_candidates) > 1
            else best[0] + 100
        )
        _score, dx, _mean_dy, _staff_penalty, chord_note_count, pairs = best
        ambiguity_margin = max(0.0, second_score - best[0])
        x_quality = math.exp(-dx / max(image_width * 0.025, 1))
        ambiguity_quality = min(1.0, ambiguity_margin / 25)
        confidence = 0.45 + 0.35 * x_quality + 0.20 * ambiguity_quality
        confidence = min(0.99, max(0.0, confidence))
        ambiguous_chord_member = group_size == 1 and chord_note_count > 1
        staff_side_mismatch = _staff_penalty > 0
        if ambiguous_chord_member or staff_side_mismatch:
            # A single printed digit next to a multi-note chord does not
            # contain enough geometry to identify the intended chord member
            # reliably. Likewise, a cross-staff candidate must not become an
            # automatic acceptance merely because its x position is close.
            confidence = min(confidence, 0.69)
        status = (
            "inferred"
            if confidence >= auto_accept_threshold(
                box["class"], FINGERING_AUTO_ACCEPT_THRESHOLD
            )
            else "review"
        )

        used_note_ids.update(
            note_item[0]["note_id"] for _box, note_item in pairs
        )

        for box, note_item in pairs:
            note, note_x, note_y = note_item
            note_id = note.get("note_id")
            ambiguous_note_id = bool(note.get("note_id_ambiguous"))
            occurrence_note_ids = [
                occurrence["note_id"]
                for occurrence in note.get("occurrences", [])
                if occurrence.get("note_id") is not None
            ]
            if not occurrence_note_ids and note_id is not None:
                occurrence_note_ids = [note_id]
            row = _base_output_row(box, system_number)
            row.update(
                {
                    "start_meas": f"{note['bps_time']:.3f}",
                    "end_meas": f"{note['bps_time']:.3f}",
                    "start_note": note_id if note_id is not None else "NA",
                    "end_note": note_id if note_id is not None else "NA",
                    "connected_note": (
                        f"[{note_id}]" if note_id is not None else "NA"
                    ),
                    "xml_measure": note["xml_measure"],
                    "xml_symbol": note["pitch_name"],
                    "xml_staff": note["staff"],
                    "target_type": "note",
                    "note_ids": json.dumps(occurrence_note_ids),
                    "pitches": json.dumps([note["pitch_name"]]),
                    "repeat_occurrences_json": json.dumps(
                        note.get("occurrences", []), ensure_ascii=False
                    ),
                    "repeat_occurrence_count": note.get(
                        "repeat_occurrence_count", 1
                    ),
                    "repeat_group_id": note.get("repeat_group_id", ""),
                    "repeat_mapping_status": _repeat_mapping_status(
                        note.get("occurrences", [])
                    ),
                    "match_source": (
                        "grouped_nearest_musicxml_note_ambiguous_bps_unison"
                        if ambiguous_note_id
                        else "grouped_nearest_musicxml_bps_note_ambiguous_chord"
                        if ambiguous_chord_member
                        else "grouped_nearest_musicxml_bps_note_staff_mismatch"
                        if staff_side_mismatch
                        else "grouped_nearest_musicxml_bps_note"
                    ),
                    "confidence": (
                        f"{min(confidence, 0.69) if ambiguous_note_id else confidence:.3f}"
                    ),
                    "match_score": (
                        f"{min(confidence, 0.69) if ambiguous_note_id else confidence:.3f}"
                    ),
                    "confidence_calibrated": "false",
                    "geometry_score": f"{x_quality:.3f}",
                    "candidate_margin": f"{ambiguity_quality:.3f}",
                    "count_agreement": "1.000",
                    "xml_time_confirmed": "true",
                    "status": "review" if ambiguous_note_id else status,
                    "target_x_px": f"{note_x:.1f}",
                    "target_y_px": f"{note_y:.1f}",
                }
            )
            output.append(row)

    return output


def _event_geometry(
    event: dict,
    systems_by_number: dict[int, SystemGeometry],
    measure_x_maps: dict[tuple[int, int], dict] | None,
) -> tuple[float, float]:
    system = systems_by_number[event["system"]]
    if "diatonic" in event:
        return note_pixel_position(event, system, measure_x_maps)
    measure_map = (
        measure_x_maps.get(
            (event["system"], event.get("system_measure_index", -1))
        )
        if measure_x_maps
        else None
    )
    if measure_map is not None and "measure_x_norm" in event:
        within = min(1.0, max(0.0, float(event["measure_x_norm"])))
        x = measure_map["left_x"] + within * (
            measure_map["right_x"] - measure_map["left_x"]
        )
    else:
        x = system.x_left + event["x_norm"] * (system.x_right - system.x_left)
    staff = system.upper if event.get("staff", 1) == 1 else system.lower
    return x, staff.center


def _anchor(
    event: dict,
    systems_by_number: dict[int, SystemGeometry],
    measure_x_maps: dict[tuple[int, int], dict] | None,
    *,
    members: list[dict] | None = None,
    target_type: str | None = None,
) -> dict:
    x, y = _event_geometry(event, systems_by_number, measure_x_maps)
    return {
        "event": event,
        "members": members if members is not None else ([event] if "midi" in event else []),
        "x": x,
        "y": y,
        "target_type": target_type or ("note" if "midi" in event else "rest"),
    }


def _anchor_occurrences(anchor: dict) -> list[dict]:
    event = anchor["event"]
    return list(event.get("occurrences") or event.get("repeat_occurrences") or [])


def _span_occurrences(start: dict, end: dict) -> list[dict]:
    starts = _anchor_occurrences(start)
    ends = _anchor_occurrences(end)
    if not starts and not ends:
        return []
    count = max(len(starts), len(ends))
    output = []
    for index in range(count):
        start_item = starts[min(index, len(starts) - 1)] if starts else {}
        end_item = ends[min(index, len(ends) - 1)] if ends else {}
        output.append(
            {
                "repeat_occurrence": index + 1,
                "repeat_occurrence_count": count,
                "repeat_group_id": start_item.get(
                    "repeat_group_id", end_item.get("repeat_group_id", "")
                ),
                "start_meas": start_item.get(
                    "bps_time", start["event"]["bps_time"]
                ),
                "end_meas": end_item.get("bps_time", end["event"]["bps_time"]),
                "start_note": start_item.get("note_id"),
                "end_note": end_item.get("note_id"),
                "mapping_status": start_item.get(
                    "mapping_status", end_item.get("mapping_status", "")
                ),
                "written_measure": start_item.get(
                    "written_measure",
                    start["event"].get("xml_measure", ""),
                ),
                "performance_measure": start_item.get(
                    "performance_measure",
                    start_item.get("unfolded_measure_index", ""),
                ),
                "is_repeated_measure": start_item.get(
                    "is_repeated_measure",
                    end_item.get("is_repeated_measure", False),
                ),
                "repeat_status": start_item.get(
                    "repeat_status", end_item.get("repeat_status", "none")
                ),
                "volta_numbers": start_item.get(
                    "volta_numbers", end_item.get("volta_numbers", "[]")
                ),
                "repeat_source": start_item.get(
                    "repeat_source", end_item.get("repeat_source", "")
                ),
            }
        )
    return output


def _row_from_anchors(
    box: dict,
    start: dict,
    end: dict,
    *,
    match_source: str,
    confidence: float,
    status: str,
    xml_symbol: str,
    geometry_score: float | None = None,
    candidate_margin: float | None = None,
    count_agreement: float | None = None,
    xml_time_confirmed: bool | None = None,
) -> dict:
    start_event = start["event"]
    end_event = end["event"]
    members = []
    for member in [*start.get("members", []), *end.get("members", [])]:
        sequence = member.get("xml_note_sequence")
        key = (sequence, member.get("note_id"), member.get("pitch_name"))
        if not any(item[0] == key for item in members):
            members.append((key, member))
    member_events = sorted(
        (item[1] for item in members),
        key=lambda member: (
            float(member.get("bps_time", 0)),
            int(member.get("note_id"))
            if member.get("note_id") is not None
            else int(member.get("xml_note_sequence", 0)),
        ),
    )
    note_ids = []
    for member in member_events:
        note_id = member.get("note_id")
        if note_id is not None and note_id not in note_ids:
            note_ids.append(note_id)
    pitches = [
        member["pitch_name"]
        for member in member_events
        if member.get("pitch_name")
    ]
    occurrences = (
        _anchor_occurrences(start)
        if start is end
        else _span_occurrences(start, end)
    )
    repeat_mapping_status = _repeat_mapping_status(occurrences)
    repeat_safe = repeat_mapping_is_safe(repeat_mapping_status)
    if status == "matched" and not repeat_safe:
        status = "review"
    row = _base_output_row(box, int(start_event["system"]))
    row.update(
        {
            "start_meas": f"{float(start_event['bps_time']):.3f}",
            "end_meas": f"{float(end_event['bps_time']):.3f}",
            "start_note": (
                note_ids[0]
                if note_ids
                else "NA" if start["target_type"] == "rest" else ""
            ),
            "end_note": (
                note_ids[-1]
                if note_ids
                else "NA" if end["target_type"] == "rest" else ""
            ),
            "connected_note": json.dumps(note_ids) if note_ids else "NA",
            "xml_measure": start_event["xml_measure"],
            "start_xml_measure": start_event["xml_measure"],
            "end_xml_measure": end_event["xml_measure"],
            "xml_symbol": xml_symbol,
            "xml_staff": start_event.get("staff", ""),
            "target_type": (
                "span" if start is not end else start.get("target_type", "note")
            ),
            "note_ids": json.dumps(note_ids),
            "pitches": json.dumps(pitches, ensure_ascii=False),
            "repeat_occurrences_json": json.dumps(occurrences, ensure_ascii=False),
            "repeat_occurrence_count": len(occurrences) if occurrences else 1,
            "repeat_group_id": (
                occurrences[0].get("repeat_group_id", "") if occurrences else ""
            ),
            "repeat_mapping_status": repeat_mapping_status,
            "match_source": match_source,
            "confidence": f"{confidence:.3f}",
            "match_score": f"{confidence:.3f}",
            "confidence_calibrated": "false",
            "geometry_score": f"{geometry_score:.3f}" if geometry_score is not None else "",
            "candidate_margin": (
                f"{candidate_margin:.3f}" if candidate_margin is not None else ""
            ),
            "count_agreement": (
                f"{count_agreement:.3f}" if count_agreement is not None else ""
            ),
            "xml_time_confirmed": str(
                (
                    xml_time_confirmed
                    if xml_time_confirmed is not None
                    else status == "matched"
                )
                and repeat_safe
            ).lower(),
            "status": status,
            "target_x_px": f"{start['x']:.1f}",
            "target_y_px": f"{start['y']:.1f}",
            "end_target_x_px": f"{end['x']:.1f}",
            "end_target_y_px": f"{end['y']:.1f}",
        }
    )
    if (
        box["class"].startswith("stem")
        and start_event.get("stem") in {"up", "down"}
    ):
        row["stem_dir"] = 0 if start_event["stem"] == "down" else 1
    return row


def _notation_rule(class_name: str) -> tuple[str, str] | None:
    for prefix, rule in POINT_NOTATION_RULES.items():
        if class_name.startswith(prefix):
            return rule
    return None


def match_point_notations(
    boxes: list[dict],
    xml_notes: list[dict],
    xml_rests: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match point-like YOLO notations to explicit MusicXML note/rest marks."""

    systems_by_number = {system.number: system for system in systems}
    chord_members: dict[int, list[dict]] = defaultdict(list)
    for note in xml_notes:
        chord_members[note["xml_chord_sequence"]].append(note)

    targets_by_rule: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen = set()
    for note in xml_notes:
        marks = [
            *(("articulation", mark) for mark in note.get("articulation_marks", [])),
            *(("ornament", mark) for mark in note.get("ornament_marks", [])),
        ]
        if note.get("fermata_marks"):
            marks.append(("fermata", "fermata"))
        for rule in marks:
            key = (rule, note["xml_chord_sequence"])
            if key in seen:
                continue
            seen.add(key)
            target = _anchor(
                note,
                systems_by_number,
                measure_x_maps,
                members=sorted(
                    chord_members[note["xml_chord_sequence"]],
                    key=lambda item: (item["midi"], item["xml_note_sequence"]),
                ),
                target_type="chord",
            )
            target["placement"] = next(
                (
                    mark.get("placement", "")
                    for mark in note.get("fermata_marks", [])
                    if rule[0] == "fermata"
                ),
                "",
            )
            targets_by_rule[rule].append(target)
    for rest in xml_rests:
        if rest.get("fermata_marks"):
            target = _anchor(
                rest,
                systems_by_number,
                measure_x_maps,
                members=[],
                target_type="rest",
            )
            target["placement"] = rest["fermata_marks"][0].get("placement", "")
            targets_by_rule[("fermata", "fermata")].append(target)

    output = []
    boxes_by_rule: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for box in boxes:
        rule = _notation_rule(box["class"])
        if rule:
            boxes_by_rule[rule].append(box)
    for rule, rule_boxes in boxes_by_rule.items():
        targets = targets_by_rule.get(rule, [])

        def cost(box: dict, target: dict) -> float:
            box_system = assign_system(box, systems, image_height)
            if box_system != target["event"]["system"]:
                return 1_000_000
            bx, by = box["x"] * image_width, box["y"] * image_height
            placement_violation = (
                box["class"].endswith("Above") and by >= target["y"]
            ) or (
                box["class"].endswith("Below") and by <= target["y"]
            )
            return (
                3.0 * abs(bx - target["x"]) / max(image_width, 1)
                + abs(by - target["y"]) / max(image_height, 1)
                + (0.35 if placement_violation else 0.0)
            )

        count_agreement = (
            min(len(rule_boxes), len(targets))
            / max(len(rule_boxes), len(targets), 1)
        )
        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
            rule_boxes, targets, cost
        ):
            confidence = min(0.99, max(0.20, math.exp(-3.0 * pair_cost)))
            note_links_complete = (
                target.get("target_type") == "rest"
                or all(
                    member.get("note_id") is not None
                    for member in target.get("members", [])
                )
            )
            status = (
                "matched"
                if mutual
                and margin >= 0.05
                and note_links_complete
                and confidence >= auto_accept_threshold(box["class"], 0.65)
                else "review"
            )
            output.append(
                _row_from_anchors(
                    box,
                    target,
                    target,
                    match_source=f"musicxml_{rule[1]}",
                    confidence=confidence,
                    status=status,
                    xml_symbol=rule[1],
                    geometry_score=confidence,
                    candidate_margin=margin,
                    count_agreement=count_agreement,
                    xml_time_confirmed=True,
                )
            )
    return output


def _small_notehead_compatible(
    class_name: str,
    note: dict,
    y: float,
    spacing: float,
    staff_lines: list[float],
) -> bool:
    if not note.get("is_grace"):
        return False
    lower = class_name.lower()
    note_type = str(note.get("note_type", "")).lower()
    if "doublewhole" in lower:
        shape_ok = note_type in {"breve", "long", "maxima"}
    elif "whole" in lower:
        shape_ok = note_type == "whole"
    elif "half" in lower:
        shape_ok = note_type == "half"
    elif "black" in lower:
        shape_ok = note_type not in {"", "whole", "half", "breve", "long", "maxima"}
    else:
        return False
    if not shape_ok:
        return False
    # Pitch geometry determines whether the centre lies on a staff line.  Ledger
    # lines follow the same half-spacing lattice, so use the nearest lattice row.
    line_position = (y - staff_lines[0]) / max(spacing, 1.0)
    line_distance = abs(line_position - round(line_position))
    on_line = line_distance <= 0.25
    return ("online" in lower and on_line) or ("inspace" in lower and not on_line)


def _small_accidental_compatible(class_name: str, note: dict) -> bool:
    if not note.get("is_grace"):
        return False
    accidental = str(note.get("accidental", "")).lower().replace("_", "-")
    lower = class_name.lower()
    expected = {
        "accidentaldoubleflatsmall": {"flat-flat", "double-flat"},
        "accidentaldoublesharpsmall": {"double-sharp", "sharp-sharp"},
        "accidentalflatsmall": {"flat"},
        "accidentalnaturalsmall": {"natural"},
        "accidentalsharpsmall": {"sharp"},
    }
    return accidental in expected.get(lower, set())


def _duration_end(row: dict, members: list[dict]) -> None:
    """Use sounding duration for symbols whose semantics cover a full note."""

    ends = [
        float(member["end_bps_time"])
        for member in members
        if member.get("end_bps_time") is not None
    ]
    if ends:
        row["end_meas"] = f"{max(ends):.3f}"
    elif row.get("status") == "matched":
        row["status"] = "review"
        row["xml_time_confirmed"] = "false"


def match_small_note_symbols(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match small noteheads and accidentals to explicit grace-note XML data."""

    systems_by_number = {system.number: system for system in systems}
    targets = []
    for note in xml_notes:
        if not note.get("is_grace"):
            continue
        anchor = _anchor(note, systems_by_number, measure_x_maps)
        system = systems_by_number[int(note["system"])]
        staff = system.upper if int(note.get("staff", 1)) == 1 else system.lower
        targets.append(
            {
                "anchor": anchor,
                "note": note,
                "spacing": staff.line_spacing,
                "staff_lines": list(staff.lines),
            }
        )

    output = []
    for family in ("notehead", "accidental"):
        family_boxes = [
            box
            for box in boxes
            if box["class"].startswith(family) and box["class"].endswith("Small")
        ]
        class_names = sorted({box["class"] for box in family_boxes})
        for class_name in class_names:
            class_boxes = [box for box in family_boxes if box["class"] == class_name]
            compatible = []
            for target in targets:
                if family == "notehead":
                    ok = _small_notehead_compatible(
                        class_name,
                        target["note"],
                        target["anchor"]["y"],
                        target["spacing"],
                        target["staff_lines"],
                    )
                else:
                    ok = _small_accidental_compatible(class_name, target["note"])
                if ok:
                    compatible.append(target)

            def cost(box: dict, target: dict) -> float:
                if assign_system(box, systems, image_height) != target["note"]["system"]:
                    return 1_000_000
                spacing = max(float(target["spacing"]), 1.0)
                bx, by = box["x"] * image_width, box["y"] * image_height
                expected_x = target["anchor"]["x"]
                if family == "accidental":
                    expected_x -= 1.1 * spacing
                dx = abs(bx - expected_x) / (4.0 * spacing)
                dy = abs(by - target["anchor"]["y"]) / (3.0 * spacing)
                return math.hypot(dx, dy)

            for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
                class_boxes, compatible, cost
            ):
                confidence = max(0.0, min(0.99, math.exp(-2.0 * pair_cost)))
                note = target["note"]
                status = (
                    "matched"
                    if mutual
                    and margin >= 0.05
                    and note.get("note_id") is not None
                    and confidence >= auto_accept_threshold(box["class"], 0.92)
                    else "review"
                )
                row = _row_from_anchors(
                    box,
                    target["anchor"],
                    target["anchor"],
                    match_source=f"musicxml_grace_{family}",
                    confidence=confidence,
                    status=status,
                    xml_symbol=(
                        note.get("accidental") if family == "accidental" else note.get("note_type")
                    ) or family,
                    geometry_score=confidence,
                    candidate_margin=margin,
                    xml_time_confirmed=True,
                )
                if family == "notehead":
                    _duration_end(row, [note])
                output.append(row)
    return output


def _grace_chords(xml_notes: list[dict]) -> list[list[dict]]:
    chords: dict[int, list[dict]] = defaultdict(list)
    for note in xml_notes:
        if note.get("is_grace"):
            chords[int(note["xml_chord_sequence"])].append(note)
    return [
        sorted(members, key=lambda note: (note.get("midi", 0), note["xml_note_sequence"]))
        for _sequence, members in sorted(chords.items())
    ]


def match_small_stems(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match small stems to grace chords and preserve every chord note ID."""

    stem_boxes = [box for box in boxes if box["class"].startswith("stemSmall")]
    systems_by_number = {system.number: system for system in systems}
    targets = []
    for members in _grace_chords(xml_notes):
        event = members[0]
        if not event.get("stem"):
            continue
        anchor = _anchor(
            event,
            systems_by_number,
            measure_x_maps,
            members=members,
            target_type="chord",
        )
        system = systems_by_number[int(event["system"])]
        staff = system.upper if int(event.get("staff", 1)) == 1 else system.lower
        targets.append({"anchor": anchor, "members": members, "spacing": staff.line_spacing})

    def cost(box: dict, target: dict) -> float:
        event = target["anchor"]["event"]
        if assign_system(box, systems, image_height) != event["system"]:
            return 1_000_000
        spacing = max(float(target["spacing"]), 1.0)
        bx, by = box["x"] * image_width, box["y"] * image_height
        member_y = [
            _anchor(member, systems_by_number, measure_x_maps)["y"]
            for member in target["members"]
        ]
        y_low, y_high = min(member_y) - 4 * spacing, max(member_y) + 4 * spacing
        y_penalty = 0.0 if y_low <= by <= y_high else min(abs(by - y_low), abs(by - y_high)) / (6 * spacing)
        return abs(bx - target["anchor"]["x"]) / (4 * spacing) + y_penalty

    output = []
    for class_name in sorted({box["class"] for box in stem_boxes}):
        class_boxes = [box for box in stem_boxes if box["class"] == class_name]
        class_targets = [
            target
            for target in targets
            if class_name != "stemSmallAcciaccatura"
            or any(member.get("grace_slash") for member in target["members"])
        ]
        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
            class_boxes, class_targets, cost
        ):
            confidence = max(0.0, min(0.99, math.exp(-2.0 * pair_cost)))
            members = target["members"]
            status = (
                "matched"
                if mutual
                and margin >= 0.05
                and all(member.get("note_id") is not None for member in members)
                and confidence >= auto_accept_threshold(box["class"], 0.92)
                else "review"
            )
            row = _row_from_anchors(
                box,
                target["anchor"],
                target["anchor"],
                match_source="musicxml_grace_stem",
                confidence=confidence,
                status=status,
                xml_symbol=target["anchor"]["event"].get("stem", "stem"),
                geometry_score=confidence,
                candidate_margin=margin,
                xml_time_confirmed=True,
            )
            _duration_end(row, members)
            output.append(row)
    return output


def _grace_beam_groups(xml_notes: list[dict]) -> list[list[dict]]:
    chords = _grace_chords(xml_notes)
    representatives = sorted(
        ((members[0], members) for members in chords),
        key=lambda item: item[0]["xml_note_sequence"],
    )
    active: dict[tuple[int, str], list[dict]] = {}
    groups = []
    for representative, members in representatives:
        values = {str(value).lower() for value in representative.get("beam_values", [])}
        key = (int(representative.get("staff", 1)), str(representative.get("voice", "1")))
        if "begin" in values:
            active[key] = list(members)
        elif key in active and values & {"continue", "end"}:
            active[key].extend(members)
        if "end" in values and key in active:
            if len(active[key]) >= 2:
                groups.append(active[key])
            del active[key]
    return groups


def match_small_beams(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match beamSmall boxes to explicit begin/continue/end grace-note beams."""

    beam_boxes = [box for box in boxes if box["class"].startswith("beamSmall")]
    systems_by_number = {system.number: system for system in systems}
    targets = []
    for members in _grace_beam_groups(xml_notes):
        start_event, end_event = members[0], members[-1]
        if start_event["system"] != end_event["system"]:
            continue
        start = _anchor(start_event, systems_by_number, measure_x_maps, members=members, target_type="beam")
        end = _anchor(end_event, systems_by_number, measure_x_maps, target_type="beam")
        system = systems_by_number[int(start_event["system"])]
        staff = system.upper if int(start_event.get("staff", 1)) == 1 else system.lower
        targets.append({"start": start, "end": end, "members": members, "spacing": staff.line_spacing})

    def cost(box: dict, target: dict) -> float:
        if assign_system(box, systems, image_height) != target["start"]["event"]["system"]:
            return 1_000_000
        spacing = max(float(target["spacing"]), 1.0)
        bx = box["x"] * image_width
        target_x = (target["start"]["x"] + target["end"]["x"]) / 2
        target_width = abs(target["end"]["x"] - target["start"]["x"])
        return (
            abs(bx - target_x) / (5 * spacing)
            + abs(box["w"] * image_width - target_width) / (10 * spacing)
        )

    output = []
    for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(beam_boxes, targets, cost):
        confidence = max(0.0, min(0.99, math.exp(-1.5 * pair_cost)))
        members = target["members"]
        status = (
            "matched"
            if mutual
            and margin >= 0.05
            and all(member.get("note_id") is not None for member in members)
            and confidence >= auto_accept_threshold(box["class"], 0.90)
            else "review"
        )
        row = _row_from_anchors(
            box,
            target["start"],
            target["end"],
            match_source="musicxml_grace_beam",
            confidence=confidence,
            status=status,
            xml_symbol="beam",
            geometry_score=confidence,
            candidate_margin=margin,
            xml_time_confirmed=True,
        )
        _duration_end(row, members)
        output.append(row)
    return output


def match_small_flags(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match un-beamed small grace-note flags by duration and stem direction."""

    flag_boxes = [box for box in boxes if box["class"].startswith("flag")]
    systems_by_number = {system.number: system for system in systems}
    targets = []
    for members in _grace_chords(xml_notes):
        event = members[0]
        if event.get("beam_values") or event.get("stem") not in {"up", "down"}:
            continue
        system = systems_by_number[int(event["system"])]
        staff = system.upper if int(event.get("staff", 1)) == 1 else system.lower
        targets.append(
            {
                "anchor": _anchor(
                    event,
                    systems_by_number,
                    measure_x_maps,
                    members=members,
                    target_type="chord",
                ),
                "members": members,
                "spacing": staff.line_spacing,
            }
        )

    def compatible(class_name: str, target: dict) -> bool:
        match = re.fullmatch(
            r"flag(128th|64th|32nd|16th|8th)(Up|Down)Small",
            class_name,
        )
        if match is None:
            return False
        duration_name, direction = match.groups()
        expected_type = "eighth" if duration_name == "8th" else duration_name
        event = target["anchor"]["event"]
        return (
            str(event.get("note_type", "")).lower() == expected_type.lower()
            and str(event.get("stem", "")).lower() == direction.lower()
        )

    output = []
    for class_name in sorted({box["class"] for box in flag_boxes}):
        class_boxes = [box for box in flag_boxes if box["class"] == class_name]
        class_targets = [target for target in targets if compatible(class_name, target)]

        def cost(box: dict, target: dict) -> float:
            event = target["anchor"]["event"]
            if assign_system(box, systems, image_height) != event["system"]:
                return 1_000_000
            spacing = max(float(target["spacing"]), 1.0)
            direction_offset = 0.9 * spacing if event["stem"] == "up" else -0.9 * spacing
            expected_x = target["anchor"]["x"] + direction_offset
            return abs(box["x"] * image_width - expected_x) / (4 * spacing)

        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
            class_boxes, class_targets, cost
        ):
            confidence = max(0.0, min(0.99, math.exp(-2.0 * pair_cost)))
            members = target["members"]
            status = (
                "matched"
                if mutual
                and margin >= 0.05
                and all(member.get("note_id") is not None for member in members)
                and confidence >= auto_accept_threshold(box["class"], 0.92)
                else "review"
            )
            row = _row_from_anchors(
                box,
                target["anchor"],
                target["anchor"],
                match_source="musicxml_grace_flag",
                confidence=confidence,
                status=status,
                xml_symbol=target["anchor"]["event"].get("note_type", "flag"),
                geometry_score=confidence,
                candidate_margin=margin,
                xml_time_confirmed=True,
            )
            _duration_end(row, members)
            output.append(row)
    return output


def _direction_spans(markers: list[dict], kind: str) -> list[tuple[dict, dict]]:
    open_markers: dict[tuple[str, int], dict] = {}
    spans = []
    for marker in sorted(markers, key=lambda item: (item["xml_measure_index"], item["bps_time"])):
        if marker.get("kind") != kind:
            continue
        key = (str(marker.get("number", "1")), int(marker.get("staff", 1)))
        marker_type = str(marker.get("type", "")).lower()
        opening_types = (
            {"crescendo", "diminuendo"}
            if kind == "wedge"
            else {"up", "down", "start", "sostenuto", "resume"}
        )
        if marker_type in opening_types:
            open_markers[key] = marker
        elif marker_type in {"stop", "discontinue"} and key in open_markers:
            spans.append((open_markers.pop(key), marker))
        elif marker_type == "change":
            if key in open_markers:
                spans.append((open_markers[key], marker))
            open_markers[key] = marker
    return spans


def _normalized_direction_text(value: object) -> tuple[str, list[str]]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    tokens = re.findall(r"[a-z0-9]+", text.casefold().replace("ß", "ss"))
    return "".join(tokens), tokens


def _text_direction_matches(class_name: str, event: dict) -> bool:
    if event.get("kind") != "words":
        return False
    compact, tokens = _normalized_direction_text(event.get("text"))
    if not compact:
        return False

    if class_name == "dynamicCrescendo":
        return "cresc" in compact and "decresc" not in compact
    if class_name == "dynamicDiminuendo":
        return any(value in compact for value in ("dim", "dimin", "decresc"))
    # A Long box represents an extended printed phrase. MusicXML in this
    # dataset stores many such phrases as several disconnected words elements;
    # a single point event cannot provide a trustworthy end time.
    if class_name.startswith(("dynamicCrescendo", "dynamicDiminuendo")):
        return False

    aliases = {
        "tempoATempo": ("atempo",),
        "tempoInTempo": ("intempo",),
        "tempoRitardando": ("ritar", "ritard"),
        "tempoRitardandoLong": ("ritar", "ritard"),
        "tempoTempo": ("tempo",),
        "tempoMunuetto": ("minuetto", "menuetto"),
        "termLegato": ("legato", "ligato"),
        "termMarschmäBig": ("marschmassig",),
        "termMitLebhaftigkeitUndDurchausMitEmpfindungUndAusdrunk": (
            "mitlebhaftigkeitunddurchausmitempfindungundausdruck",
        ),
        "termPiù": ("piu",),
    }
    if class_name in aliases:
        return any(alias in compact for alias in aliases[class_name])

    for prefix in ("tempo", "term"):
        if class_name.startswith(prefix):
            suffix, suffix_tokens = _normalized_direction_text(
                class_name[len(prefix) :]
            )
            if not suffix:
                return False
            if len(suffix_tokens) == 1 and len(suffix) <= 3:
                return suffix in tokens
            return suffix in compact

    if class_name in {
        "IlFine",
        "LangsamUndSehnsuchtvoll",
        "MarciaDaCapoAlFineSenzaRepetizione",
        "keyboardMitEinerSaite",
        "keyboardSulUnaCorda",
    }:
        expected, _tokens = _normalized_direction_text(class_name.removeprefix("keyboard"))
        return expected in compact
    return False


def match_text_directions(
    boxes: list[dict],
    text_directions: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match printed tempo/term/words boxes to MusicXML direction words."""

    supported_boxes = [
        box
        for box in boxes
        if box["class"].startswith(("tempo", "term"))
        or box["class"]
        in {
            "IlFine",
            "LangsamUndSehnsuchtvoll",
            "MarciaDaCapoAlFineSenzaRepetizione",
            "keyboardMitEinerSaite",
            "keyboardSulUnaCorda",
            "dynamicCrescendo",
            "dynamicDiminuendo",
        }
    ]
    systems_by_number = {system.number: system for system in systems}
    output = []
    for class_name in sorted({box["class"] for box in supported_boxes}):
        class_boxes = [box for box in supported_boxes if box["class"] == class_name]
        targets = [
            {
                "anchor": _anchor(
                    event,
                    systems_by_number,
                    measure_x_maps,
                    members=[],
                    target_type="direction",
                ),
                "event": event,
            }
            for event in text_directions
            if _text_direction_matches(class_name, event)
        ]

        def cost(box: dict, target: dict) -> float:
            event = target["event"]
            if assign_system(box, systems, image_height) != event["system"]:
                return 1_000_000
            system = systems_by_number[int(event["system"])]
            spacing = max(
                (system.upper.line_spacing + system.lower.line_spacing) / 2,
                1.0,
            )
            return abs(
                box["x"] * image_width - target["anchor"]["x"]
            ) / (7 * spacing)

        count_agreement = min(len(class_boxes), len(targets)) / max(
            len(class_boxes), len(targets), 1
        )
        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
            class_boxes, targets, cost
        ):
            confidence = max(0.0, min(0.99, math.exp(-1.8 * pair_cost)))
            status = (
                "matched"
                if mutual
                and margin >= 0.05
                and confidence >= auto_accept_threshold(box["class"], 0.90)
                else "review"
            )
            output.append(
                _row_from_anchors(
                    box,
                    target["anchor"],
                    target["anchor"],
                    match_source="musicxml_direction_words",
                    confidence=confidence,
                    status=status,
                    xml_symbol=target["event"].get("text", "words"),
                    geometry_score=confidence,
                    candidate_margin=margin,
                    count_agreement=count_agreement,
                    xml_time_confirmed=True,
                )
            )
    return output


def _wavy_line_spans(xml_notes: list[dict]) -> list[tuple[dict, dict]]:
    open_lines: dict[tuple[int, str, str], dict] = {}
    spans = []
    for note in sorted(xml_notes, key=lambda item: item["xml_note_sequence"]):
        for mark in note.get("wavy_line_marks", []):
            key = (
                int(note.get("staff", 1)),
                str(note.get("voice", "1")),
                str(mark.get("number", "1")),
            )
            mark_type = str(mark.get("type", "")).lower()
            if mark_type == "start":
                open_lines[key] = note
            elif mark_type == "stop" and key in open_lines:
                spans.append((open_lines.pop(key), note))
    return spans


def match_wavy_line_spans(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match ornament wiggle boxes to paired MusicXML wavy-line endpoints."""

    class_boxes = [box for box in boxes if box["class"] == "ornamentWiggleTrill"]
    systems_by_number = {system.number: system for system in systems}
    targets = []
    for start_event, end_event in _wavy_line_spans(xml_notes):
        if start_event["system"] != end_event["system"]:
            continue
        start = _anchor(start_event, systems_by_number, measure_x_maps)
        end = _anchor(end_event, systems_by_number, measure_x_maps)
        targets.append({"start": start, "end": end})

    def cost(box: dict, target: dict) -> float:
        if assign_system(box, systems, image_height) != target["start"]["event"]["system"]:
            return 1_000_000
        system = systems_by_number[int(target["start"]["event"]["system"])]
        spacing = max(system.upper.line_spacing, 1.0)
        center = (target["start"]["x"] + target["end"]["x"]) / 2
        width = abs(target["end"]["x"] - target["start"]["x"])
        return (
            abs(box["x"] * image_width - center) / (6 * spacing)
            + abs(box["w"] * image_width - width) / (10 * spacing)
        )

    output = []
    for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
        class_boxes, targets, cost
    ):
        confidence = max(0.0, min(0.99, math.exp(-1.5 * pair_cost)))
        note_ids_present = all(
            anchor["event"].get("note_id") is not None
            for anchor in (target["start"], target["end"])
        )
        status = (
            "matched"
            if mutual
            and margin >= 0.05
            and note_ids_present
            and confidence >= auto_accept_threshold(box["class"], 0.80)
            else "review"
        )
        output.append(
            _row_from_anchors(
                box,
                target["start"],
                target["end"],
                match_source="musicxml_wavy_line_span",
                confidence=confidence,
                status=status,
                xml_symbol="wavy-line",
                geometry_score=confidence,
                candidate_margin=margin,
                xml_time_confirmed=True,
            )
        )
    return output


def match_direction_symbols(
    boxes: list[dict],
    xml_notes: list[dict],
    direction_markers: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Match ottava spans and pedal endpoints from MusicXML direction data."""

    systems_by_number = {system.number: system for system in systems}
    output = []
    ottava_boxes = [box for box in boxes if box["class"].startswith("ottavaBracket")]
    ottava_targets = []
    for start_event, end_event in _direction_spans(direction_markers, "octave"):
        if start_event["system"] != end_event["system"]:
            continue
        members = [
            note for note in xml_notes
            if int(note.get("staff", 1)) == int(start_event.get("staff", 1))
            and float(start_event["bps_time"])
            <= float(note["bps_time"])
            < float(end_event["bps_time"])
        ]
        start = _anchor(start_event, systems_by_number, measure_x_maps, members=members, target_type="ottava")
        end = _anchor(end_event, systems_by_number, measure_x_maps, target_type="ottava")
        ottava_targets.append({"start": start, "end": end, "members": members})

    def ottava_cost(box: dict, target: dict) -> float:
        if assign_system(box, systems, image_height) != target["start"]["event"]["system"]:
            return 1_000_000
        system = systems_by_number[int(target["start"]["event"]["system"])]
        spacing = max(system.upper.line_spacing, 1.0)
        target_x = (target["start"]["x"] + target["end"]["x"]) / 2
        target_width = abs(target["end"]["x"] - target["start"]["x"])
        return abs(box["x"] * image_width - target_x) / (8 * spacing) + abs(box["w"] * image_width - target_width) / (12 * spacing)

    for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(ottava_boxes, ottava_targets, ottava_cost):
        confidence = max(0.0, min(0.99, math.exp(-1.5 * pair_cost)))
        members = target["members"]
        status = (
            "matched"
            if mutual and margin >= 0.05 and members
            and all(member.get("note_id") is not None for member in members)
            and confidence >= auto_accept_threshold(box["class"], 0.90)
            else "review"
        )
        output.append(
            _row_from_anchors(
                box, target["start"], target["end"],
                match_source="musicxml_octave_shift", confidence=confidence,
                status=status, xml_symbol="octave-shift", geometry_score=confidence,
                candidate_margin=margin, xml_time_confirmed=True,
            )
        )

    for direction_type, class_name in (
        ("crescendo", "dynamicCrescendoHairpin"),
        ("diminuendo", "dynamicDiminuendoHairpin"),
    ):
        wedge_boxes = [box for box in boxes if box["class"] == class_name]
        wedge_targets = []
        for start_event, end_event in _direction_spans(direction_markers, "wedge"):
            if (
                start_event.get("type") != direction_type
                or start_event["system"] != end_event["system"]
            ):
                continue
            wedge_targets.append(
                {
                    "start": _anchor(
                        start_event,
                        systems_by_number,
                        measure_x_maps,
                        members=[],
                        target_type="direction_span",
                    ),
                    "end": _anchor(
                        end_event,
                        systems_by_number,
                        measure_x_maps,
                        members=[],
                        target_type="direction_span",
                    ),
                }
            )

        def wedge_cost(box: dict, target: dict) -> float:
            if assign_system(box, systems, image_height) != target["start"]["event"]["system"]:
                return 1_000_000
            system = systems_by_number[int(target["start"]["event"]["system"])]
            spacing = max(system.upper.line_spacing, 1.0)
            center = (target["start"]["x"] + target["end"]["x"]) / 2
            width = abs(target["end"]["x"] - target["start"]["x"])
            return (
                abs(box["x"] * image_width - center) / (7 * spacing)
                + abs(box["w"] * image_width - width) / (12 * spacing)
            )

        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
            wedge_boxes, wedge_targets, wedge_cost
        ):
            confidence = max(0.0, min(0.99, math.exp(-1.5 * pair_cost)))
            status = (
                "matched"
                if mutual
                and margin >= 0.05
                and confidence >= auto_accept_threshold(box["class"], 0.85)
                else "review"
            )
            output.append(
                _row_from_anchors(
                    box,
                    target["start"],
                    target["end"],
                    match_source="musicxml_wedge_span",
                    confidence=confidence,
                    status=status,
                    xml_symbol=direction_type,
                    geometry_score=confidence,
                    candidate_margin=margin,
                    xml_time_confirmed=True,
                )
            )

    pedal_rules = {
        "keyboardPed": {"start", "sostenuto", "resume"},
        "keyboardPedalPed": {"start", "sostenuto", "resume"},
        "keyboardPedalUp": {"stop", "discontinue"},
    }
    for class_prefix, accepted_types in pedal_rules.items():
        pedal_boxes = [
            box for box in boxes
            if box["class"] == class_prefix
        ]
        pedal_targets = []
        for marker in direction_markers:
            if marker.get("kind") != "pedal" or marker.get("type") not in accepted_types:
                continue
            anchor = _anchor(marker, systems_by_number, measure_x_maps, members=[], target_type="pedal")
            pedal_targets.append({"anchor": anchor})

        def pedal_cost(box: dict, target: dict) -> float:
            event = target["anchor"]["event"]
            if assign_system(box, systems, image_height) != event["system"]:
                return 1_000_000
            system = systems_by_number[int(event["system"])]
            spacing = max(system.lower.line_spacing, 1.0)
            return abs(box["x"] * image_width - target["anchor"]["x"]) / (5 * spacing)

        for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(pedal_boxes, pedal_targets, pedal_cost):
            confidence = max(0.0, min(0.99, math.exp(-2.0 * pair_cost)))
            status = (
                "matched" if mutual and margin >= 0.05
                and confidence >= auto_accept_threshold(box["class"], 0.90)
                else "review"
            )
            output.append(
                _row_from_anchors(
                    box, target["anchor"], target["anchor"],
                    match_source="musicxml_pedal_direction", confidence=confidence,
                    status=status, xml_symbol="pedal", geometry_score=confidence,
                    candidate_margin=margin, xml_time_confirmed=True,
                )
            )
    return output


def _find_candidate_note(
    notes: list[dict],
    *,
    bps_time: str,
    pitch: str,
    measure: object,
    staff: object,
    voice: object,
) -> dict | None:
    candidates = [
        note
        for note in notes
        if note["pitch_name"] == pitch
        and str(note["xml_measure"]) == str(measure)
        and str(note["staff"]) == str(staff)
        and str(note["voice"]) == str(voice)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda note: abs(note["bps_time"] - float(bps_time)))


def _slur_target_segments(
    target: dict,
    systems_by_number: dict[int, SystemGeometry],
) -> list[dict]:
    """Expand one XML slur into one scan target per printed system segment."""

    start = target["start"]
    end = target["end"]
    start_system = int(start["event"]["system"])
    end_system = int(end["event"]["system"])
    low, high = sorted((start_system, end_system))
    orientation = target["candidate"].get("orientation") or "over"
    segments = []
    for system_number in range(low, high + 1):
        system = systems_by_number.get(system_number)
        if system is None:
            continue
        if start_system == end_system:
            segment_type = "full"
            x0, y0 = start["x"], start["y"]
            x1, y1 = end["x"], end["y"]
        elif system_number == start_system:
            segment_type = "start"
            x0, y0 = start["x"], start["y"]
            x1, y1 = system.x_right, start["y"]
        elif system_number == end_system:
            segment_type = "end"
            x0, y0 = system.x_left, end["y"]
            x1, y1 = end["x"], end["y"]
        else:
            segment_type = "middle"
            x0, x1 = system.x_left, system.x_right
            y0 = y1 = system.upper.center
        staff_spacings = [
            system.upper.line_spacing,
            system.lower.line_spacing,
        ]
        segments.append(
            {
                **target,
                "segment_type": segment_type,
                "segment_system": system_number,
                "segment_x0": min(x0, x1),
                "segment_x1": max(x0, x1),
                "segment_y0": y0,
                "segment_y1": y1,
                "staff_spacing": sum(staff_spacings) / len(staff_spacings),
                "orientation": orientation,
            }
        )
    return segments


def _slur_segment_score(
    box: dict,
    target: dict,
    box_system: int,
    image_width: int,
    image_height: int,
) -> float:
    """Return conservative scan/XML slur compatibility in the range [0, 1]."""

    if int(target["segment_system"]) != int(box_system):
        return 0.0
    box_center_x = box["x"] * image_width
    box_center_y = box["y"] * image_height
    box_width = box["w"] * image_width
    predicted_center_x = (target["segment_x0"] + target["segment_x1"]) / 2
    predicted_span = max(8.0, target["segment_x1"] - target["segment_x0"])
    x_error = abs(box_center_x - predicted_center_x)
    x_tolerance = max(18.0, predicted_span * 0.35)
    x_center_score = math.exp(-0.5 * (x_error / x_tolerance) ** 2)

    width_ratio = max(0.05, box_width / predicted_span)
    width_score = math.exp(-abs(math.log(width_ratio)) / 0.65)

    spacing = max(1.0, float(target["staff_spacing"]))
    if target["orientation"] == "under":
        note_side_y = max(target["segment_y0"], target["segment_y1"])
        expected_y = note_side_y + min(
            max(spacing * 1.1, predicted_span * 0.38), spacing * 3.5
        )
        wrong_side = max(0.0, note_side_y - box_center_y)
    else:
        note_side_y = min(target["segment_y0"], target["segment_y1"])
        expected_y = note_side_y - min(
            max(spacing * 1.1, predicted_span * 0.38), spacing * 3.5
        )
        wrong_side = max(0.0, box_center_y - note_side_y)
    vertical_error = abs(box_center_y - expected_y)
    vertical_score = math.exp(-0.5 * (vertical_error / (spacing * 2.0)) ** 2)
    orientation_score = math.exp(-wrong_side / spacing)
    return (
        0.42 * x_center_score
        + 0.28 * width_score
        + 0.23 * vertical_score
        + 0.07 * orientation_score
    )


def _cross_page_span_class_matches(
    box_class: str,
    span: dict,
    endpoint_role: str,
) -> bool:
    kind = str(span.get("span_type", ""))
    span_class = str(span.get("class", ""))
    if kind in {"slur", "tie"}:
        return box_class == kind
    if kind == "wavy-line":
        return box_class == "ornamentWiggleTrill"
    if kind == "octave-shift":
        return box_class.startswith("ottavaBracket")
    if kind == "wedge":
        return box_class == span_class
    if kind == "pedal":
        if endpoint_role == "end":
            return box_class == "keyboardPedalUp"
        return box_class in {"keyboardPed", "keyboardPedalPed"}
    return False


def match_cross_page_spans(
    boxes: list[dict],
    xml_notes: list[dict],
    whole_score_spans: list[dict],
    page_number: int,
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Attach complete whole-score endpoints to cross-page YOLO segments.

    A single page cannot geometrically verify the remote endpoint, so these
    rows intentionally remain in review even when both XML times are known.
    """

    systems_by_number = {system.number: system for system in systems}
    targets = []
    for span in whole_score_spans:
        if (
            str(span.get("cross_page", "")).strip().lower() != "true"
            or span.get("status") != "paired"
        ):
            continue
        start_page = str(span.get("start_page", ""))
        end_page = str(span.get("end_page", ""))
        current_page = str(page_number)
        if current_page == start_page:
            role = "start"
        elif current_page == end_page:
            role = "end"
        else:
            continue
        note_id = span.get(f"{role}_note", "")
        xml_measure = span.get(f"{role}_xml_measure", "")
        staff = span.get(f"{role}_staff", "")
        local_notes = [
            note
            for note in xml_notes
            if str(note.get("note_id", "")) == str(note_id)
        ]
        if not local_notes:
            local_notes = [
                note
                for note in xml_notes
                if str(note.get("xml_measure", "")) == str(xml_measure)
                and (
                    str(staff) == ""
                    or str(note.get("staff", "")) == str(staff)
                )
            ]
        if not local_notes:
            continue
        time_value = span.get(f"{role}_meas", "")
        try:
            local_note = min(
                local_notes,
                key=lambda note: abs(
                    float(note.get("bps_time", 0)) - float(time_value)
                ),
            )
        except (TypeError, ValueError):
            local_note = local_notes[0]
        local_anchor = _anchor(
            local_note,
            systems_by_number,
            measure_x_maps,
        )
        system = systems_by_number.get(int(local_note["system"]))
        if system is None:
            continue
        if role == "start":
            segment_x0, segment_x1 = local_anchor["x"], system.x_right
        else:
            segment_x0, segment_x1 = system.x_left, local_anchor["x"]
        targets.append(
            {
                "span": span,
                "role": role,
                "local": local_anchor,
                "system": system,
                "segment_x0": min(segment_x0, segment_x1),
                "segment_x1": max(segment_x0, segment_x1),
            }
        )

    compatible_boxes = [
        box
        for box in boxes
        if any(
            _cross_page_span_class_matches(
                box["class"], target["span"], target["role"]
            )
            for target in targets
        )
    ]

    def cost(box: dict, target: dict) -> float:
        if not _cross_page_span_class_matches(
            box["class"], target["span"], target["role"]
        ):
            return 1_000_000
        if assign_system(box, systems, image_height) != target["local"]["event"]["system"]:
            return 1_000_000
        bx = box["x"] * image_width
        if target["span"].get("span_type") == "pedal":
            return abs(bx - target["local"]["x"]) / max(image_width, 1)
        target_center = (target["segment_x0"] + target["segment_x1"]) / 2
        target_width = target["segment_x1"] - target["segment_x0"]
        return (
            2.0 * abs(bx - target_center) / max(image_width, 1)
            + abs(box["w"] * image_width - target_width)
            / max(image_width, 1)
        )

    output = []
    for box, target, pair_cost, margin, mutual in _mutual_geometry_pairs(
        compatible_boxes, targets, cost
    ):
        confidence = min(0.99, max(0.20, math.exp(-3.0 * pair_cost)))
        span = target["span"]
        semantic_complete = bool(span.get("start_meas") and span.get("end_meas"))
        row = _row_from_anchors(
            box,
            target["local"],
            target["local"],
            match_source="whole_score_cross_page_span_candidate",
            confidence=confidence,
            status="review",
            xml_symbol=str(span.get("span_type", "span")),
            geometry_score=confidence,
            candidate_margin=margin,
            xml_time_confirmed=semantic_complete,
        )
        connected = span.get("connected_note", "") or "NA"
        row.update(
            {
                "start_meas": span.get("start_meas", ""),
                "end_meas": span.get("end_meas", ""),
                "start_note": span.get("start_note", "") or "NA",
                "end_note": span.get("end_note", "") or "NA",
                "connected_note": connected,
                "note_ids": connected,
                "start_xml_measure": span.get("start_xml_measure", ""),
                "end_xml_measure": span.get("end_xml_measure", ""),
                "cross_page_span_id": span.get("span_id", ""),
                "start_xml_page": span.get("start_page", ""),
                "end_xml_page": span.get("end_page", ""),
                "target_type": "span",
                "xml_time_confirmed": str(semantic_complete).lower(),
                "target_x_px": f"{target['segment_x0']:.1f}",
                "target_y_px": f"{target['local']['y']:.1f}",
                "end_target_x_px": f"{target['segment_x1']:.1f}",
                "end_target_y_px": f"{target['local']['y']:.1f}",
                "status": "review",
            }
        )
        if not mutual:
            row["confidence"] = f"{min(confidence, 0.49):.3f}"
            row["match_score"] = row["confidence"]
        output.append(row)
    return output


def match_xml_spans(
    boxes: list[dict],
    xml_notes: list[dict],
    bps_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
    note_x_overrides: dict[int, float] | None = None,
) -> list[dict]:
    """Match YOLO slur/tie boxes to paired MusicXML endpoints."""

    systems_by_number = {system.number: system for system in systems}
    def chord_key(note: dict) -> tuple[int, int, int] | None:
        chord_sequence = note.get("xml_chord_sequence")
        if chord_sequence is None:
            return None
        return (
            int(note.get("system", 0)),
            int(note.get("staff", 1)),
            int(chord_sequence),
        )

    chord_members = index_chord_members(xml_notes, chord_key)

    def slur_chord_members(note: dict) -> list[dict]:
        """Return every note in the endpoint chord selected by a slur."""

        chord_sequence = note.get("xml_chord_sequence")
        if chord_sequence is None:
            return [note]
        key = chord_key(note)
        return sorted(
            chord_members.get(key, [note]),
            key=lambda member: (
                int(member.get("note_id"))
                if member.get("note_id") is not None
                else int(member.get("xml_note_sequence", 0))
            ),
        )

    targets_by_class: dict[str, list[dict]] = defaultdict(list)
    slurs, _slur_issues = build_slur_candidates(xml_notes, bps_notes)
    ties, _tie_issues = build_tie_candidates(xml_notes, bps_notes)
    for class_name, candidates in (("slur", slurs), ("tie", ties)):
        for candidate in candidates:
            if class_name == "slur":
                start = _find_candidate_note(
                    xml_notes,
                    bps_time=candidate["start_meas"],
                    pitch=candidate["start_pitch"],
                    measure=candidate["start_xml_measure"],
                    staff=candidate["start_staff"],
                    voice=candidate["start_voice"],
                )
                end = _find_candidate_note(
                    xml_notes,
                    bps_time=candidate["end_meas"],
                    pitch=candidate["end_pitch"],
                    measure=candidate["end_xml_measure"],
                    staff=candidate["end_staff"],
                    voice=candidate["end_voice"],
                )
            else:
                start = _find_candidate_note(
                    xml_notes,
                    bps_time=candidate["start_meas"],
                    pitch=candidate["pitch"],
                    measure=candidate["start_xml_measure"],
                    staff=candidate["staff"],
                    voice=candidate["voice"],
                )
                end = _find_candidate_note(
                    xml_notes,
                    bps_time=candidate["end_meas"],
                    pitch=candidate["pitch"],
                    measure=candidate["end_xml_measure"],
                    staff=candidate["staff"],
                    voice=candidate["voice"],
                )
            if start is None or end is None:
                continue
            start_anchor = _anchor(
                start,
                systems_by_number,
                measure_x_maps,
                members=(slur_chord_members(start) if class_name == "slur" else None),
            )
            end_anchor = _anchor(
                end,
                systems_by_number,
                measure_x_maps,
                members=(slur_chord_members(end) if class_name == "slur" else None),
            )
            if note_x_overrides:
                start_sequence = start.get("xml_note_sequence")
                end_sequence = end.get("xml_note_sequence")
                if start_sequence in note_x_overrides:
                    start_anchor["x"] = note_x_overrides[start_sequence]
                if end_sequence in note_x_overrides:
                    end_anchor["x"] = note_x_overrides[end_sequence]
            targets_by_class[class_name].append(
                {
                    "candidate": candidate,
                    "start": start_anchor,
                    "end": end_anchor,
                }
            )

    output = []
    for class_name in ("slur", "tie"):
        class_boxes = [box for box in boxes if box["class"] == class_name]
        base_targets = targets_by_class[class_name]
        targets = (
            [
                segment
                for target in base_targets
                for segment in _slur_target_segments(target, systems_by_number)
            ]
            if class_name == "slur"
            else base_targets
        )

        def cost(box: dict, target: dict) -> float:
            box_system = assign_system(box, systems, image_height)
            if class_name == "slur":
                return 1.0 - _slur_segment_score(
                    box,
                    target,
                    box_system,
                    image_width,
                    image_height,
                )
            start, end = target["start"], target["end"]
            if box_system not in {
                start["event"]["system"],
                end["event"]["system"],
            }:
                return 1_000_000
            bx = box["x"] * image_width
            by = box["y"] * image_height
            if start["event"]["system"] == end["event"]["system"]:
                center_x = (start["x"] + end["x"]) / 2
                center_y = (start["y"] + end["y"]) / 2
                expected_width = abs(end["x"] - start["x"])
                return (
                    2.0 * abs(bx - center_x) / max(image_width, 1)
                    + 0.35 * abs(by - center_y) / max(image_height, 1)
                    + abs(box["w"] * image_width - expected_width)
                    / max(image_width, 1)
                )
            local = start if box_system == start["event"]["system"] else end
            return (
                2.0 * abs(bx - local["x"]) / max(image_width, 1)
                + 0.35 * abs(by - local["y"]) / max(image_height, 1)
                + 0.25
            )

        for box, target, pair_cost in _greedy_pairs(class_boxes, targets, cost):
            if class_name == "slur":
                confidence = max(0.0, min(0.99, 1.0 - pair_cost))
                if confidence < 0.42:
                    continue
                ranked_for_box = [
                    (
                        1.0 - cost(box, option),
                        option,
                    )
                    for option in targets
                    if int(option["segment_system"])
                    == assign_system(box, systems, image_height)
                ]
                ranked_for_box.sort(key=lambda item: item[0], reverse=True)
                best_target = ranked_for_box[0][1] if ranked_for_box else None
                second_score = (
                    ranked_for_box[1][0] if len(ranked_for_box) > 1 else 0.0
                )
                margin = confidence - second_score
                feasible_boxes = [
                    candidate_box
                    for candidate_box in class_boxes
                    if assign_system(candidate_box, systems, image_height)
                    == int(target["segment_system"])
                ]
                mutual_best = bool(feasible_boxes) and min(
                    feasible_boxes, key=lambda candidate_box: cost(candidate_box, target)
                ) is box
                box_best = best_target is target
            else:
                confidence = min(0.99, max(0.20, math.exp(-3.0 * pair_cost)))
                ranked_for_box = sorted(
                    (
                        (candidate_cost, option)
                        for option in targets
                        if (candidate_cost := cost(box, option)) < 1_000_000
                    ),
                    key=lambda item: item[0],
                )
                best_target = ranked_for_box[0][1] if ranked_for_box else None
                second_score = (
                    math.exp(-3.0 * ranked_for_box[1][0])
                    if len(ranked_for_box) > 1
                    else 0.0
                )
                margin = max(0.0, confidence - second_score)
                feasible_boxes = [
                    candidate_box
                    for candidate_box in class_boxes
                    if cost(candidate_box, target) < 1_000_000
                ]
                mutual_best = bool(feasible_boxes) and min(
                    feasible_boxes,
                    key=lambda candidate_box: cost(candidate_box, target),
                ) is box
                box_best = best_target is target
            candidate = target["candidate"]
            confirmed = candidate["status"] == "time_confirmed"
            note_id_ambiguous = bool(candidate.get("note_id_ambiguous"))
            status = (
                "matched"
                if (
                    confirmed
                    and confidence
                    >= auto_accept_threshold(
                        class_name, 0.82 if class_name == "slur" else 0.60
                    )
                    and margin >= (0.12 if class_name == "slur" else 0.08)
                    and mutual_best
                    and box_best
                    and (
                        class_name != "slur"
                        or target.get("segment_type") == "full"
                    )
                )
                else "review"
            )
            row = _row_from_anchors(
                box,
                target["start"],
                target["end"],
                match_source=(
                    f"musicxml_slur_{target.get('segment_type', 'full')}_endpoint_candidate"
                    + ("_ambiguous_bps_unison" if note_id_ambiguous else "")
                    if class_name == "slur"
                    else "musicxml_tie_endpoints"
                    + ("_ambiguous_bps_unison" if note_id_ambiguous else "")
                ),
                confidence=confidence,
                status=status,
                xml_symbol=class_name,
                geometry_score=confidence,
                candidate_margin=margin,
                xml_time_confirmed=confirmed,
            )
            # connected_note contains every note in both endpoint chords for a
            # slur, while start_note/end_note continue to identify the exact
            # MusicXML noteheads carrying the slur marks.  A tie is pitch-
            # specific and therefore deliberately remains a two-note span.
            if class_name == "slur":
                connected = endpoint_note_ids(
                    target["start"]["event"],
                    target["end"]["event"],
                    start_members=target["start"].get("members"),
                    end_members=target["end"].get("members"),
                    expand_chords=True,
                )
                row["start_note"] = target["start"]["event"].get("note_id", "")
                row["end_note"] = target["end"]["event"].get("note_id", "")
                row["connected_note"] = json.dumps(connected)
                row["note_ids"] = json.dumps(connected)
            output.append(row)
    return output


def _tuplet_groups(notes: list[dict]) -> list[dict]:
    ordered = sorted(notes, key=lambda note: note["xml_note_sequence"])
    groups = []
    for start in ordered:
        if not any(mark["type"] == "start" for mark in start.get("tuplet_marks", [])):
            continue
        count = int(start.get("actual_notes") or 0)
        if not count:
            continue
        members = [
            note
            for note in ordered
            if note["xml_measure"] == start["xml_measure"]
            and note["staff"] == start["staff"]
            and note["voice"] == start["voice"]
            and note.get("actual_notes") == count
            and note["xml_note_sequence"] >= start["xml_note_sequence"]
        ][:count]
        if len(members) == count:
            groups.append({"actual_notes": count, "members": members})
    return groups


def match_tuplets(
    boxes: list[dict],
    xml_notes: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    systems_by_number = {system.number: system for system in systems}
    groups = []
    for group in _tuplet_groups(xml_notes):
        anchors = [
            _anchor(note, systems_by_number, measure_x_maps)
            for note in group["members"]
        ]
        groups.append({**group, "anchors": anchors})
    tuplet_boxes = [box for box in boxes if box["class"].startswith("tuplet")]

    def box_count(box: dict) -> int | None:
        match = re.fullmatch(r"tuplet(\d+)", box["class"])
        return int(match.group(1)) if match else None

    def cost(box: dict, group: dict) -> float:
        count = box_count(box)
        if count is not None and count != group["actual_notes"]:
            return 1_000_000
        box_system = assign_system(box, systems, image_height)
        if box_system != group["members"][0]["system"]:
            return 1_000_000
        x = sum(anchor["x"] for anchor in group["anchors"]) / len(group["anchors"])
        y = sum(anchor["y"] for anchor in group["anchors"]) / len(group["anchors"])
        return (
            2.0 * abs(box["x"] * image_width - x) / max(image_width, 1)
            + 0.3 * abs(box["y"] * image_height - y) / max(image_height, 1)
        )

    output = []
    count_agreement = (
        min(len(tuplet_boxes), len(groups))
        / max(len(tuplet_boxes), len(groups), 1)
    )
    for box, group, pair_cost, margin, mutual in _mutual_geometry_pairs(
        tuplet_boxes, groups, cost
    ):
        confidence = min(0.99, max(0.20, math.exp(-3.0 * pair_cost)))
        start, end = group["anchors"][0], group["anchors"][-1]
        start["members"] = group["members"]
        end["members"] = []
        semantic_complete = all(
            member.get("note_id") is not None
            and member.get("bps_time") is not None
            for member in group["members"]
        )
        row = _row_from_anchors(
                box,
                start,
                end,
                match_source="musicxml_tuplet_group",
                confidence=confidence,
                status=(
                    "matched"
                    if mutual
                    and margin >= 0.05
                    and semantic_complete
                    and confidence >= auto_accept_threshold(box["class"], 0.60)
                    else "review"
                ),
                xml_symbol=f"tuplet{group['actual_notes']}",
                geometry_score=confidence,
                candidate_margin=margin,
                count_agreement=count_agreement,
                xml_time_confirmed=semantic_complete,
            )
        output.append(
            row
        )
    return output


def _is_geometric_span_class(class_name: str) -> bool:
    return (
        class_name in GEOMETRIC_SPAN_CLASSES
        or class_name == "ottavaBracket"
        or class_name.startswith("tuplet")
        or class_name.startswith("dynamicCrescendo")
        or class_name.startswith("dynamicDiminuendo")
        or class_name in {"keyboardPedalPed", "ornamentWiggleTrill"}
    )


def estimate_all_symbol_times(
    boxes: list[dict],
    xml_notes: list[dict],
    xml_rests: list[dict],
    systems: list[SystemGeometry],
    image_width: int,
    image_height: int,
    measure_x_maps: dict[tuple[int, int], dict] | None = None,
) -> list[dict]:
    """Give every box a conservative geometry-derived musical-time candidate."""

    systems_by_number = {system.number: system for system in systems}
    anchors = [
        _anchor(note, systems_by_number, measure_x_maps) for note in xml_notes
    ] + [
        _anchor(rest, systems_by_number, measure_x_maps) for rest in xml_rests
    ]
    rows = []
    for box in boxes:
        system_number = assign_system(box, systems, image_height)
        candidates = [
            anchor
            for anchor in anchors
            if anchor["event"]["system"] == system_number
        ]
        if not candidates:
            row = _base_output_row(box, system_number)
            row.update(
                {
                    "start_meas": "",
                    "end_meas": "",
                    "start_note": "",
                    "end_note": "",
                    "connected_note": "",
                    "xml_measure": "",
                    "xml_symbol": "",
                    "xml_staff": "",
                    "match_source": "no_musicxml_anchor",
                    "status": "unresolved",
                }
            )
            rows.append(row)
            continue
        bx = box["x"] * image_width
        by = box["y"] * image_height

        def distance(anchor: dict, x: float) -> float:
            return abs(anchor["x"] - x) + 0.18 * abs(anchor["y"] - by)

        if _is_geometric_span_class(box["class"]):
            start_x = (box["x"] - box["w"] / 2) * image_width
            end_x = (box["x"] + box["w"] / 2) * image_width
            start = min(candidates, key=lambda anchor: distance(anchor, start_x))
            end = min(candidates, key=lambda anchor: distance(anchor, end_x))
            if end["event"]["bps_time"] < start["event"]["bps_time"]:
                start, end = end, start
            source = "geometric_span_time_estimate"
        else:
            start = min(candidates, key=lambda anchor: distance(anchor, bx))
            end = start
            source = "geometric_nearest_anchor_time_estimate"
        nearest_distance = distance(start, bx) / max(image_width, 1)
        confidence = min(0.49, max(0.10, 0.49 * math.exp(-4 * nearest_distance)))
        row = _row_from_anchors(
            box,
            start,
            end,
            match_source=source,
            confidence=confidence,
            status="review",
            xml_symbol=start["event"].get("pitch_name", "rest"),
            geometry_score=confidence,
            xml_time_confirmed=False,
        )
        rows.append(row)
    return rows


def unresolved_fingering_rows(
    boxes: list[dict],
    systems: list[SystemGeometry],
    image_height: int,
) -> list[dict]:
    """Keep known fingering classes but leave unknown semantic links blank."""

    output = []
    for box in boxes:
        if box["class"] not in FINGERING_CLASS_NAMES:
            continue
        system_number = assign_system(box, systems, image_height)
        row = _base_output_row(box, system_number)
        row.update(
            {
                # These fields apply to fingering, but the source MusicXML has
                # no fingering elements.  Blank means unknown; it is different
                # from NA, which means the field does not apply.
                "start_meas": "",
                "end_meas": "",
                "start_note": "",
                "end_note": "",
                "connected_note": "",
                "xml_measure": "",
                "xml_symbol": "",
                "xml_staff": "",
                "match_source": "",
                "confidence": "",
                "status": "unresolved",
                "target_x_px": "",
                "target_y_px": "",
            }
        )
        output.append(row)
    return output


def conservative_all_symbol_rows(
    boxes: list[dict],
    systems: list[SystemGeometry],
    image_height: int,
) -> list[dict]:
    """Create documented BPS-OMR rows without guessing semantic links."""

    output = []
    for box in boxes:
        if not box["class"]:
            raise ValueError(
                f"Missing class name for class_id {box['class_id']}"
            )

        system_number = assign_system(box, systems, image_height)
        row = _base_output_row(box, system_number)
        class_name = box["class"]

        row.update(
            {
                "musical_time": "",
                "start_meas": "",
                "end_meas": "",
                "start_note": "",
                "end_note": "",
                "connected_note": "",
                "stem_dir": "NA",
                "xml_measure": "",
                "xml_symbol": "",
                "xml_staff": "",
                "match_source": "",
                "confidence": "",
                "status": "unresolved",
                "target_x_px": "",
                "target_y_px": "",
            }
        )

        timeline_flag = musical_time_for_class(class_name)
        row["musical_time"] = "" if timeline_flag is None else timeline_flag

        # BPS-OMR examples explicitly use NA note links for dynamics and
        # timeline-independent terms/tempos.
        if (
            class_name.startswith("dynamic")
            or class_name.startswith("tempo")
            or class_name.startswith("term")
        ):
            row["start_note"] = "NA"
            row["end_note"] = "NA"
            row["connected_note"] = "NA"

        output.append(row)

    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_detailed_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=DETAILED_OUTPUT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_overlay(
    image: Image.Image,
    rows: list[dict],
    mode: str,
) -> Image.Image:
    """Compatibility wrapper around the isolated overlay renderer."""

    return render_alignment_overlay(image, rows, mode, DYNAMIC_CLASS_NAMES)


def draw_overlay(
    image: Image.Image,
    rows: list[dict],
    output_path: Path,
    mode: str,
) -> None:
    """Compatibility wrapper that writes one alignment overlay."""

    write_alignment_overlay(
        image, rows, output_path, mode, DYNAMIC_CLASS_NAMES
    )

def validate_dynamicf_ground_truth(
    rows: list[dict],
    ground_truth_path: Path | None,
) -> dict:
    if ground_truth_path is None or not ground_truth_path.exists():
        return {"available": False}

    expected = {}
    with ground_truth_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            expected[int(row["txt_line"])] = {
                "xml_measure": int(row["xml_measure"]),
                "bps_time": round(float(row["bps_time"]), 3),
            }

    actual = {
        int(row["txt_line"]): {
            "xml_measure": int(row["xml_measure"]),
            "bps_time": round(float(row["start_meas"]), 3),
        }
        for row in rows
        if int(row["class_id"]) == 18 and row["status"] != "unmatched"
    }

    mismatches = []
    for line_number in sorted(set(expected) | set(actual)):
        if expected.get(line_number) != actual.get(line_number):
            mismatches.append(
                {
                    "txt_line": line_number,
                    "expected": expected.get(line_number),
                    "actual": actual.get(line_number),
                }
            )

    return {
        "available": True,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def run_alignment(
    image_path: Path,
    yolo_path: Path,
    xml_path: Path,
    bps_note_path: Path,
    output_dir: Path,
    page_number: int = 1,
    dynamicf_ground_truth: Path | None = None,
    infer_fingerings: bool = False,
    notes_json_path: Path | None = None,
    include_all_symbols: bool = False,
    repeat_mapping_path: Path | None = None,
    clean_image_path: Path | None = None,
    system_start_measures: list[int] | None = None,
    page_end_measure: int | None = None,
    whole_score_spans: list[dict] | None = None,
    render_qa_images: bool = True,
    render_class_overlays: bool = True,
    render_auxiliary_overlays: bool = True,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    systems = detect_systems(image)
    categories = (
        load_categories(notes_json_path)
        if notes_json_path is not None
        else None
    )
    if include_all_symbols and categories is None:
        raise ValueError("--all-symbols requires --notes-json")
    boxes = load_yolo(yolo_path, categories=categories)
    target_boxes = boxes if include_all_symbols else [
        box
        for box in boxes
        if (
            box["class"] in DYNAMIC_CLASS_NAMES | FINGERING_CLASS_NAMES
            if categories is not None
            else box["class_id"] in TARGET_CLASSES
        )
    ]
    if system_start_measures and len(system_start_measures) != len(systems):
        raise ValueError(
            "The scan has "
            f"{len(systems)} detected systems, but "
            f"{len(system_start_measures)} printed-measure anchors were supplied"
        )
    xml_page = parse_musicxml_page(
        xml_path,
        page_number=page_number,
        system_start_measures=system_start_measures,
        page_end_measure=page_end_measure,
    )
    clean_reference_requested = clean_image_path is not None
    clean_reference_used = False
    note_x_overrides: dict[int, float] = {}
    if clean_image_path is not None:
        clean_image = Image.open(clean_image_path).convert("RGB")
        measure_x_maps, note_x_overrides, measure_x_diagnostics = (
            build_clean_reference_geometry(image, systems, clean_image, xml_page)
        )
        clean_reference_used = bool(measure_x_maps)
    else:
        measure_x_maps = {}
        measure_x_diagnostics = []
    if not measure_x_maps:
        fallback_maps, fallback_diagnostics = build_measure_local_x_maps(
            image, systems, xml_page
        )
        measure_x_maps = fallback_maps
        measure_x_diagnostics = [
            *measure_x_diagnostics,
            *fallback_diagnostics,
        ]
    bps_notes = load_bps_notes(bps_note_path)
    if repeat_mapping_path is not None:
        with repeat_mapping_path.open(newline="", encoding="utf-8") as file:
            repeat_rows = list(csv.DictReader(file))
        expanded_notes = attach_repeat_occurrences(
            xml_page["notes"], repeat_rows, bps_notes
        )
        attach_timeline_repeat_occurrences(xml_page["dynamics"], repeat_rows)
        attach_timeline_repeat_occurrences(xml_page["rests"], repeat_rows)
        attach_timeline_repeat_occurrences(
            xml_page["text_directions"], repeat_rows
        )
        attach_timeline_repeat_occurrences(
            xml_page["direction_markers"], repeat_rows
        )
    else:
        attach_bps_note_ids(xml_page["notes"], bps_notes)
        expanded_notes = xml_page["notes"]

    dynamic_rows, unused_xml = match_dynamics(
        target_boxes,
        xml_page["dynamics"],
        systems,
        image.height,
    )
    if include_all_symbols:
        rows_by_line = {
            int(row["txt_line"]): row
            for row in estimate_all_symbol_times(
                target_boxes,
                xml_page["notes"],
                xml_page["rests"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            )
        }
        for row in dynamic_rows:
            if row["status"] != "unmatched":
                rows_by_line[int(row["txt_line"])] = row
        for row in match_point_notations(
            target_boxes,
            xml_page["notes"],
            xml_page["rests"],
            systems,
            image.width,
            image.height,
            measure_x_maps=measure_x_maps,
        ):
            rows_by_line[int(row["txt_line"])] = row
        for row in match_xml_spans(
            target_boxes,
            xml_page["notes"],
            bps_notes,
            systems,
            image.width,
            image.height,
            measure_x_maps=measure_x_maps,
            note_x_overrides=note_x_overrides,
        ):
            rows_by_line[int(row["txt_line"])] = row
        for row in match_tuplets(
            target_boxes,
            xml_page["notes"],
            systems,
            image.width,
            image.height,
            measure_x_maps=measure_x_maps,
        ):
            rows_by_line[int(row["txt_line"])] = row
        for matcher_rows in (
            match_small_note_symbols(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_small_stems(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_small_beams(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_small_flags(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_direction_symbols(
                target_boxes,
                xml_page["notes"],
                xml_page["direction_markers"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_text_directions(
                target_boxes,
                xml_page["text_directions"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
            match_wavy_line_spans(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ),
        ):
            for row in matcher_rows:
                rows_by_line[int(row["txt_line"])] = row
        if whole_score_spans:
            for row in match_cross_page_spans(
                target_boxes,
                xml_page["notes"],
                whole_score_spans,
                page_number,
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            ):
                line = int(row["txt_line"])
                if rows_by_line[line].get("status") != "matched":
                    rows_by_line[line] = row
        if infer_fingerings:
            fingering_rows = match_fingerings(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            )
            for row in fingering_rows:
                if row["status"] != "unmatched":
                    rows_by_line[int(row["txt_line"])] = row
        rows = sorted(
            rows_by_line.values(),
            key=lambda row: int(row["txt_line"]),
        )
    else:
        if infer_fingerings:
            fingering_rows = match_fingerings(
                target_boxes,
                xml_page["notes"],
                systems,
                image.width,
                image.height,
                measure_x_maps=measure_x_maps,
            )
        else:
            fingering_rows = unresolved_fingering_rows(
                target_boxes,
                systems,
                image.height,
            )
        rows = sorted(
            dynamic_rows + fingering_rows,
            key=lambda row: int(row["txt_line"]),
        )

    finalize_match_diagnostics(rows)
    attach_review_note_candidates(
        rows,
        xml_page["notes"],
        systems,
        image.width,
        image.height,
        measure_x_maps=measure_x_maps,
    )

    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    page_stem = image_path.stem
    csv_filename = (
        f"{page_stem}_bps_omr_all_symbols.csv"
        if include_all_symbols
        else f"{page_stem}_bps_omr.csv"
    )
    csv_path = output_dir / csv_filename
    detailed_csv_path = output_dir / f"{page_stem}_alignment_detailed.csv"
    dynamics_overlay = qa_dir / f"{page_stem}_dynamics.png"
    fingering_overlay = qa_dir / f"{page_stem}_fingerings.png"
    all_symbols_overlay = qa_dir / f"{page_stem}_all_symbols.png"
    review_overlay = qa_dir / f"{page_stem}_needs_review.png"
    report_path = qa_dir / f"{page_stem}_report.json"

    from bpsd_aligner.review_candidates import normalize_review_candidates

    review_candidates_path, review_candidate_sets_path = normalize_review_candidates(
        rows,
        page_id=page_stem,
        output_dir=output_dir,
    )

    write_csv(csv_path, rows)
    write_detailed_csv(detailed_csv_path, rows)
    if render_qa_images and render_auxiliary_overlays:
        draw_overlay(image, rows, dynamics_overlay, mode="dynamics")
        draw_overlay(image, rows, fingering_overlay, mode="fingerings")
    if include_all_symbols and render_qa_images:
        if render_auxiliary_overlays:
            draw_overlay(image, rows, all_symbols_overlay, mode="all")
        draw_overlay(
            image,
            [row for row in rows if row["status"] != "matched"],
            review_overlay,
            mode="all",
        )

    class_overlays = {}
    if include_all_symbols and render_qa_images and render_class_overlays:
        rows_by_class = defaultdict(list)
        for row in rows:
            rows_by_class[row["class"]].append(row)
        used_slugs = set()
        for class_name, class_rows in sorted(rows_by_class.items()):
            base_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", class_name).strip("-._")
            base_slug = base_slug or "unknown"
            slug = base_slug
            suffix = 2
            while slug in used_slugs:
                slug = f"{base_slug}-{suffix}"
                suffix += 1
            used_slugs.add(slug)
            overlay_path = qa_dir / f"{page_stem}_class_{slug}.png"
            draw_overlay(image, class_rows, overlay_path, mode="class")
            class_overlays[f"class_{slug}_overlay"] = str(overlay_path)

    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["class"]][row["status"]] += 1

    report = {
        "page": page_number,
        "detected_systems": len(systems),
        "target_yolo_boxes": len(target_boxes),
        "output_rows": len(rows),
        "counts": {
            class_name: dict(status_counts)
            for class_name, status_counts in sorted(counts.items())
        },
        "unused_musicxml_dynamic_events": [
            {
                "class": event["class"],
                "system": event["system"],
                "xml_measure": event["xml_measure"],
                "xml_symbol": event["xml_symbol"],
                "bps_time": round(event["bps_time"], 3),
            }
            for event in unused_xml
        ],
        "dynamicf_ground_truth": validate_dynamicf_ground_truth(
            rows,
            dynamicf_ground_truth,
        ),
        "musicxml_page_notes": len(xml_page["notes"]),
        "musicxml_text_directions": len(xml_page["text_directions"]),
        "musicxml_direction_markers": len(xml_page["direction_markers"]),
        "whole_score_cross_page_span_candidates": sum(
            row.get("match_source") == "whole_score_cross_page_span_candidate"
            for row in rows
        ),
        "score_layout": {
            "source": xml_page["layout_source"],
            "system_start_measures": xml_page["system_start_measures"],
            "page_end_measure": xml_page["page_end_measure"],
        },
        "musicxml_page_notes_with_bps_id": sum(
            note.get("note_id") is not None for note in xml_page["notes"]
        ),
        "unfolded_note_occurrences": len(expanded_notes),
        "repeat_mapping": str(repeat_mapping_path or ""),
        "measure_local_x_mapping": {
            "mapped_measures": len(measure_x_maps),
            "diagnostics": measure_x_diagnostics,
        },
        "clean_reference": {
            "requested": clean_reference_requested,
            "used": clean_reference_used,
            "snapped_noteheads": len(note_x_overrides),
        },
        "fingering_mode": (
            "inferred_candidates"
            if infer_fingerings
            else "strict_blank_unknowns"
        ),
        "include_all_symbols": include_all_symbols,
        "class_overlay_count": len(class_overlays),
        "class_overlay_available_count": len(counts),
        "limitations": (
            [
                "Direct XML notation matching covers dynamics, words, wedges, staccato, fermata, slur, tie, ornaments and wavy-line spans, tuplets, grace-note small components and flags, ottava, and pedal endpoints.",
                "Classes without direct XML evidence receive geometry-derived time candidates and status=review.",
                "Only rows with no usable MusicXML anchor remain unresolved with blank times.",
                (
                    "Repeat occurrences are attached from the verified mapping."
                    if repeat_mapping_path is not None
                    else "Repeat mapping was not supplied for this run."
                ),
            ]
            if include_all_symbols
            else [
                "The source MusicXML contains no fingering elements.",
                "Fingering note links are inferred from scan geometry and BPSD notes.",
                "Rows with status=review require manual verification.",
            ]
            if infer_fingerings
            else [
                "The source MusicXML contains no fingering elements.",
                "Unknown fingering time and note-link fields are intentionally blank.",
            ]
        ),
        "outputs": {
            "csv": str(csv_path),
            "detailed_csv": str(detailed_csv_path),
            "review_note_candidates_csv": str(review_candidates_path),
            "review_candidate_sets_csv": str(review_candidate_sets_path),
            "dynamics_overlay": (
                str(dynamics_overlay)
                if render_qa_images and render_auxiliary_overlays
                else None
            ),
            "fingering_overlay": (
                str(fingering_overlay)
                if render_qa_images and render_auxiliary_overlays
                else None
            ),
            "all_symbols_overlay": (
                str(all_symbols_overlay)
                if include_all_symbols
                and render_qa_images
                and render_auxiliary_overlays
                else None
            ),
            "review_overlay": (
                str(review_overlay)
                if include_all_symbols and render_qa_images
                else None
            ),
            **class_overlays,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align selected YOLO glyphs with BPSD MusicXML/notes."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path)
    parser.add_argument("--repeat-mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument(
        "--system-start-measures",
        help=(
            "Comma-separated printed first measure for every scanned system; "
            "for example 198,202,206,211,223,235."
        ),
    )
    parser.add_argument(
        "--page-end-measure",
        type=int,
        help="Printed final measure on the scanned page.",
    )
    parser.add_argument("--dynamicf-ground-truth", type=Path)
    parser.add_argument(
        "--infer-fingerings",
        action="store_true",
        help=(
            "Generate non-authoritative fingering note candidates. "
            "The default leaves unknown fingering fields blank."
        ),
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help=(
            "Include every YOLO row. Unknown BPS-OMR semantic fields "
            "remain blank."
        ),
    )
    parser.add_argument(
        "--no-qa-images",
        action="store_true",
        help="Skip review overlays for lightweight regression or CI runs.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    system_start_measures = (
        [int(value.strip()) for value in args.system_start_measures.split(",")]
        if args.system_start_measures
        else None
    )
    report = run_alignment(
        image_path=args.image,
        yolo_path=args.yolo,
        xml_path=args.xml,
        bps_note_path=args.bps_notes,
        output_dir=args.output_dir,
        page_number=args.page,
        dynamicf_ground_truth=args.dynamicf_ground_truth,
        infer_fingerings=args.infer_fingerings,
        notes_json_path=args.notes_json,
        include_all_symbols=args.all_symbols,
        repeat_mapping_path=args.repeat_mapping,
        system_start_measures=system_start_measures,
        page_end_measure=args.page_end_measure,
        render_qa_images=not args.no_qa_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
