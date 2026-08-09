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
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DYNAMIC_CLASS_BY_GLYPH = {
    "f": (18, "dynamicF"),
    "p": (20, "dynamicP"),
    "s": (21, "dynamicS"),
}

FINGERING_CLASSES = {
    25: "fingering1",
    26: "fingering2",
    27: "fingering3",
    28: "fingering4",
    29: "fingering5",
}

DYNAMIC_CLASS_NAMES = {"dynamicF", "dynamicP", "dynamicS"}
FINGERING_CLASS_NAMES = {f"fingering{digit}" for digit in range(1, 6)}
FINGERING_DY_WEIGHT = 0.18
FINGERING_WRONG_STAFF_PENALTY_RATIO = 0.02
FINGERING_STRICT_STAFF = False
FINGERING_AUTO_ACCEPT_THRESHOLD = 0.95

TARGET_CLASSES = {
    **{class_id: name for class_id, name in DYNAMIC_CLASS_BY_GLYPH.values()},
    **FINGERING_CLASSES,
}

OUTPUT_FIELDS = [
    "class_id",
    "x",
    "y",
    "w",
    "h",
    "class",
    "musical_time",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
]

DETAILED_OUTPUT_FIELDS = [
    *OUTPUT_FIELDS,
    "txt_line",
    "system",
    "xml_measure",
    "xml_symbol",
    "xml_staff",
    "target_type",
    "note_ids",
    "pitches",
    "repeat_occurrences_json",
    "repeat_occurrence_count",
    "repeat_group_id",
    "match_source",
    "confidence",
    "status",
    "target_x_px",
    "target_y_px",
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
    "status",
]


@dataclass
class StaffGeometry:
    center: float
    line_spacing: float
    lines: list[float]


@dataclass
class SystemGeometry:
    number: int
    upper: StaffGeometry
    lower: StaffGeometry
    x_left: float
    x_right: float

    @property
    def center(self) -> float:
        return (self.upper.center + self.lower.center) / 2


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def _consecutive_groups(values: Iterable[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _staff_pattern_candidates(
    row_ink: np.ndarray,
    crop_width: int,
) -> list[list[float]]:
    """Find non-overlapping five-line patterns in a horizontal ink profile.

    Dense chords can make several neighboring rows look like long horizontal
    lines, while noteheads can interrupt a genuine staff line.  Scoring a
    complete, approximately equally spaced five-line pattern is therefore more
    stable than thresholding each row independently.
    """

    candidates = []
    height = len(row_ink)
    # Scale the minimum spacing with page height: the rendered reference score
    # is smaller than the scan, while seven-pixel repeated strokes in the
    # larger Op90 scan come from dense chord beams rather than staff lines.
    minimum_spacing = max(7, round(height * 0.0043))
    for spacing_twice in range(2 * minimum_spacing, 49):
        spacing = spacing_twice / 2
        final_offset = round(4 * spacing)
        for top in range(2, height - final_offset - 2):
            lines = []
            strengths = []
            for line_index in range(5):
                predicted = round(top + line_index * spacing)
                search = range(predicted - 2, predicted + 3)
                row = max(search, key=lambda y: row_ink[y])
                lines.append(float(row))
                strengths.append(float(row_ink[row]) / crop_width)

            average_strength = sum(strengths) / 5
            minimum_strength = min(strengths)
            if average_strength < 0.38 or minimum_strength < 0.25:
                continue

            # Prefer strong complete patterns, then patterns whose selected
            # rows stay closest to the proposed equal spacing.
            spacing_error = sum(
                abs((lines[index] - lines[index - 1]) - spacing)
                for index in range(1, 5)
            ) / 4
            score = average_strength + 0.15 * minimum_strength - 0.02 * spacing_error
            candidates.append((score, spacing, lines))

    selected = []
    for score, spacing, lines in sorted(candidates, reverse=True):
        top = min(lines)
        bottom = max(lines)
        overlaps = any(
            not (
                bottom < min(existing) - spacing
                or top > max(existing) + spacing
            )
            for existing in selected
        )
        if not overlaps:
            selected.append(lines)

    return sorted(selected, key=lambda lines: sum(lines) / len(lines))


def detect_systems(image: Image.Image) -> list[SystemGeometry]:
    """Detect paired piano systems from five-line horizontal patterns."""

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    x0 = round(width * 0.05)
    x1 = round(width * 0.95)
    crop_width = x1 - x0

    row_ink = (gray[:, x0:x1] < 170).sum(axis=1)
    raw_staff_groups = _staff_pattern_candidates(row_ink, crop_width)
    strict_staff_groups = [
        group
        for group in raw_staff_groups
        if min(
            _longest_true_run(gray[round(line), x0:x1] < 190)
            for line in group
        )
        >= crop_width * 0.20
    ]
    # Most pages are best served by the long-line filter because it rejects
    # beams and footer text.  A scan can, however, contain a genuine staff line
    # broken by damage or dense notation.  If strict filtering leaves an
    # impossible odd count, fall back only when the complete five-line pattern
    # set itself forms valid piano pairs.  This preserves the strict result on
    # normal pages while recovering damaged staves such as Op110 p6 and Op111
    # p1.
    if len(strict_staff_groups) >= 2 and len(strict_staff_groups) % 2 == 0:
        staff_groups = strict_staff_groups
    elif len(raw_staff_groups) >= 2 and len(raw_staff_groups) % 2 == 0:
        staff_groups = raw_staff_groups
    else:
        staff_groups = strict_staff_groups
    if len(staff_groups) % 2 != 0 or len(staff_groups) < 2:
        raise ValueError(
            "Could not detect paired five-line staves "
            f"(strict={len(strict_staff_groups)}, raw={len(raw_staff_groups)})"
        )

    systems = []
    for system_index in range(0, len(staff_groups), 2):
        upper_lines = staff_groups[system_index]
        lower_lines = staff_groups[system_index + 1]
        all_lines = upper_lines + lower_lines

        votes = np.zeros(width, dtype=int)
        for center in all_lines:
            row = round(center)
            neighborhood = gray[max(0, row - 1) : min(height, row + 2), :]
            votes += (neighborhood < 190).any(axis=0)

        candidate_columns = votes >= 6
        smoothed = np.convolve(
            candidate_columns.astype(int),
            np.ones(31, dtype=int),
            mode="same",
        )
        long_staff_columns = np.where(smoothed >= 20)[0]
        if not len(long_staff_columns):
            raise ValueError(f"Could not detect x range for system {system_index // 2 + 1}")

        upper_spacing = median(
            b - a for a, b in zip(upper_lines, upper_lines[1:])
        )
        lower_spacing = median(
            b - a for a, b in zip(lower_lines, lower_lines[1:])
        )

        systems.append(
            SystemGeometry(
                number=system_index // 2 + 1,
                upper=StaffGeometry(
                    center=sum(upper_lines) / 5,
                    line_spacing=float(upper_spacing),
                    lines=upper_lines,
                ),
                lower=StaffGeometry(
                    center=sum(lower_lines) / 5,
                    line_spacing=float(lower_spacing),
                    lines=lower_lines,
                ),
                x_left=float(long_staff_columns[0]),
                x_right=float(long_staff_columns[-1]),
            )
        )

    return systems


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def detect_barlines(
    image: Image.Image,
    systems: list[SystemGeometry],
    expected_boundary_counts: list[int],
) -> list[list[int]]:
    """Detect piano-system barlines using continuous vertical ink.

    Note stems and dense chords can have a high total ink count, but unlike a
    barline they do not form one continuous line from the upper staff to the
    lower staff.  System-edge estimates from horizontal staff lines are
    refined by searching for the actual vertical start/end barline nearby.
    """

    if len(systems) != len(expected_boundary_counts):
        raise ValueError(
            "One expected boundary count is required for each system"
        )

    gray = np.asarray(image.convert("L"))
    output = []
    for system, expected_count in zip(systems, expected_boundary_counts):
        y0 = round(system.upper.lines[0])
        y1 = round(system.lower.lines[-1])
        x0 = round(system.x_left)
        x1 = round(system.x_right)
        band_height = y1 - y0 + 1

        longest_runs = []
        for x in range(x0, x1 + 1):
            dark = gray[y0 : y1 + 1, x] < 170
            longest_runs.append(_longest_true_run(dark))

        candidate_columns = [
            x0 + index
            for index, run_length in enumerate(longest_runs)
            if run_length >= band_height * 0.85
        ]
        groups = _consecutive_groups(candidate_columns)
        boundaries = [
            round((group[0] + group[-1]) / 2)
            for group in groups
        ]

        edge_window = max(20, round((x1 - x0) * 0.04))
        refined_edges = []
        for estimated_edge in (x0, x1):
            search_left = max(0, estimated_edge - edge_window)
            search_right = min(
                gray.shape[1] - 1,
                estimated_edge + edge_window,
            )
            scored = []
            for x in range(search_left, search_right + 1):
                dark = gray[y0 : y1 + 1, x] < 170
                longest_run = _longest_true_run(dark)
                ink_count = int(dark.sum())
                score = (
                    2 * longest_run
                    + ink_count
                    - 0.4 * abs(x - estimated_edge)
                )
                scored.append((score, x))
            refined_edges.append(max(scored)[1])

        left_edge, right_edge = refined_edges
        edge_tolerance = 8
        interior_boundaries = [
            boundary
            for boundary in boundaries
            if boundary > left_edge + edge_tolerance
            and boundary < right_edge - edge_tolerance
        ]
        boundaries = [
            left_edge,
            *interior_boundaries,
            right_edge,
        ]

        if len(boundaries) != expected_count:
            raise ValueError(
                f"System {system.number}: expected {expected_count} "
                f"measure boundaries, detected {len(boundaries)} "
                f"({boundaries})"
            )
        output.append(boundaries)

    return output


def align_barlines_from_reference(
    image: Image.Image,
    systems: list[SystemGeometry],
    reference_systems: list[SystemGeometry],
    reference_boundaries: list[list[int]],
) -> list[list[dict]]:
    """Find scan barlines near normalized boundaries from a clean score.

    The reference supplies only a search position.  The selected x coordinate
    still comes from vertical ink in the target scan.  Low-continuity lines are
    retained but explicitly marked for visual review because printed symbols
    can occlude an otherwise genuine barline.
    """

    if not (
        len(systems)
        == len(reference_systems)
        == len(reference_boundaries)
    ):
        raise ValueError("Target and reference must have the same system count")

    gray = np.asarray(image.convert("L"))
    aligned = []
    for system, reference_system, boundaries in zip(
        systems,
        reference_systems,
        reference_boundaries,
    ):
        predicted = [
            system.x_left
            + (
                (boundary - reference_system.x_left)
                / (reference_system.x_right - reference_system.x_left)
            )
            * (system.x_right - system.x_left)
            for boundary in boundaries
        ]

        y0 = round(system.upper.lines[0])
        y1 = round(system.lower.lines[-1])
        band_height = y1 - y0 + 1
        system_output = []
        for index, predicted_x in enumerate(predicted):
            if index in {0, len(predicted) - 1}:
                edge_window = max(
                    20,
                    round((system.x_right - system.x_left) * 0.04),
                )
                search_left = max(0, round(predicted_x) - edge_window)
                search_right = min(
                    gray.shape[1] - 1,
                    round(predicted_x) + edge_window,
                )
            else:
                search_left = round((predicted[index - 1] + predicted_x) / 2)
                search_right = round((predicted_x + predicted[index + 1]) / 2)

            scored = []
            for x in range(search_left, search_right + 1):
                dark = gray[y0 : y1 + 1, x] < 170
                longest_run = _longest_true_run(dark)
                ink_count = int(dark.sum())
                score = (
                    2 * longest_run
                    + ink_count
                    - 0.4 * abs(x - predicted_x)
                )
                scored.append((score, x, longest_run, ink_count))

            _score, x, longest_run, ink_count = max(scored)
            vertical_coverage = longest_run / band_height
            ink_coverage = ink_count / band_height
            status = (
                "system_edge"
                if index in {0, len(predicted) - 1}
                else (
                    "detected"
                    if vertical_coverage >= 0.85
                    else "review_occluded"
                )
            )
            system_output.append(
                {
                    "x": x,
                    "predicted_x": predicted_x,
                    "vertical_coverage": vertical_coverage,
                    "ink_coverage": ink_coverage,
                    "status": status,
                }
            )
        aligned.append(system_output)

    return aligned


def assign_system(box: dict, systems: list[SystemGeometry], image_height: int) -> int:
    y_pixel = box["y"] * image_height
    return min(systems, key=lambda system: abs(system.center - y_pixel)).number


def _pitch_midi(pitch: ET.Element) -> tuple[int, str, int]:
    step = (pitch.findtext("step") or "C").strip()
    alter = _int_text(pitch.find("alter"), 0)
    octave = _int_text(pitch.find("octave"), 4)
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


def parse_musicxml_page(xml_path: Path, page_number: int = 1) -> dict:
    """Parse layout, dynamics and note events for one MusicXML page."""

    root = ET.parse(xml_path).getroot()
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
            if attributes.find("divisions") is not None:
                divisions = _int_text(attributes.find("divisions"), divisions)
            time_element = attributes.find("time")
            if time_element is not None:
                beats = _int_text(time_element.find("beats"), beats)
                beat_type = _int_text(time_element.find("beat-type"), beat_type)
        nominal_duration = divisions * beats * 4 / beat_type
        cursor = 0.0
        max_cursor = 0.0
        last_note_onset = 0.0
        last_note_x = 0.0
        notes = []
        rests = []
        dynamics = []
        clef_events: dict[int, list[tuple[float, dict]]] = defaultdict(list)

        for child in measure:
            name = _local_name(child.tag)

            if name == "attributes":
                for clef in child.findall("clef"):
                    staff_number = int(clef.attrib.get("number", "1"))
                    clef_value = {
                        "sign": clef.findtext("sign") or "G",
                        "line": _int_text(clef.find("line"), 2),
                    }
                    clefs[staff_number] = clef_value
                    clef_events[staff_number].append(
                        (cursor, dict(clef_value))
                    )

            elif name == "backup":
                cursor -= _float_text(child.find("duration"))

            elif name == "forward":
                cursor += _float_text(child.find("duration"))
                max_cursor = max(max_cursor, cursor)

            elif name == "direction":
                offset = _float_text(child.find("offset"))
                onset = cursor + offset
                staff = _int_text(child.find("staff"), 1)
                direction_type = child.find("direction-type")
                if direction_type is None:
                    continue
                dynamics_element = direction_type.find("dynamics")
                if dynamics_element is None:
                    continue
                default_x = float(dynamics_element.attrib.get("default-x", "0"))
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

            elif name == "note":
                is_chord = child.find("chord") is not None
                if not is_chord:
                    current_chord_sequence = xml_chord_sequence
                    xml_chord_sequence += 1
                duration = _float_text(child.find("duration"))
                onset = last_note_onset if is_chord else cursor
                if not is_chord:
                    last_note_onset = onset
                default_x = float(child.attrib.get("default-x", last_note_x))
                if not is_chord:
                    last_note_x = default_x
                staff = _int_text(child.find("staff"), 1)
                voice = (child.findtext("voice") or "1").strip()
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
                pitch = child.find("pitch")
                if pitch is not None:
                    midi, pitch_name, diatonic = _pitch_midi(pitch)
                    slur_marks = []
                    tie_marks = []
                    articulation_marks = []
                    fermata_marks = []
                    tuplet_marks = []
                    notations = child.find("notations")
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
                    time_modification = child.find("time-modification")
                    actual_notes = (
                        _int_text(time_modification.find("actual-notes"), 0)
                        if time_modification is not None
                        else 0
                    )
                    normal_notes = (
                        _int_text(time_modification.find("normal-notes"), 0)
                        if time_modification is not None
                        else 0
                    )
                    if not tie_marks:
                        tie_marks = [
                            {"type": tie.attrib.get("type", "")}
                            for tie in child.findall("tie")
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
                            "stem": (child.findtext("stem") or "").strip(),
                            "slur_marks": slur_marks,
                            "tie_marks": tie_marks,
                            "articulation_marks": articulation_marks,
                            "fermata_marks": fermata_marks,
                            "tuplet_marks": tuplet_marks,
                            "actual_notes": actual_notes,
                            "normal_notes": normal_notes,
                        }
                    )
                    xml_note_sequence += 1
                elif child.find("rest") is not None:
                    fermata_marks = []
                    notations = child.find("notations")
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
            }
        )

    page_measures = [
        measure for measure in raw_measures if measure["page"] == page_number
    ]
    if not page_measures:
        raise ValueError(f"MusicXML page {page_number} does not exist")
    first_measure = raw_measures[0]
    score_has_pickup = (
        first_measure["actual_duration"] < first_measure["nominal_duration"]
    )
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
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
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
                    "xml_measure_index": measure["measure_index"],
                    "timeline_offset": timeline_offset,
                    "measure_within": within,
                    "bps_time": (measure["measure"] - 1) + within,
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

    return {
        "measures": page_measures,
        "notes": notes,
        "rests": rests,
        "dynamics": dynamics,
        "system_widths": dict(system_widths),
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
    by_time_pitch: dict[tuple[float, int], deque[int]] = defaultdict(deque)
    for note in bps_notes:
        by_time_pitch[(round(note["bps_time"], 3), note["midi"])].append(
            note["note_id"]
        )

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
        note["note_id"] = (
            by_time_pitch[key].popleft() if by_time_pitch[key] else None
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
            and bps_note["bps_time"] <= note["bps_time"] <= bps_note["end_time"]
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
                "note_id": item.get("note_id"),
                "mapping_status": item["repeat_mapping_status"],
            }
            for item in occurrences
        ]
        first = occurrences[0]
        original["written_bps_time"] = first["written_bps_time"]
        original["bps_time"] = first["bps_time"]
        original["note_id"] = first.get("note_id")
        original["repeat_occurrence_count"] = len(occurrences)
        original["repeat_group_id"] = first["repeat_group_id"]

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
            }
            for row in mappings
        ]
        event["repeat_occurrences"] = occurrences
        if occurrences:
            event["written_bps_time"] = event["bps_time"]
            event["bps_time"] = occurrences[0]["bps_time"]
            event["repeat_occurrence_count"] = len(occurrences)
            event["repeat_group_id"] = occurrences[0]["repeat_group_id"]


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
        status = (
            "time_confirmed"
            if start_match in {"exact", "within_tied_span"}
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
        status = (
            "time_confirmed"
            if start_match in {"exact", "within_tied_span"}
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
    return {
        "class_id": box["class_id"],
        "x": f"{box['x']:.6f}",
        "y": f"{box['y']:.6f}",
        "w": f"{box['w']:.6f}",
        "h": f"{box['h']:.6f}",
        "class": box["class"],
        "musical_time": 0,
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
        "match_source": "NA",
        "confidence": "0.000",
        "status": "unmatched",
        "target_x_px": "NA",
        "target_y_px": "NA",
    }


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
                item["component_index"],
            ),
        )
        pair_count = min(len(yolo_items), len(xml_items))

        count_confidence = (
            1.0
            if len(yolo_items) == len(xml_items)
            else pair_count / max(len(yolo_items), len(xml_items), 1)
        )

        for box, event in zip(yolo_items[:pair_count], xml_items[:pair_count]):
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
                        event.get("repeat_occurrences", []),
                        ensure_ascii=False,
                    ),
                    "repeat_occurrence_count": event.get(
                        "repeat_occurrence_count", 1
                    ),
                    "repeat_group_id": event.get("repeat_group_id", ""),
                    "match_source": "musicxml_dynamic",
                    "confidence": f"{count_confidence:.3f}",
                    "status": (
                        "matched"
                        if count_confidence == 1.0
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
            if confidence >= FINGERING_AUTO_ACCEPT_THRESHOLD
            else "review"
        )

        used_note_ids.update(
            note_item[0]["note_id"] for _box, note_item in pairs
        )

        for box, note_item in pairs:
            note, note_x, note_y = note_item
            note_id = note["note_id"]
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
                    "start_note": note_id,
                    "end_note": note_id,
                    "connected_note": f"[{note_id}]",
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
                    "match_source": (
                        "grouped_nearest_musicxml_bps_note_ambiguous_chord"
                        if ambiguous_chord_member
                        else "grouped_nearest_musicxml_bps_note_staff_mismatch"
                        if staff_side_mismatch
                        else "grouped_nearest_musicxml_bps_note"
                    ),
                    "confidence": f"{confidence:.3f}",
                    "status": status,
                    "target_x_px": f"{note_x:.1f}",
                    "target_y_px": f"{note_y:.1f}",
                }
            )
            output.append(row)

    return output


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

        # These assignments are directly supported by BPS-OMR annotations.pdf:
        # dynamics/slurs/ties/tuplets/fingerings are on the musical timeline;
        # terms and tempos are outside it.  The document does not explicitly
        # classify articulation or fermata, so those flags remain blank.
        if (
            class_name.startswith("dynamic")
            or class_name.startswith("fingering")
            or class_name in {"slur", "tie"}
            or class_name.startswith("tuplet")
        ):
            row["musical_time"] = 0
        elif class_name.startswith("tempo") or class_name.startswith("term"):
            row["musical_time"] = 1

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


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image: Image.Image,
    rows: list[dict],
    output_path: Path,
    mode: str,
) -> None:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font = _load_font(15 if mode == "dynamics" else 12)

    for row in rows:
        is_dynamic = row["class"] in DYNAMIC_CLASS_NAMES
        is_fingering = row["class"].startswith("fingering")
        if mode == "dynamics" and not is_dynamic:
            continue
        if mode == "fingerings" and not is_fingering:
            continue

        x = float(row["x"])
        y = float(row["y"])
        box_width = float(row["w"])
        box_height = float(row["h"])
        rectangle = (
            round((x - box_width / 2) * width),
            round((y - box_height / 2) * height),
            round((x + box_width / 2) * width),
            round((y + box_height / 2) * height),
        )

        status = row["status"]
        color = {
            "matched": "green",
            "inferred": "blue",
            "review": "darkorange",
            "unresolved": "gray",
        }.get(status, "red")
        draw.rectangle(rectangle, outline=color, width=2)

        if mode == "all":
            label = row["class"]
        elif is_dynamic:
            label = (
                f"{row['class']} m{row['xml_measure']} "
                f"t={row['start_meas']}"
            )
        else:
            if status == "unresolved":
                label = f"{row['class'][-1]} unresolved"
            else:
                label = (
                    f"{row['class'][-1]} n={row['start_note']} "
                    f"t={row['start_meas']} c={row['confidence']}"
                )
            if (
                row["target_x_px"] not in {"", "NA"}
                and row["target_y_px"] not in {"", "NA"}
            ):
                box_center = (
                    round(x * width),
                    round(y * height),
                )
                note_center = (
                    round(float(row["target_x_px"])),
                    round(float(row["target_y_px"])),
                )
                draw.line(
                    (box_center, note_center),
                    fill=color,
                    width=1,
                )
                radius = 3
                draw.ellipse(
                    (
                        note_center[0] - radius,
                        note_center[1] - radius,
                        note_center[0] + radius,
                        note_center[1] + radius,
                    ),
                    outline=color,
                    width=1,
                )

        text_y = max(0, rectangle[1] - (18 if is_dynamic else 14))
        draw.text(
            (rectangle[0], text_y),
            label,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


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
    xml_page = parse_musicxml_page(xml_path, page_number=page_number)
    measure_x_maps, measure_x_diagnostics = build_measure_local_x_maps(
        image, systems, xml_page
    )
    bps_notes = load_bps_notes(bps_note_path)
    if repeat_mapping_path is not None:
        with repeat_mapping_path.open(newline="", encoding="utf-8") as file:
            repeat_rows = list(csv.DictReader(file))
        expanded_notes = attach_repeat_occurrences(
            xml_page["notes"], repeat_rows, bps_notes
        )
        attach_timeline_repeat_occurrences(xml_page["dynamics"], repeat_rows)
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
            for row in conservative_all_symbol_rows(
                target_boxes,
                systems,
                image.height,
            )
        }
        for row in dynamic_rows:
            rows_by_line[int(row["txt_line"])] = row
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

    qa_dir = output_dir / "qa"
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
    report_path = qa_dir / f"{page_stem}_report.json"

    write_csv(csv_path, rows)
    write_detailed_csv(detailed_csv_path, rows)
    draw_overlay(image, rows, dynamics_overlay, mode="dynamics")
    draw_overlay(image, rows, fingering_overlay, mode="fingerings")
    if include_all_symbols:
        draw_overlay(image, rows, all_symbols_overlay, mode="all")

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
        "musicxml_page_notes_with_bps_id": sum(
            note.get("note_id") is not None for note in xml_page["notes"]
        ),
        "unfolded_note_occurrences": len(expanded_notes),
        "repeat_mapping": str(repeat_mapping_path or ""),
        "measure_local_x_mapping": {
            "mapped_measures": len(measure_x_maps),
            "diagnostics": measure_x_diagnostics,
        },
        "fingering_mode": (
            "inferred_candidates"
            if infer_fingerings
            else "strict_blank_unknowns"
        ),
        "include_all_symbols": include_all_symbols,
        "limitations": (
            [
                "Only dynamicF/P/S currently have confirmed MusicXML times.",
                "All other uncertain semantic fields are intentionally blank.",
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
            "dynamics_overlay": str(dynamics_overlay),
            "fingering_overlay": str(fingering_overlay),
            "all_symbols_overlay": (
                str(all_symbols_overlay) if include_all_symbols else None
            ),
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
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
