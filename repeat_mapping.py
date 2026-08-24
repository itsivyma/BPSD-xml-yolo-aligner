"""Build a performance timeline from repetition-preserving MusicXML."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from bpsd_aligner.musicxml import (
    child as _shared_child,
    child_text as _shared_child_text,
    local_name as _shared_local_name,
    parse_musicxml,
)


def _local(tag: str) -> str:
    return _shared_local_name(tag)


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return _shared_child(element, name)


def _text(element: ET.Element, name: str, default: str = "") -> str:
    return _shared_child_text(element, name, default)


def _first_part_measures(path: Path) -> list[ET.Element]:
    root = parse_musicxml(path)
    part = next(
        element
        for element in root.iter()
        if _local(element.tag) == "part"
        and any(_local(child.tag) == "measure" for child in element)
    )
    return [child for child in part if _local(child.tag) == "measure"]


def _measure_number(measure: ET.Element, fallback: int) -> str:
    return measure.attrib.get("number", str(fallback))


def measure_fingerprint(measure: ET.Element) -> str:
    """Return a layout-independent representation of musical note content."""

    events = []
    for child in measure:
        if _local(child.tag) != "note":
            continue
        pitch = _child(child, "pitch")
        # Sibelius exports may use different divisions and may insert hidden
        # padding rests in otherwise equivalent scores.  Neither changes note
        # identity, so fingerprints intentionally exclude duration and rests.
        if pitch is None:
            continue
        pitch_name = (
            f"{_text(pitch, 'step')}:{_text(pitch, 'alter', '0')}:"
            f"{_text(pitch, 'octave')}"
        )
        events.append(
            (
                pitch_name,
                _text(child, "staff", "1"),
                _child(child, "chord") is not None,
                _child(child, "grace") is not None,
            )
        )
    return json.dumps(events, ensure_ascii=False, separators=(",", ":"))


def _ending_numbers(value: str) -> list[int]:
    """Parse MusicXML ending labels such as ``1``, ``1,2``, or ``1-2``."""

    return sorted({int(number) for number in re.findall(r"\d+", value)})


def _repeat_metadata(measures: list[ET.Element]) -> list[dict]:
    """Return measure-local repeat markers and active volta numbers."""

    metadata = []
    active_endings: set[int] = set()
    for measure in measures:
        forward = False
        backward_times: list[int] = []
        ending_starts: set[int] = set()
        ending_stops: set[int] = set()
        navigation: set[str] = set()
        for element in measure.iter():
            name = _local(element.tag)
            if name == "repeat":
                direction = element.attrib.get("direction", "").strip().lower()
                if direction == "forward":
                    forward = True
                elif direction == "backward":
                    try:
                        repeat_times = int(float(element.attrib.get("times", "2")))
                    except ValueError:
                        repeat_times = 2
                    backward_times.append(max(2, repeat_times))
            elif name == "ending":
                numbers = set(_ending_numbers(element.attrib.get("number", "")))
                ending_type = element.attrib.get("type", "").strip().lower()
                if ending_type == "start":
                    ending_starts.update(numbers)
                elif ending_type in {"stop", "discontinue"}:
                    ending_stops.update(numbers)
            elif name == "sound":
                for attribute in ("dacapo", "dalsegno", "tocoda", "fine"):
                    if element.attrib.get(attribute):
                        navigation.add(attribute)
            elif name in {"segno", "coda"}:
                navigation.add(name)

        active_endings.update(ending_starts)
        volta_numbers = sorted(active_endings | ending_stops)
        # Some Sibelius MusicXML exports place both ``ending 1 start`` and
        # ``ending 2 start`` on the measure that contains the backward repeat.
        # The backward-repeat measure belongs to the first pass; the following
        # measure remains available to the second pass.
        if backward_times and 1 in volta_numbers and len(volta_numbers) > 1:
            volta_numbers = [1]
        metadata.append(
            {
                "forward": forward,
                "backward_times": max(backward_times) if backward_times else None,
                "volta_numbers": volta_numbers,
                "navigation": sorted(navigation),
            }
        )
        # A stop/discontinue closes the entire volta bracket currently active.
        # Clearing only the explicitly named number can leave an exporter-added
        # first-ending marker active for the rest of the score.
        if ending_stops:
            active_endings.clear()
    return metadata


def _measure_duration(measure: ET.Element) -> int:
    """Return the maximum MusicXML cursor reached by any voice in a measure."""

    cursor = 0
    maximum = 0
    for child in measure:
        name = _local(child.tag)
        try:
            duration = int(float(_text(child, "duration", "0")))
        except ValueError:
            duration = 0
        if name == "backup":
            cursor = max(0, cursor - duration)
        elif name == "forward":
            cursor += duration
            maximum = max(maximum, cursor)
        elif name == "note" and _child(child, "chord") is None:
            cursor += duration
            maximum = max(maximum, cursor)
    return maximum


def _first_measure_is_pickup(measures: list[ET.Element]) -> bool:
    """Detect a short first measure using its MusicXML meter and divisions."""

    if not measures:
        return False
    first = measures[0]
    if first.attrib.get("implicit", "").strip().lower() == "yes":
        return True
    attributes = _child(first, "attributes")
    time = _child(attributes, "time") if attributes is not None else None
    try:
        divisions = int(float(_text(attributes, "divisions", "0")))
        beats = int(float(_text(time, "beats", "0")))
        beat_type = int(float(_text(time, "beat-type", "0")))
    except ValueError:
        return False
    if divisions <= 0 or beats <= 0 or beat_type <= 0:
        return False
    nominal_duration = divisions * beats * 4 / beat_type
    actual_duration = _measure_duration(first)
    return 0 < actual_duration < nominal_duration


def build_structural_performance_order(
    measures: list[ET.Element],
) -> tuple[list[int], list[dict], list[dict], list[str]]:
    """Expand standard repeats/endings into zero-based written measure indices.

    The traversal is controlled solely by MusicXML ``repeat`` and ``ending``
    elements.  Navigation jumps such as D.C./D.S./Coda are reported explicitly
    until their target semantics can be guaranteed; they are never guessed.
    """

    metadata = _repeat_metadata(measures)
    open_starts: list[int] = []
    sections: list[dict] = []
    warnings: list[str] = []
    for index, item in enumerate(metadata):
        if item["forward"]:
            open_starts.append(index)
        if item["backward_times"] is not None:
            if open_starts:
                start = open_starts.pop()
            else:
                # A backward repeat without a matching forward marker normally
                # returns to the beginning.  In scores with an anacrusis,
                # Sibelius commonly leaves the pickup outside that repeat.
                start = 1 if _first_measure_is_pickup(measures) else 0
            sections.append(
                {
                    "start": start,
                    "end": index,
                    "times": int(item["backward_times"]),
                }
            )
    for start in open_starts:
        warnings.append(
            f"Forward repeat at written measure index {start + 1} has no backward repeat."
        )

    sections.sort(key=lambda section: (section["start"], section["end"]))
    for group_number, section in enumerate(sections, start=1):
        extended_end = section["end"]
        while (
            extended_end + 1 < len(metadata)
            and metadata[extended_end + 1]["volta_numbers"]
        ):
            extended_end += 1
        section["extended_end"] = extended_end
        section["repeat_group_id"] = f"R{group_number:02d}"

    unsupported_navigation = sorted(
        {
            marker
            for item in metadata
            for marker in item["navigation"]
        }
    )
    if unsupported_navigation:
        warnings.append(
            "MusicXML navigation markers require review and were not expanded: "
            + ", ".join(unsupported_navigation)
        )

    end_to_section = {section["end"]: section for section in sections}
    passes = {section["repeat_group_id"]: 1 for section in sections}
    order: list[int] = []
    index = 0
    maximum_steps = max(1000, len(measures) * 64)
    steps = 0
    while index < len(measures):
        steps += 1
        if steps > maximum_steps:
            raise ValueError("MusicXML repeat traversal exceeded its safety limit")

        containing = [
            section
            for section in sections
            if section["start"] <= index <= section["extended_end"]
        ]
        active_section = (
            max(containing, key=lambda section: (section["start"], -section["end"]))
            if containing
            else None
        )
        current_pass = (
            passes[active_section["repeat_group_id"]]
            if active_section is not None
            else 1
        )
        volta_numbers = metadata[index]["volta_numbers"]
        included = not volta_numbers or current_pass in volta_numbers
        if included:
            order.append(index)

        section = end_to_section.get(index)
        if included and section is not None:
            group_id = section["repeat_group_id"]
            if passes[group_id] < section["times"]:
                passes[group_id] += 1
                for nested in sections:
                    if (
                        nested is not section
                        and section["start"] <= nested["start"]
                        and nested["end"] <= section["end"]
                    ):
                        passes[nested["repeat_group_id"]] = 1
                index = section["start"]
                continue
        index += 1

    return order, metadata, sections, warnings


def align_fingerprints(
    written: list[str], unfolded: list[str]
) -> tuple[list[int | None], list[dict]]:
    """Map every unfolded item to a written index using exact matching blocks."""

    mapping: list[int | None] = [None] * len(unfolded)
    evidence: list[dict] = []

    def add_blocks(target_start: int, target_end: int) -> int:
        matcher = SequenceMatcher(
            None,
            written,
            unfolded[target_start:target_end],
            autojunk=False,
        )
        added = 0
        for block in matcher.get_matching_blocks():
            if not block.size:
                continue
            evidence.append(
                {
                    "written_start": block.a + 1,
                    "unfolded_start": target_start + block.b + 1,
                    "length": block.size,
                }
            )
            for offset in range(block.size):
                unfolded_index = target_start + block.b + offset
                written_index = block.a + offset
                if mapping[unfolded_index] is None:
                    mapping[unfolded_index] = written_index
                    added += 1
        return added

    add_blocks(0, len(unfolded))
    while True:
        gaps = []
        start = None
        for index, value in enumerate(mapping + [0]):
            if value is None and start is None:
                start = index
            elif value is not None and start is not None:
                gaps.append((start, index))
                start = None
        progress = sum(add_blocks(start, end) for start, end in gaps)
        if not progress:
            break

    # Exporters can rewrite a small ending measure while leaving exact musical
    # anchors on both sides.  Fill only a gap whose number of unfolded items
    # exactly equals the number of missing consecutive written items.  This is
    # deterministic interpolation, not fuzzy content matching.
    gaps = []
    start = None
    for index, value in enumerate(mapping + [0]):
        if value is None and start is None:
            start = index
        elif value is not None and start is not None:
            gaps.append((start, index))
            start = None
    for start, end in gaps:
        if start == 0 or end >= len(mapping):
            continue
        left = mapping[start - 1]
        right = mapping[end]
        if left is None or right is None:
            continue
        if right - left - 1 != end - start:
            continue
        for offset, unfolded_index in enumerate(range(start, end), start=1):
            mapping[unfolded_index] = left + offset
        evidence.append(
            {
                "written_start": left + 2,
                "unfolded_start": start + 1,
                "length": end - start,
                "method": "bounded_contiguous_interpolation",
            }
        )

    return mapping, evidence


def build_repeat_mapping(
    written_xml: Path, unfolded_xml: Path | None = None
) -> dict:
    """Build repeat occurrences from written MusicXML structure.

    ``unfolded_xml`` is optional validation evidence only.  It never changes
    the structural traversal or fills a missing occurrence.
    """

    written_measures = _first_part_measures(written_xml)
    mapping, metadata, sections, structural_warnings = (
        build_structural_performance_order(written_measures)
    )
    by_written: dict[int, list[int]] = defaultdict(list)
    for unfolded_index, written_index in enumerate(mapping, start=1):
        by_written[written_index].append(unfolded_index)

    group_by_written: dict[int, list[str]] = defaultdict(list)
    for section in sections:
        for written_index in range(
            section["start"], section["extended_end"] + 1
        ):
            group_by_written[written_index].append(section["repeat_group_id"])

    unfolded_measures = (
        _first_part_measures(unfolded_xml) if unfolded_xml is not None else []
    )
    validation = {
        "provided": unfolded_xml is not None,
        "expected_measure_count": len(mapping),
        "unfolded_measure_count": len(unfolded_measures),
        "mismatch_indices": [],
        "passed": True,
    }
    if unfolded_measures:
        written_fingerprints = [measure_fingerprint(m) for m in written_measures]
        unfolded_fingerprints = [measure_fingerprint(m) for m in unfolded_measures]
        expected_fingerprints = [written_fingerprints[index] for index in mapping]
        mismatch_indices = [
            index
            for index, (expected, actual) in enumerate(
                zip(expected_fingerprints, unfolded_fingerprints, strict=False),
                start=1,
            )
            if expected != actual
        ]
        mismatch_indices.extend(
            range(
                min(len(expected_fingerprints), len(unfolded_fingerprints)) + 1,
                max(len(expected_fingerprints), len(unfolded_fingerprints)) + 1,
            )
        )
        validation["mismatch_indices"] = sorted(set(mismatch_indices))
        validation["passed"] = not mismatch_indices
        if mismatch_indices:
            structural_warnings.append(
                "The structural repeat traversal differs from unfolded MusicXML "
                f"at {len(set(mismatch_indices))} performance measure positions."
            )

    validation_mismatches = set(validation["mismatch_indices"])
    rows = []
    for unfolded_index, written_zero_index in enumerate(mapping, start=1):
        occurrences = by_written[written_zero_index]
        item = metadata[written_zero_index]
        statuses = []
        if item["forward"]:
            statuses.append("repeat_start")
        if item["volta_numbers"]:
            statuses.append(
                "ending_" + "-".join(str(value) for value in item["volta_numbers"])
            )
        if item["backward_times"] is not None:
            statuses.append("repeat_end")
        if len(occurrences) > 1 and not statuses:
            statuses.append("repeat_body")
        if item["navigation"]:
            statuses.append("navigation_review")
        unfolded_measure = (
            _measure_number(unfolded_measures[unfolded_index - 1], unfolded_index)
            if unfolded_index <= len(unfolded_measures)
            else str(unfolded_index)
        )
        rows.append(
            {
                "unfolded_measure_index": unfolded_index,
                "unfolded_measure": unfolded_measure,
                "performance_measure": unfolded_index,
                "written_measure_index": written_zero_index + 1,
                "written_measure": _measure_number(
                    written_measures[written_zero_index], written_zero_index + 1
                ),
                "is_repeated_measure": len(occurrences) > 1,
                "repeat_occurrence": occurrences.index(unfolded_index) + 1,
                "repeat_occurrence_count": len(occurrences),
                "repeat_group_id": "+".join(group_by_written[written_zero_index]),
                "repeat_status": "+".join(statuses) if statuses else "none",
                "volta_numbers": json.dumps(item["volta_numbers"]),
                "repeat_source": "repetition_musicxml",
                "mapping_status": (
                    "structural_unfolded_disagreement"
                    if unfolded_index in validation_mismatches
                    else "parsed_repeat_structure"
                ),
            }
        )

    return {
        "written_xml": str(written_xml),
        "unfolded_xml": str(unfolded_xml) if unfolded_xml is not None else "",
        "repeat_source": "repetition_musicxml",
        "written_measure_count": len(written_measures),
        "unfolded_measure_count": len(mapping),
        "mapped_unfolded_measures": len(mapping),
        "unresolved_unfolded_measures": [],
        "repeat_group_count": len(sections),
        "repeat_sections": sections,
        "structural_warnings": structural_warnings,
        "unfolded_validation": validation,
        "rows": rows,
    }


def write_repeat_mapping(report: dict, output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(
        json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--written-xml", type=Path, required=True)
    parser.add_argument("--unfolded-xml", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = build_repeat_mapping(args.written_xml, args.unfolded_xml)
    write_repeat_mapping(report, args.output_csv, args.output_json)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    validation = report["unfolded_validation"]
    if report["unresolved_unfolded_measures"] or (
        validation["provided"] and not validation["passed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
