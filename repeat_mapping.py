"""Map repetition-preserving MusicXML measures onto an unfolded timeline."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from defusedxml import ElementTree as SafeET


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local(child.tag) == name), None)


def _text(element: ET.Element, name: str, default: str = "") -> str:
    child = _child(element, name)
    return (child.text or default) if child is not None else default


def _first_part_measures(path: Path) -> list[ET.Element]:
    root = SafeET.parse(
        path,
        forbid_dtd=False,
        forbid_entities=True,
        forbid_external=True,
    ).getroot()
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


def build_repeat_mapping(written_xml: Path, unfolded_xml: Path) -> dict:
    written_measures = _first_part_measures(written_xml)
    unfolded_measures = _first_part_measures(unfolded_xml)
    written_fingerprints = [measure_fingerprint(m) for m in written_measures]
    unfolded_fingerprints = [measure_fingerprint(m) for m in unfolded_measures]
    mapping, evidence = align_fingerprints(written_fingerprints, unfolded_fingerprints)
    interpolated_unfolded = {
        index
        for block in evidence
        if block.get("method") == "bounded_contiguous_interpolation"
        for index in range(
            int(block["unfolded_start"]),
            int(block["unfolded_start"]) + int(block["length"]),
        )
    }

    unresolved = [index + 1 for index, value in enumerate(mapping) if value is None]
    by_written: dict[int, list[int]] = defaultdict(list)
    for unfolded_index, written_index in enumerate(mapping, start=1):
        if written_index is not None:
            by_written[written_index].append(unfolded_index)

    repeated_indices = sorted(
        index for index, occurrences in by_written.items() if len(occurrences) > 1
    )
    repeat_group_by_index: dict[int, str] = {}
    group = 0
    previous = None
    for written_index in repeated_indices:
        if previous is None or written_index != previous + 1:
            group += 1
        repeat_group_by_index[written_index] = f"R{group:02d}"
        previous = written_index

    rows = []
    for unfolded_index, written_zero_index in enumerate(mapping, start=1):
        if written_zero_index is None:
            rows.append(
                {
                    "unfolded_measure_index": unfolded_index,
                    "unfolded_measure": _measure_number(
                        unfolded_measures[unfolded_index - 1], unfolded_index
                    ),
                    "written_measure_index": "",
                    "written_measure": "",
                    "is_repeated_measure": "",
                    "repeat_occurrence": "",
                    "repeat_occurrence_count": "",
                    "repeat_group_id": "",
                    "mapping_status": "unresolved",
                }
            )
            continue
        occurrences = by_written[written_zero_index]
        rows.append(
            {
                "unfolded_measure_index": unfolded_index,
                "unfolded_measure": _measure_number(
                    unfolded_measures[unfolded_index - 1], unfolded_index
                ),
                "written_measure_index": written_zero_index + 1,
                "written_measure": _measure_number(
                    written_measures[written_zero_index], written_zero_index + 1
                ),
                "is_repeated_measure": len(occurrences) > 1,
                "repeat_occurrence": occurrences.index(unfolded_index) + 1,
                "repeat_occurrence_count": len(occurrences),
                "repeat_group_id": repeat_group_by_index.get(written_zero_index, ""),
                "mapping_status": (
                    "matched_contextual"
                    if unfolded_index in interpolated_unfolded
                    else "matched_fingerprint"
                ),
            }
        )

    return {
        "written_xml": str(written_xml),
        "unfolded_xml": str(unfolded_xml),
        "written_measure_count": len(written_measures),
        "unfolded_measure_count": len(unfolded_measures),
        "mapped_unfolded_measures": len(unfolded_measures) - len(unresolved),
        "unresolved_unfolded_measures": unresolved,
        "repeat_group_count": group,
        "matching_blocks": evidence,
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
    parser.add_argument("--unfolded-xml", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = build_repeat_mapping(args.written_xml, args.unfolded_xml)
    write_repeat_mapping(report, args.output_csv, args.output_json)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if report["unresolved_unfolded_measures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
