"""Create a batch sheet for first-page MusicXML tuplet groups."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from PIL import Image, ImageDraw

from bps_xml_alignment import (
    align_barlines_from_reference,
    attach_bps_note_ids,
    detect_systems,
    load_bps_notes,
    load_categories,
    load_yolo,
    parse_musicxml_page,
)
from slur_endpoint_check import _endpoint_geometry, _font, _measure_boundaries_for_page
from staccato_batch_check import _box_pixels


TUPLET_CLASSES = {"tuplet3", "tuplet5", "tuplet6"}


def _groups(notes: list[dict]) -> list[dict]:
    ordered = sorted(notes, key=lambda note: note["xml_note_sequence"])
    output = []
    for start in ordered:
        if not any(mark["type"] == "start" for mark in start.get("tuplet_marks", [])):
            continue
        count = start.get("actual_notes", 0)
        members = [
            note
            for note in ordered
            if note["xml_measure"] == start["xml_measure"]
            and note["staff"] == start["staff"]
            and note["voice"] == start["voice"]
            and note.get("actual_notes") == count
            and note["xml_note_sequence"] >= start["xml_note_sequence"]
        ][:count]
        if len(members) != count:
            raise ValueError(
                f"Tuplet {count} in measure {start['xml_measure']} has {len(members)} members"
            )
        output.append(
            {
                "actual_notes": count,
                "normal_notes": start.get("normal_notes", 0),
                "xml_measure": start["xml_measure"],
                "system": start["system"],
                "staff": start["staff"],
                "voice": start["voice"],
                "members": members,
            }
        )
    return output


def _beat(note: dict) -> float:
    return 1 + (note["bps_time"] - (note["xml_measure"] - 1)) * 3


def generate_sheet(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    output_path: Path,
    output_csv_path: Path,
    page: int = 1,
) -> None:
    scan_image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    xml_page = parse_musicxml_page(xml_path, page)
    attach_bps_note_ids(xml_page["notes"], load_bps_notes(bps_notes_path))
    groups = _groups(xml_page["notes"])
    boxes = [
        box
        for box in load_yolo(yolo_path, load_categories(notes_json_path))
        if box["class"] in TUPLET_CLASSES
    ]
    if len(boxes) != len(groups):
        raise ValueError(f"YOLO tuplets={len(boxes)}, XML groups={len(groups)}")

    clean_systems, clean_boundaries = _measure_boundaries_for_page(clean_image, xml_page)
    scan_systems = detect_systems(scan_image)
    scan_boundaries = align_barlines_from_reference(
        scan_image, scan_systems, clean_systems, clean_boundaries
    )
    first_measure_by_system = {}
    for measure in xml_page["measures"]:
        first_measure_by_system.setdefault(measure["system"], measure["measure"])

    group_geometries = []
    for group in groups:
        group_geometries.append(
            [
                _endpoint_geometry(
                    note,
                    clean_image,
                    scan_image,
                    clean_systems,
                    scan_systems,
                    clean_boundaries,
                    scan_boundaries,
                    first_measure_by_system,
                )
                for note in group["members"]
            ]
        )

    width, height = scan_image.size
    costs = []
    for box in boxes:
        bx, by = box["x"] * width, box["y"] * height
        box_system = min(scan_systems, key=lambda system: abs(system.center - by)).number
        row = []
        box_count = int(box["class"].removeprefix("tuplet"))
        for group, geometries in zip(groups, group_geometries):
            gx = sum(item["scan"]["x"] for item in geometries) / len(geometries)
            gy = sum(item["scan"]["y"] for item in geometries) / len(geometries)
            row.append(
                (1_000_000 if box_system != group["system"] else 0)
                + (1_000_000 if box_count != group["actual_notes"] else 0)
                + 1.5 * abs(bx - gx)
                + 0.2 * abs(by - gy)
            )
        costs.append(row)
    order = min(
        itertools.permutations(range(len(groups))),
        key=lambda candidate: sum(costs[i][j] for i, j in enumerate(candidate)),
    )
    pairs = [
        (box, groups[group_index], group_geometries[group_index])
        for box, group_index in zip(boxes, order)
    ]
    pairs.sort(key=lambda item: item[0]["txt_line"])

    panel_width, panel_height, header_height = 700, 400, 140
    canvas = Image.new("RGB", (1420, 1220), "#d9d9d9")
    rows = []
    for index, (box, group, geometries) in enumerate(pairs):
        x0, y0, x1, y1 = _box_pixels(box, scan_image.size)
        note_xs = [item["scan"]["x"] for item in geometries]
        note_ys = [item["scan"]["y"] for item in geometries]
        crop = (
            round(min(x0, *note_xs) - 120),
            round(min(y0, *note_ys) - 100),
            round(max(x1, *note_xs) + 120),
            round(max(y1, *note_ys) + 100),
        )
        cropped = scan_image.crop(crop)
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        panel.paste(
            cropped.resize(
                (panel_width, panel_height - header_height),
                Image.Resampling.LANCZOS,
            ),
            (0, header_height),
        )
        draw = ImageDraw.Draw(panel)
        members = group["members"]
        note_ids = "+".join(str(note.get("note_id", "")) for note in members)
        pitches = "+".join(note["pitch_name"] for note in members)
        start, end = members[0], members[-1]
        draw.text(
            (18, 9),
            f"Y{box['txt_line']} {box['class']} = complete {group['actual_notes']}-note group",
            fill="black",
            font=_font(23, True),
        )
        draw.text(
            (18, 41),
            (
                f"XML m.{group['xml_measure']}; staff {group['staff']}; "
                f"BPSD {start['bps_time']:.3f}-{end['bps_time']:.3f}"
            ),
            fill="black",
            font=_font(18, True),
        )
        draw.text(
            (18, 70),
            f"beats {_beat(start):.3f}-{_beat(end):.3f}; note IDs {note_ids}",
            fill="black",
            font=_font(16, True),
        )
        draw.text((18, 97), f"pitches {pitches}", fill="black", font=_font(15, True))
        draw.text((18, 119), "blue = tuplet number; green = every group note", fill="#1261ff", font=_font(15, True))
        sx = panel_width / cropped.width
        sy = (panel_height - header_height) / cropped.height

        def point(px: float, py: float) -> tuple[float, float]:
            return ((px - crop[0]) * sx, header_height + (py - crop[1]) * sy)

        bx0, by0 = point(x0, y0)
        bx1, by1 = point(x1, y1)
        draw.rectangle((bx0, by0, bx1, by1), outline="#1261ff", width=3)
        number_center = ((bx0 + bx1) / 2, (by0 + by1) / 2)
        for geometry in geometries:
            px, py = point(geometry["scan"]["x"], geometry["scan"]["y"])
            draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline="#00a63c", width=3)
        start_point = point(geometries[0]["scan"]["x"], geometries[0]["scan"]["y"])
        end_point = point(geometries[-1]["scan"]["x"], geometries[-1]["scan"]["y"])
        draw.line((*number_center, *start_point), fill="#00a63c", width=2)
        draw.line((*number_center, *end_point), fill="#00a63c", width=2)
        canvas.paste(panel, (10 + index % 2 * panel_width, 10 + index // 2 * panel_height))
        rows.append(
            {
                "page_id": image_path.stem,
                "yolo_line": box["txt_line"],
                "class_id": box["class_id"],
                "class": box["class"],
                "xml_measure": group["xml_measure"],
                "actual_notes": group["actual_notes"],
                "normal_notes": group["normal_notes"],
                "start_bps_time": f"{start['bps_time']:.3f}",
                "end_note_bps_time": f"{end['bps_time']:.3f}",
                "start_beat": f"{_beat(start):.3f}",
                "end_note_beat": f"{_beat(end):.3f}",
                "note_ids": note_ids,
                "pitches": pitches,
                "staff": group["staff"],
                "match_status": "barline_aligned_candidate",
                "review_status": "needs_manual_confirmation",
                "comment": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clean-image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    generate_sheet(
        image_path=args.image,
        clean_image_path=args.clean_image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
    )


if __name__ == "__main__":
    main()
