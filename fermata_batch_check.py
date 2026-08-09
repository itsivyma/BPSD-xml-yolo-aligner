"""Create a batch sheet for first-page fermata targets."""

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
    note_pixel_position,
    parse_musicxml_page,
)
from slur_endpoint_check import _endpoint_geometry, _font, _measure_boundaries_for_page
from staccato_batch_check import _box_pixels, _chord_notes


FERMATA_CLASSES = {"fermataAbove", "fermataBelow"}


def _rest_geometry(
    rest: dict,
    clean_systems: list,
    scan_systems: list,
    clean_boundaries: list[list[int]],
    scan_boundaries: list[list[dict]],
    first_measure_by_system: dict[int, int],
) -> dict:
    system_index = rest["system"] - 1
    clean_system = clean_systems[system_index]
    scan_system = scan_systems[system_index]
    clean_x = clean_system.x_left + rest["x_norm"] * (
        clean_system.x_right - clean_system.x_left
    )
    measure_index = rest["xml_measure"] - first_measure_by_system[rest["system"]]
    clean_left = clean_boundaries[system_index][measure_index]
    clean_right = clean_boundaries[system_index][measure_index + 1]
    scan_left = scan_boundaries[system_index][measure_index]["x"]
    scan_right = scan_boundaries[system_index][measure_index + 1]["x"]
    within = (clean_x - clean_left) / (clean_right - clean_left)
    scan_x = scan_left + within * (scan_right - scan_left)
    staff = scan_system.upper if rest["staff"] == 1 else scan_system.lower
    return {"scan": {"x": scan_x, "y": staff.center}}


def _beat_position(event: dict) -> float:
    return 1 + (event["bps_time"] - (event["xml_measure"] - 1)) * 3


def _best_assignment(
    boxes: list[dict],
    targets: list[dict],
    geometries: list[dict],
    scan_systems: list,
    image_size: tuple[int, int],
) -> list[tuple[dict, dict, dict]]:
    width, height = image_size
    costs = []
    for box in boxes:
        bx = box["x"] * width
        by = box["y"] * height
        box_system = min(
            scan_systems,
            key=lambda system: abs(system.center - by),
        ).number
        row = []
        for target, geometry in zip(targets, geometries):
            tx = geometry["scan"]["x"]
            ty = geometry["scan"]["y"]
            violation = (
                box["class"] == "fermataAbove" and by >= ty
            ) or (
                box["class"] == "fermataBelow" and by <= ty
            )
            row.append(
                (1_000_000 if box_system != target["system"] else 0)
                + 1.4 * abs(bx - tx)
                + 0.25 * abs(by - ty)
                + (300 if violation else 0)
            )
        costs.append(row)
    best = min(
        itertools.permutations(range(len(targets))),
        key=lambda order: sum(costs[index][target] for index, target in enumerate(order)),
    )
    return [
        (box, targets[target_index], geometries[target_index])
        for box, target_index in zip(boxes, best)
    ]


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
    targets = []
    for note in xml_page["notes"]:
        if note.get("fermata_marks"):
            targets.append({**note, "target_type": "chord"})
    for rest in xml_page["rests"]:
        if rest.get("fermata_marks"):
            targets.append({**rest, "target_type": "rest"})

    categories = load_categories(notes_json_path)
    boxes = [
        box
        for box in load_yolo(yolo_path, categories)
        if box["class"] in FERMATA_CLASSES
    ]
    if len(boxes) != len(targets):
        raise ValueError(f"YOLO fermatas={len(boxes)}, XML targets={len(targets)}")

    clean_systems, clean_boundaries = _measure_boundaries_for_page(
        clean_image,
        xml_page,
    )
    scan_systems = detect_systems(scan_image)
    scan_boundaries = align_barlines_from_reference(
        scan_image,
        scan_systems,
        clean_systems,
        clean_boundaries,
    )
    first_measure_by_system = {}
    for measure in xml_page["measures"]:
        first_measure_by_system.setdefault(measure["system"], measure["measure"])

    geometries = []
    for target in targets:
        if target["target_type"] == "chord":
            geometries.append(
                _endpoint_geometry(
                    target,
                    clean_image,
                    scan_image,
                    clean_systems,
                    scan_systems,
                    clean_boundaries,
                    scan_boundaries,
                    first_measure_by_system,
                )
            )
        else:
            geometries.append(
                _rest_geometry(
                    target,
                    clean_systems,
                    scan_systems,
                    clean_boundaries,
                    scan_boundaries,
                    first_measure_by_system,
                )
            )
    pairs = _best_assignment(boxes, targets, geometries, scan_systems, scan_image.size)
    pairs.sort(key=lambda item: item[0]["txt_line"])

    panel_width, panel_height, header_height = 700, 380, 125
    canvas = Image.new("RGB", (1420, 780), "#d9d9d9")
    rows = []
    for index, (box, target, geometry) in enumerate(pairs):
        if target["target_type"] == "chord":
            members = _chord_notes(target, xml_page["notes"])
            member_geometries = [
                _endpoint_geometry(
                    member,
                    clean_image,
                    scan_image,
                    clean_systems,
                    scan_systems,
                    clean_boundaries,
                    scan_boundaries,
                    first_measure_by_system,
                )
                for member in members
            ]
            pitches = "+".join(member["pitch_name"] for member in members)
            note_ids = "+".join(str(member.get("note_id", "")) for member in members)
        else:
            members = []
            member_geometries = [geometry]
            pitches = ""
            note_ids = ""

        x0, y0, x1, y1 = _box_pixels(box, scan_image.size)
        target_xs = [item["scan"]["x"] for item in member_geometries]
        target_ys = [item["scan"]["y"] for item in member_geometries]
        crop = (
            round(min(x0, *target_xs) - 170),
            round(min(y0, *target_ys) - 100),
            round(max(x1, *target_xs) + 170),
            round(max(y1, *target_ys) + 100),
        )
        cropped = scan_image.crop(crop)
        resized = cropped.resize(
            (panel_width, panel_height - header_height),
            Image.Resampling.LANCZOS,
        )
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        panel.paste(resized, (0, header_height))
        draw = ImageDraw.Draw(panel)
        beat = _beat_position(target)
        target_text = f"chord {pitches}" if members else "rest"
        draw.text((18, 10), f"Y{box['txt_line']} {box['class']}", fill="black", font=_font(24, True))
        draw.text(
            (18, 43),
            f"XML m.{target['xml_measure']}; {target_text}; BPSD {target['bps_time']:.3f} = beat {beat:g}",
            fill="black",
            font=_font(18, True),
        )
        detail = (
            f"note IDs {note_ids}; staff {target['staff']}"
            if members
            else f"no MIDI/note ID; staff {target['staff']}"
        )
        draw.text((18, 72), detail, fill="black", font=_font(17, True))
        draw.text((18, 98), "blue = fermata; green = full target", fill="#1261ff", font=_font(16, True))
        sx = panel_width / cropped.width
        sy = (panel_height - header_height) / cropped.height

        def point(px: float, py: float) -> tuple[float, float]:
            return ((px - crop[0]) * sx, header_height + (py - crop[1]) * sy)

        bx0, by0 = point(x0, y0)
        bx1, by1 = point(x1, y1)
        draw.rectangle((bx0, by0, bx1, by1), outline="#1261ff", width=3)
        for member_index, item in enumerate(member_geometries, start=1):
            px, py = point(item["scan"]["x"], item["scan"]["y"])
            draw.ellipse((px - 14, py - 14, px + 14, py + 14), outline="#00a63c", width=4)
            draw.line(((bx0 + bx1) / 2, (by0 + by1) / 2, px, py), fill="#00a63c", width=2)
            label = f"N{member_index}" if members else "R"
            draw.text((px - 10, py - 42), label, fill="#00a63c", font=_font(18, True))
        canvas.paste(panel, (10 + index % 2 * panel_width, 10 + index // 2 * panel_height))
        rows.append(
            {
                "page_id": image_path.stem,
                "yolo_line": box["txt_line"],
                "class_id": box["class_id"],
                "class": box["class"],
                "xml_measure": target["xml_measure"],
                "bps_time": f"{target['bps_time']:.3f}",
                "beat_position": f"{beat:g}",
                "target_type": target["target_type"],
                "note_ids": note_ids,
                "pitches": pitches,
                "chord_note_count": len(members),
                "staff": target["staff"],
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
