"""Create a first-page batch sheet for YOLO staccato-to-note review."""

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
from slur_endpoint_check import (
    _endpoint_geometry,
    _font,
    _measure_boundaries_for_page,
)


STACCATO_CLASSES = {"articStaccatoAbove", "articStaccatoBelow"}


def _box_pixels(box: dict, size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = size
    return (
        (box["x"] - box["w"] / 2) * width,
        (box["y"] - box["h"] / 2) * height,
        (box["x"] + box["w"] / 2) * width,
        (box["y"] + box["h"] / 2) * height,
    )


def _pair_cost(box: dict, note: dict, geometry: dict, image: Image.Image) -> float:
    width, height = image.size
    bx = box["x"] * width
    by = box["y"] * height
    nx = geometry["scan"]["x"]
    ny = geometry["scan"]["y"]
    systems = detect_systems(image)
    box_system = min(systems, key=lambda system: abs(system.center - by)).number
    if box_system != note["system"]:
        return 1_000_000
    placement_violation = (
        box["class"] == "articStaccatoAbove" and by >= ny
    ) or (
        box["class"] == "articStaccatoBelow" and by <= ny
    )
    return 1.5 * abs(bx - nx) + 0.25 * abs(by - ny) + (
        300 if placement_violation else 0
    )


def _best_assignment(
    boxes: list[dict],
    notes: list[dict],
    geometries: list[dict],
    image: Image.Image,
) -> list[tuple[dict, dict, dict]]:
    if len(boxes) != len(notes):
        raise ValueError(
            f"Expected equal YOLO/XML staccato counts; got {len(boxes)} and {len(notes)}"
        )
    costs = [
        [
            _pair_cost(box, note, geometry, image)
            for note, geometry in zip(notes, geometries)
        ]
        for box in boxes
    ]
    best_cost = float("inf")
    best_order = None
    for order in itertools.permutations(range(len(notes))):
        cost = sum(
            costs[box_index][note_index]
            for box_index, note_index in enumerate(order)
        )
        if cost < best_cost:
            best_cost = cost
            best_order = order
    assert best_order is not None
    return [
        (box, notes[note_index], geometries[note_index])
        for box, note_index in zip(boxes, best_order)
    ]


def _beat_position(note: dict) -> float:
    return 1 + (note["bps_time"] - (note["xml_measure"] - 1)) * 3


def _chord_notes(anchor: dict, notes: list[dict]) -> list[dict]:
    """Return every pitched note sharing the articulated chord onset."""

    return sorted(
        (
            note
            for note in notes
            if note["xml_chord_sequence"] == anchor["xml_chord_sequence"]
        ),
        key=lambda note: (note["midi"], note["xml_note_sequence"]),
    )


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
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    notes = [
        note
        for note in xml_page["notes"]
        if "staccato" in note.get("articulation_marks", [])
    ]
    categories = load_categories(notes_json_path)
    boxes = [
        box
        for box in load_yolo(yolo_path, categories)
        if box["class"] in STACCATO_CLASSES
    ]

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
    geometries = [
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
        for note in notes
    ]
    pairs = _best_assignment(boxes, notes, geometries, scan_image)
    pairs.sort(key=lambda item: item[0]["txt_line"])

    panel_width = 700
    panel_height = 360
    header_height = 112
    columns = 2
    rows = (len(pairs) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (panel_width * columns + 20, panel_height * rows + 20),
        "#d9d9d9",
    )
    canvas_draw = ImageDraw.Draw(canvas)
    csv_rows = []
    for index, (box, note, geometry) in enumerate(pairs):
        chord_notes = _chord_notes(note, xml_page["notes"])
        chord_geometries = [
            _endpoint_geometry(
                chord_note,
                clean_image,
                scan_image,
                clean_systems,
                scan_systems,
                clean_boundaries,
                scan_boundaries,
                first_measure_by_system,
            )
            for chord_note in chord_notes
        ]
        x0, y0, x1, y1 = _box_pixels(box, scan_image.size)
        nx = geometry["scan"]["x"]
        ny = geometry["scan"]["y"]
        chord_xs = [item["scan"]["x"] for item in chord_geometries]
        chord_ys = [item["scan"]["y"] for item in chord_geometries]
        crop = (
            round(min(x0, *chord_xs) - 150),
            round(min(y0, *chord_ys) - 90),
            round(max(x1, *chord_xs) + 150),
            round(max(y1, *chord_ys) + 90),
        )
        crop_image = scan_image.crop(crop)
        scale = panel_width / crop_image.width
        resized = crop_image.resize(
            (panel_width, panel_height - header_height),
            Image.Resampling.LANCZOS,
        )
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        panel.paste(resized, (0, header_height))
        draw = ImageDraw.Draw(panel)
        beat = _beat_position(note)
        pitches = "+".join(chord_note["pitch_name"] for chord_note in chord_notes)
        note_ids = "+".join(
            str(chord_note.get("note_id", "")) for chord_note in chord_notes
        )
        draw.text(
            (18, 10),
            f"Y{box['txt_line']} {box['class']}",
            fill="black",
            font=_font(24, bold=True),
        )
        draw.text(
            (18, 43),
            (
                f"XML m.{note['xml_measure']}; chord {pitches}; "
                f"BPSD {note['bps_time']:.3f} = beat {beat:g}"
            ),
            fill="black",
            font=_font(18, bold=True),
        )
        draw.text(
            (18, 72),
            f"note IDs {note_ids}; staff {note['staff']}; blue = dot, green = chord notes",
            fill="#1261ff",
            font=_font(16, bold=True),
        )
        sx = panel_width / crop_image.width
        sy = (panel_height - header_height) / crop_image.height

        def point(px: float, py: float) -> tuple[float, float]:
            return ((px - crop[0]) * sx, header_height + (py - crop[1]) * sy)

        bx0, by0 = point(x0, y0)
        bx1, by1 = point(x1, y1)
        draw.rectangle((bx0, by0, bx1, by1), outline="#1261ff", width=3)
        for chord_index, chord_geometry in enumerate(chord_geometries, start=1):
            px, py = point(
                chord_geometry["scan"]["x"],
                chord_geometry["scan"]["y"],
            )
            radius = 13
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                outline="#00a63c",
                width=4,
            )
            draw.line(
                ((bx0 + bx1) / 2, (by0 + by1) / 2, px, py),
                fill="#00a63c",
                width=2,
            )
            draw.text(
                (px - 10, py - 42),
                f"N{chord_index}",
                fill="#00a63c",
                font=_font(18, bold=True),
            )
        column = index % columns
        row = index // columns
        panel_x = 10 + column * panel_width
        panel_y = 10 + row * panel_height
        canvas.paste(panel, (panel_x, panel_y))
        canvas_draw.rectangle(
            (panel_x, panel_y, panel_x + panel_width - 1, panel_y + panel_height - 1),
            outline="#aaaaaa",
            width=1,
        )
        csv_rows.append(
            {
                "page_id": image_path.stem,
                "yolo_line": box["txt_line"],
                "class_id": box["class_id"],
                "class": box["class"],
                "xml_measure": note["xml_measure"],
                "bps_time": f"{note['bps_time']:.3f}",
                "beat_position": f"{beat:g}",
                "note_ids": note_ids,
                "pitches": pitches,
                "chord_note_count": len(chord_notes),
                "staff": note["staff"],
                "match_status": "barline_aligned_candidate",
                "review_status": "needs_manual_confirmation",
                "comment": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


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
