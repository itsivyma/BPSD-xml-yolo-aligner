"""Draw one scan-level fingering-to-note candidate for human review."""

from __future__ import annotations

import argparse
import csv
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


def generate_check(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    yolo_line: int,
    pitch: str,
    bps_time: float,
    note_id: int,
    beat_position: float | None,
    output_path: Path,
    output_csv_path: Path,
    page: int = 1,
) -> None:
    image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    systems = detect_systems(image)
    categories = load_categories(notes_json_path)
    boxes = load_yolo(yolo_path, categories)
    box = next(item for item in boxes if item["txt_line"] == yolo_line)
    if not box["class"].startswith("fingering"):
        raise ValueError(f"Y{yolo_line} is {box['class']}, not fingering1..5")

    xml_page = parse_musicxml_page(xml_path, page)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    matches = [
        note
        for note in xml_page["notes"]
        if note.get("note_id") == note_id
        and note["pitch_name"] == pitch
        and abs(note["bps_time"] - bps_time) < 0.001
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one note candidate; found {len(matches)}")
    note = matches[0]
    clean_systems, clean_boundaries = _measure_boundaries_for_page(
        clean_image,
        xml_page,
    )
    scan_boundaries = align_barlines_from_reference(
        image,
        systems,
        clean_systems,
        clean_boundaries,
    )
    first_measure_by_system = {}
    for measure in xml_page["measures"]:
        first_measure_by_system.setdefault(
            measure["system"],
            measure["measure"],
        )
    geometry = _endpoint_geometry(
        note,
        clean_image,
        image,
        clean_systems,
        systems,
        clean_boundaries,
        scan_boundaries,
        first_measure_by_system,
    )
    system = systems[note["system"] - 1]
    staff = system.upper if note["staff"] == 1 else system.lower
    snapped = geometry["scan"]

    width, height = image.size
    x0 = (box["x"] - box["w"] / 2) * width
    y0 = (box["y"] - box["h"] / 2) * height
    x1 = (box["x"] + box["w"] / 2) * width
    y1 = (box["y"] + box["h"] / 2) * height
    crop = (
        round(min(x0, snapped["x"]) - 150),
        round(min(y0, snapped["y"], staff.lines[0]) - 90),
        round(max(x1, snapped["x"]) + 150),
        round(max(y1, snapped["y"], staff.lines[-1]) + 90),
    )
    crop_image = image.crop(crop)
    target_width = 760
    scale = target_width / crop_image.width
    crop_image = crop_image.resize(
        (target_width, round(crop_image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    header_height = 135
    canvas = Image.new(
        "RGB",
        (target_width, header_height + crop_image.height),
        "white",
    )
    canvas.paste(crop_image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    digit = box["class"].removeprefix("fingering")
    draw.text((24, 14), f"Y{yolo_line} fingering{digit} note check", fill="black", font=_font(28, bold=True))
    position_text = f"BPSD time {bps_time:.3f}"
    if beat_position is not None:
        position_text += f" = beat {beat_position:g} in the 3/4 measure"
    draw.text(
        (24, 55),
        f"Candidate: {pitch}; {position_text}; note ID {note_id}; staff {note['staff']}",
        fill="black",
        font=_font(19, bold=True),
    )
    draw.text(
        (24, 88),
        "Blue = YOLO fingering box; green = candidate connected note",
        fill="#1261ff",
        font=_font(17, bold=True),
    )

    def point(x: float, y: float) -> tuple[float, float]:
        return ((x - crop[0]) * scale, header_height + (y - crop[1]) * scale)

    bx0, by0 = point(x0, y0)
    bx1, by1 = point(x1, y1)
    draw.rectangle((bx0, by0, bx1, by1), outline="#1261ff", width=2)
    draw.text(
        (bx0, by0 - 27),
        f"Y{yolo_line} = {digit}",
        fill="#1261ff",
        font=_font(20, bold=True),
    )
    nx, ny = point(snapped["x"], snapped["y"])
    radius = 13
    draw.ellipse(
        (nx - radius, ny - radius, nx + radius, ny + radius),
        outline="#00a63c",
        width=3,
    )
    draw.text(
        (nx - 12, ny - 48),
        "N",
        fill="#00a63c",
        font=_font(23, bold=True),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "yolo_line",
                "class_id",
                "class",
                "pitch_candidate",
                "bps_time_candidate",
                "beat_position_candidate",
                "note_id_candidate",
                "staff_candidate",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "yolo_line": yolo_line,
                "class_id": box["class_id"],
                "class": box["class"],
                "pitch_candidate": pitch,
                "bps_time_candidate": f"{bps_time:.3f}",
                "beat_position_candidate": (
                    "" if beat_position is None else f"{beat_position:g}"
                ),
                "note_id_candidate": note_id,
                "staff_candidate": note["staff"],
                "review_status": "needs_manual_confirmation",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clean-image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--yolo-line", type=int, required=True)
    parser.add_argument("--pitch", required=True)
    parser.add_argument("--bps-time", type=float, required=True)
    parser.add_argument("--note-id", type=int, required=True)
    parser.add_argument("--beat-position", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    generate_check(
        image_path=args.image,
        clean_image_path=args.clean_image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        yolo_line=args.yolo_line,
        pitch=args.pitch,
        bps_time=args.bps_time,
        note_id=args.note_id,
        beat_position=args.beat_position,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
    )


if __name__ == "__main__":
    main()
