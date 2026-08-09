"""Create a contact sheet for reviewing high-confidence slur candidates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw

from bps_xml_alignment import (
    align_barlines_from_reference,
    attach_bps_note_ids,
    build_slur_candidates,
    detect_systems,
    load_bps_notes,
    load_categories,
    load_yolo,
    parse_musicxml_page,
)
from slur_endpoint_check import (
    _endpoint_geometry,
    _find_endpoint_note,
    _font,
    _measure_boundaries_for_page,
)


OUTPUT_FIELDS = [
    "yolo_line",
    "candidate_id",
    "score",
    "start_pitch",
    "start_meas",
    "start_note_candidate",
    "scan_start_x",
    "scan_start_y",
    "end_pitch",
    "end_meas",
    "end_note_candidate",
    "scan_end_x",
    "scan_end_y",
    "review_status",
]


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _target_box(box: dict, image: Image.Image) -> tuple[float, float, float, float]:
    return (
        (box["x"] - box["w"] / 2) * image.width,
        (box["y"] - box["h"] / 2) * image.height,
        (box["x"] + box["w"] / 2) * image.width,
        (box["y"] + box["h"] / 2) * image.height,
    )


def _panel_endpoint(
    draw: ImageDraw.ImageDraw,
    *,
    source_x: float,
    source_y: float,
    crop: tuple[int, int, int, int],
    scale: float,
    image_x: int,
    image_y: int,
    color: str,
    label: str,
) -> None:
    x = image_x + (source_x - crop[0]) * scale
    y = image_y + (source_y - crop[1]) * scale
    radius = 13
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=4,
    )
    draw.text(
        (x - 9, y - 39),
        label,
        fill=color,
        font=_font(21, True),
        stroke_width=1,
        stroke_fill="white",
    )


def generate_sheet(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    summary_path: Path,
    output_path: Path,
    output_csv_path: Path,
    page: int = 1,
) -> None:
    scan_image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    categories = load_categories(notes_json_path)
    boxes = load_yolo(yolo_path, categories)
    box_by_line = {box["txt_line"]: box for box in boxes}

    summary_rows = [
        row
        for row in _load_csv(summary_path)
        if row["status"] == "high_confidence_candidate"
    ]
    summary_rows.sort(key=lambda row: int(row["yolo_line"]))
    if not summary_rows:
        raise ValueError("Summary contains no high-confidence candidates")

    xml_page = parse_musicxml_page(xml_path, page)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    candidates, issues = build_slur_candidates(xml_page["notes"], bps_notes)
    if issues:
        raise ValueError(f"Unpaired MusicXML slur endpoints: {issues}")
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in candidates
    }

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
        first_measure_by_system.setdefault(
            measure["system"],
            measure["measure"],
        )

    panels = []
    output_rows = []
    for summary in summary_rows:
        yolo_line = int(summary["yolo_line"])
        box = box_by_line[yolo_line]
        candidate = candidate_by_id[summary["best_candidate_id"]]
        start_note = _find_endpoint_note(
            xml_page["notes"],
            int(candidate["start_note_candidate"]),
            float(candidate["start_meas"]),
            candidate["start_pitch"],
        )
        end_note = _find_endpoint_note(
            xml_page["notes"],
            int(candidate["end_note_candidate"]),
            float(candidate["end_meas"]),
            candidate["end_pitch"],
        )
        start_geometry = _endpoint_geometry(
            start_note,
            clean_image,
            scan_image,
            clean_systems,
            scan_systems,
            clean_boundaries,
            scan_boundaries,
            first_measure_by_system,
        )
        end_geometry = _endpoint_geometry(
            end_note,
            clean_image,
            scan_image,
            clean_systems,
            scan_systems,
            clean_boundaries,
            scan_boundaries,
            first_measure_by_system,
        )
        box_rect = _target_box(box, scan_image)
        crop = (
            max(
                0,
                round(
                    min(
                        start_geometry["scan"]["x"],
                        end_geometry["scan"]["x"],
                        box_rect[0],
                    )
                    - 65
                ),
            ),
            max(
                0,
                round(
                    min(
                        start_geometry["scan"]["y"],
                        end_geometry["scan"]["y"],
                        box_rect[1],
                    )
                    - 70
                ),
            ),
            min(
                scan_image.width,
                round(
                    max(
                        start_geometry["scan"]["x"],
                        end_geometry["scan"]["x"],
                        box_rect[2],
                    )
                    + 65
                ),
            ),
            min(
                scan_image.height,
                round(
                    max(
                        start_geometry["scan"]["y"],
                        end_geometry["scan"]["y"],
                        box_rect[3],
                    )
                    + 70
                ),
            ),
        )
        panels.append(
            {
                "summary": summary,
                "candidate": candidate,
                "box": box,
                "box_rect": box_rect,
                "crop": crop,
                "start": start_geometry["scan"],
                "end": end_geometry["scan"],
            }
        )
        output_rows.append(
            {
                "yolo_line": yolo_line,
                "candidate_id": candidate["candidate_id"],
                "score": summary["best_score"],
                "start_pitch": candidate["start_pitch"],
                "start_meas": candidate["start_meas"],
                "start_note_candidate": candidate["start_note_candidate"],
                "scan_start_x": start_geometry["scan"]["x"],
                "scan_start_y": f"{start_geometry['scan']['y']:.1f}",
                "end_pitch": candidate["end_pitch"],
                "end_meas": candidate["end_meas"],
                "end_note_candidate": candidate["end_note_candidate"],
                "scan_end_x": end_geometry["scan"]["x"],
                "scan_end_y": f"{end_geometry['scan']['y']:.1f}",
                "review_status": "needs_manual_confirmation",
            }
        )

    columns = 3
    rows = math.ceil(len(panels) / columns)
    panel_width = 520
    panel_height = 330
    header_height = 105
    canvas = Image.new(
        "RGB",
        (columns * panel_width, header_height + rows * panel_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (25, 14),
        "High-confidence slur endpoint candidates",
        fill="black",
        font=_font(29, True),
    )
    draw.text(
        (25, 52),
        (
            "Blue = YOLO box; green A/red B = MusicXML endpoint notes. "
            "All panels remain unconfirmed."
        ),
        fill="#1261ff",
        font=_font(19, True),
    )

    for index, panel in enumerate(panels):
        column = index % columns
        row = index // columns
        panel_x = column * panel_width
        panel_y = header_height + row * panel_height
        summary = panel["summary"]
        candidate = panel["candidate"]
        draw.text(
            (panel_x + 14, panel_y + 8),
            (
                f"Y{summary['yolo_line']} ↔ {candidate['candidate_id']}  "
                f"score {float(summary['best_score']):.2f}"
            ),
            fill="black",
            font=_font(21, True),
        )
        draw.text(
            (panel_x + 14, panel_y + 37),
            (
                f"A {candidate['start_pitch']} "
                f"{candidate['start_meas']} (#{candidate['start_note_candidate']})"
                f"  →  B {candidate['end_pitch']} "
                f"{candidate['end_meas']} (#{candidate['end_note_candidate']})"
            ),
            fill="#333333",
            font=_font(17, True),
        )
        crop = panel["crop"]
        crop_image = scan_image.crop(crop)
        max_image_width = panel_width - 28
        max_image_height = panel_height - 82
        scale = min(
            max_image_width / crop_image.width,
            max_image_height / crop_image.height,
        )
        crop_image = crop_image.resize(
            (
                round(crop_image.width * scale),
                round(crop_image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )
        image_x = panel_x + 14
        image_y = panel_y + 72
        canvas.paste(crop_image, (image_x, image_y))

        box_rect = panel["box_rect"]
        draw.rectangle(
            (
                image_x + (box_rect[0] - crop[0]) * scale,
                image_y + (box_rect[1] - crop[1]) * scale,
                image_x + (box_rect[2] - crop[0]) * scale,
                image_y + (box_rect[3] - crop[1]) * scale,
            ),
            outline="#1261ff",
            width=2,
        )
        _panel_endpoint(
            draw,
            source_x=panel["start"]["x"],
            source_y=panel["start"]["y"],
            crop=crop,
            scale=scale,
            image_x=image_x,
            image_y=image_y,
            color="#00aa45",
            label="A",
        )
        _panel_endpoint(
            draw,
            source_x=panel["end"]["x"],
            source_y=panel["end"]["y"],
            crop=crop,
            scale=scale,
            image_x=image_x,
            image_y=image_y,
            color="#e53329",
            label="B",
        )
        draw.rectangle(
            (
                panel_x,
                panel_y,
                panel_x + panel_width - 1,
                panel_y + panel_height - 1,
            ),
            outline="#bbbbbb",
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clean-image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
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
        summary_path=args.summary,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
    )


if __name__ == "__main__":
    main()
