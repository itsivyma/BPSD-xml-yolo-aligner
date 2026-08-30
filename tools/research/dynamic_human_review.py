"""Build a human-review sheet for MusicXML-aligned dynamic glyphs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bps_xml_alignment import (
    attach_bps_note_ids,
    detect_systems,
    load_bps_notes,
    load_categories,
    load_yolo,
    match_dynamics,
    parse_musicxml_page,
)


REVIEW_FIELDS = [
    "page_id",
    "yolo_line",
    "class_id",
    "class",
    "x",
    "y",
    "w",
    "h",
    "candidate_start_meas",
    "candidate_end_meas",
    "xml_measure",
    "xml_symbol",
    "candidate_status",
    "review_status",
    "corrected_start_meas",
    "corrected_end_meas",
    "comment",
]


def build_review(
    *,
    image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    output_dir: Path,
    page_number: int,
) -> tuple[Path, Path]:
    page_id = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{page_id}_dynamic_human_review.csv"
    prior_reviews = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            prior_reviews = {
                int(row["yolo_line"]): row
                for row in csv.DictReader(file)
            }

    image = Image.open(image_path).convert("RGB")
    systems = detect_systems(image)
    categories = load_categories(notes_json_path)
    boxes = load_yolo(yolo_path, categories=categories)
    dynamic_boxes = [
        box for box in boxes if box["class_id"] in {18, 20, 21}
    ]

    xml_page = parse_musicxml_page(xml_path, page_number=page_number)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    aligned, _unused_xml = match_dynamics(
        dynamic_boxes,
        xml_page["dynamics"],
        systems,
        image.height,
    )
    aligned_by_line = {
        int(row["txt_line"]): row
        for row in aligned
    }

    review_rows = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=14)
    colors = {
        "dynamicF": (0, 160, 0),
        "dynamicP": (0, 90, 220),
        "dynamicS": (210, 70, 0),
    }

    for box in sorted(dynamic_boxes, key=lambda item: item["txt_line"]):
        candidate = aligned_by_line.get(box["txt_line"], {})
        start = candidate.get("start_meas", "")
        end = candidate.get("end_meas", "")
        prior = prior_reviews.get(box["txt_line"], {})
        review_rows.append(
            {
                "page_id": page_id,
                "yolo_line": box["txt_line"],
                "class_id": box["class_id"],
                "class": box["class"],
                "x": f"{box['x']:.6f}",
                "y": f"{box['y']:.6f}",
                "w": f"{box['w']:.6f}",
                "h": f"{box['h']:.6f}",
                "candidate_start_meas": start,
                "candidate_end_meas": end,
                "xml_measure": candidate.get("xml_measure", ""),
                "xml_symbol": candidate.get("xml_symbol", ""),
                "candidate_status": candidate.get("status", "unmatched"),
                "review_status": prior.get("review_status", "pending"),
                "corrected_start_meas": prior.get(
                    "corrected_start_meas",
                    "",
                ),
                "corrected_end_meas": prior.get(
                    "corrected_end_meas",
                    "",
                ),
                "comment": prior.get("comment", ""),
            }
        )

        x0 = round((box["x"] - box["w"] / 2) * image.width)
        y0 = round((box["y"] - box["h"] / 2) * image.height)
        x1 = round((box["x"] + box["w"] / 2) * image.width)
        y1 = round((box["y"] + box["h"] / 2) * image.height)
        color = colors[box["class"]]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=1)
        label = (
            f"Y{box['txt_line']} {box['class']} "
            f"t={start or '?'}"
        )
        label_y = max(0, y0 - 12)
        draw.text((x0, label_y), label, fill=color, font=font)

    overlay_path = output_dir / f"{page_id}_dynamic_human_review.png"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)
    overlay.save(overlay_path)
    for system in systems:
        spacing = max(
            system.upper.line_spacing,
            system.lower.line_spacing,
        )
        crop_top = max(0, round(system.upper.lines[0] - 7 * spacing))
        crop_bottom = min(
            image.height,
            round(system.lower.lines[-1] + 7 * spacing),
        )
        crop = overlay.crop((0, crop_top, image.width, crop_bottom))
        crop.save(
            output_dir
            / f"{page_id}_dynamic_system_{system.number:02d}.png"
        )
    return csv_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    csv_path, overlay_path = build_review(
        image_path=args.image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        output_dir=args.output_dir,
        page_number=args.page,
    )
    print(csv_path)
    print(overlay_path)


if __name__ == "__main__":
    main()
