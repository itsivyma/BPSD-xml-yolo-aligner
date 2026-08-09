"""Draw review panels for scan slurs that have no MusicXML slur element."""

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
    _draw_endpoint,
    _draw_yolo_boxes,
    _endpoint_geometry,
    _find_endpoint_note,
    _font,
    _measure_boundaries_for_page,
    _panel_point,
)


def _parse_review(value: str) -> dict:
    parts = value.split(",")
    if len(parts) not in {7, 9}:
        raise argparse.ArgumentTypeError(
            "review must be "
            "YOLO_LINE,START_PITCH,START_TIME,START_ID,"
            "END_PITCH,END_TIME,END_ID[,START_X_PX,END_X_PX]"
        )
    review = {
        "yolo_line": int(parts[0]),
        "start_pitch": parts[1],
        "start_time": float(parts[2]),
        "start_id": int(parts[3]),
        "end_pitch": parts[4],
        "end_time": float(parts[5]),
        "end_id": int(parts[6]),
    }
    if len(parts) == 9:
        review["start_x_px"] = float(parts[7])
        review["end_x_px"] = float(parts[8])
    return review


def generate_scan_only_check(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    reviews: list[dict],
    output_path: Path,
    output_csv_path: Path,
    page: int = 1,
    symbol_kind: str = "slur",
) -> None:
    scan_image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    xml_page = parse_musicxml_page(xml_path, page)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)

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

    categories = load_categories(notes_json_path)
    yolo_boxes = load_yolo(yolo_path, categories)
    review_geometry = []
    for review in reviews:
        start_note = _find_endpoint_note(
            xml_page["notes"],
            review["start_id"],
            review["start_time"],
            review["start_pitch"],
        )
        end_note = _find_endpoint_note(
            xml_page["notes"],
            review["end_id"],
            review["end_time"],
            review["end_pitch"],
        )
        if (
            symbol_kind == "slur"
            and (start_note["slur_marks"] or end_note["slur_marks"])
        ):
            raise ValueError(
                f"Y{review['yolo_line']} is not scan-only: "
                "an endpoint contains an XML slur mark"
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
        if "start_x_px" in review:
            start_geometry["scan"]["x"] = review["start_x_px"]
        if "end_x_px" in review:
            end_geometry["scan"]["x"] = review["end_x_px"]
        review_geometry.append(
            (review, start_note, end_note, start_geometry, end_geometry)
        )

    panel_width = 520
    panel_height = 390
    header_height = 120
    canvas = Image.new(
        "RGB",
        (30 + len(reviews) * (panel_width + 20), header_height + panel_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 14),
        (
            "Tie endpoint candidates"
            if symbol_kind == "tie"
            else "Scan-priority slur endpoint candidates"
            if symbol_kind == "scan-override"
            else "Scan-only slur endpoint candidates"
        ),
        fill="black",
        font=_font(28, bold=True),
    )
    draw.text(
        (30, 52),
        (
            "MusicXML contains matching <tie> start/stop elements."
            if symbol_kind == "tie"
            else "Scan endpoint overrides the differing MusicXML slur endpoint."
            if symbol_kind == "scan-override"
            else "MusicXML contains the notes, but no corresponding <slur> element."
        ),
        fill="#b35a00",
        font=_font(19, bold=True),
    )
    draw.text(
        (30, 80),
        "Blue = target YOLO box; A/B = image-derived endpoint candidates",
        fill="#1261ff",
        font=_font(17, bold=True),
    )

    csv_rows = []
    for index, (
        review,
        start_note,
        end_note,
        start_geometry,
        end_geometry,
    ) in enumerate(review_geometry):
        panel_x = 30 + index * (panel_width + 20)
        system = scan_systems[start_note["system"] - 1]
        start_staff = (
            system.upper if start_note["staff"] == 1 else system.lower
        )
        end_staff = (
            system.upper if end_note["staff"] == 1 else system.lower
        )
        crop = (
            round(
                min(
                    start_geometry["scan"]["x"],
                    end_geometry["scan"]["x"],
                )
                - 105
            ),
            round(min(start_staff.lines[0], end_staff.lines[0]) - 80),
            round(
                max(
                    start_geometry["scan"]["x"],
                    end_geometry["scan"]["x"],
                )
                + 105
            ),
            round(max(start_staff.lines[-1], end_staff.lines[-1]) + 90),
        )
        crop_image = scan_image.crop(crop)
        scale = min(
            panel_width / crop_image.width,
            (panel_height - 55) / crop_image.height,
        )
        crop_image = crop_image.resize(
            (
                round(crop_image.width * scale),
                round(crop_image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )
        image_y = header_height + 50
        canvas.paste(crop_image, (panel_x, image_y))
        draw.text(
            (panel_x, header_height + 5),
            (
                f"Y{review['yolo_line']}: "
                f"{review['start_pitch']} {review['start_time']:.3f} "
                f"→ {review['end_pitch']} {review['end_time']:.3f}"
            ),
            fill="black",
            font=_font(20, bold=True),
        )
        _draw_yolo_boxes(
            draw,
            yolo_boxes,
            review["yolo_line"],
            scan_image.size,
            crop,
            scale,
            panel_x,
            image_y,
            target_class="tie" if symbol_kind == "tie" else "slur",
        )
        _draw_endpoint(
            draw,
            _panel_point(
                start_geometry["scan"],
                crop,
                scale,
                panel_x,
                image_y,
            ),
            "#00aa45",
            "A",
        )
        _draw_endpoint(
            draw,
            _panel_point(
                end_geometry["scan"],
                crop,
                scale,
                panel_x,
                image_y,
            ),
            "#e53329",
            "B",
        )
        csv_rows.append(
            {
                "yolo_line": review["yolo_line"],
                "class": "slur",
                "start_pitch_candidate": review["start_pitch"],
                "start_meas_candidate": f"{review['start_time']:.3f}",
                "start_note_candidate": review["start_id"],
                "end_pitch_candidate": review["end_pitch"],
                "end_meas_candidate": f"{review['end_time']:.3f}",
                "end_note_candidate": review["end_id"],
                "xml_slur_status": "absent",
                "review_status": "needs_manual_confirmation",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
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
    parser.add_argument("--review", type=_parse_review, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument(
        "--symbol-kind",
        choices=("slur", "tie", "scan-override"),
        default="slur",
    )
    args = parser.parse_args()
    generate_scan_only_check(
        image_path=args.image,
        clean_image_path=args.clean_image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        reviews=args.review,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
        symbol_kind=args.symbol_kind,
    )


if __name__ == "__main__":
    main()
