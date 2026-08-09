"""Draw a two-panel QA image for one cross-system MusicXML slur."""

from __future__ import annotations

import argparse
import csv
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
    _draw_endpoint,
    _draw_yolo_boxes,
    _endpoint_geometry,
    _find_endpoint_note,
    _font,
    _measure_boundaries_for_page,
    _panel_point,
)


def generate_cross_system_check(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    candidate_id: str,
    start_yolo_line: int,
    end_yolo_line: int,
    output_path: Path,
    output_csv_path: Path,
    page: int = 1,
) -> None:
    scan_image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    xml_page = parse_musicxml_page(xml_path, page)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    candidates, issues = build_slur_candidates(xml_page["notes"], bps_notes)
    if issues:
        raise ValueError(f"Unpaired MusicXML slur endpoints: {issues}")
    candidate = next(
        row for row in candidates if row["candidate_id"] == candidate_id
    )
    if candidate["start_system"] == candidate["end_system"]:
        raise ValueError(f"{candidate_id} is not a cross-system slur")

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

    categories = load_categories(notes_json_path)
    yolo_boxes = load_yolo(yolo_path, categories)
    for line in (start_yolo_line, end_yolo_line):
        box = next(box for box in yolo_boxes if box["txt_line"] == line)
        if box["class"] != "slur":
            raise ValueError(f"YOLO line {line} is not a slur")

    start_system = scan_systems[start_note["system"] - 1]
    end_system = scan_systems[end_note["system"] - 1]
    start_crop = (
        max(0, round(start_geometry["scan"]["x"] - 180)),
        max(0, round(start_system.upper.lines[0] - 105)),
        min(scan_image.width, round(start_system.x_right + 75)),
        min(scan_image.height, round(start_system.lower.lines[-1] + 90)),
    )
    end_crop = (
        max(0, round(end_system.x_left - 75)),
        max(0, round(end_system.upper.lines[0] - 105)),
        min(scan_image.width, round(end_geometry["scan"]["x"] + 180)),
        min(scan_image.height, round(end_system.lower.lines[-1] + 90)),
    )

    panel_width = 650
    max_image_height = 500
    panel_y = 165
    canvas = Image.new("RGB", (1340, 700), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 14),
        f"{candidate_id} cross-system slur endpoint check",
        fill="black",
        font=_font(29, True),
    )
    draw.text(
        (30, 52),
        (
            f"A = {candidate['start_pitch']} start "
            f"{candidate['start_meas']} (#{candidate['start_note_candidate']})"
            f"   →   B = {candidate['end_pitch']} stop "
            f"{candidate['end_meas']} (#{candidate['end_note_candidate']})"
        ),
        fill="#333333",
        font=_font(20, True),
    )
    draw.text(
        (30, 86),
        (
            "Blue = target YOLO segment; gray = nearby slurs. "
            "The two blue boxes represent one XML slur across a system break."
        ),
        fill="#1261ff",
        font=_font(18, True),
    )

    panel_data = [
        (
            "System 1 end — Y26 / start A",
            start_crop,
            start_yolo_line,
            start_geometry["scan"],
            "#00aa45",
            "A",
            20,
        ),
        (
            "System 2 start — Y28 / stop B",
            end_crop,
            end_yolo_line,
            end_geometry["scan"],
            "#e53329",
            "B",
            680,
        ),
    ]
    for title, crop, target_line, point, color, label, panel_x in panel_data:
        draw.text(
            (panel_x, 130),
            title,
            fill="black",
            font=_font(21, True),
        )
        crop_image = scan_image.crop(crop)
        scale = min(
            panel_width / crop_image.width,
            max_image_height / crop_image.height,
        )
        crop_image = crop_image.resize(
            (
                round(crop_image.width * scale),
                round(crop_image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(crop_image, (panel_x, panel_y))
        _draw_yolo_boxes(
            draw,
            yolo_boxes,
            target_line,
            scan_image.size,
            crop,
            scale,
            panel_x,
            panel_y,
        )
        _draw_endpoint(
            draw,
            _panel_point(
                point,
                crop,
                scale,
                panel_x,
                panel_y,
            ),
            color,
            label,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    rows = [
        {
            "candidate_id": candidate_id,
            "segment": "start",
            "yolo_line": start_yolo_line,
            "pitch": candidate["start_pitch"],
            "bps_time": candidate["start_meas"],
            "note_candidate": candidate["start_note_candidate"],
            "scan_x": start_geometry["scan"]["x"],
            "scan_y": f"{start_geometry['scan']['y']:.1f}",
            "review_status": "needs_manual_confirmation",
        },
        {
            "candidate_id": candidate_id,
            "segment": "end",
            "yolo_line": end_yolo_line,
            "pitch": candidate["end_pitch"],
            "bps_time": candidate["end_meas"],
            "note_candidate": candidate["end_note_candidate"],
            "scan_x": end_geometry["scan"]["x"],
            "scan_y": f"{end_geometry['scan']['y']:.1f}",
            "review_status": "needs_manual_confirmation",
        },
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
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
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--start-yolo-line", type=int, required=True)
    parser.add_argument("--end-yolo-line", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    generate_cross_system_check(
        image_path=args.image,
        clean_image_path=args.clean_image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        candidate_id=args.candidate,
        start_yolo_line=args.start_yolo_line,
        end_yolo_line=args.end_yolo_line,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
    )


if __name__ == "__main__":
    main()
