"""Generate one review image for a MusicXML slur and one YOLO slur box.

This is deliberately a QA-only helper.  It does not update the formal
BPS-OMR CSV or confirm a MusicXML/YOLO match.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bps_xml_alignment import (
    align_barlines_from_reference,
    attach_bps_note_ids,
    build_slur_candidates,
    detect_barlines,
    detect_systems,
    load_bps_notes,
    load_categories,
    load_yolo,
    note_pixel_position,
    parse_musicxml_page,
    snap_notehead_x,
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _find_endpoint_note(
    notes: list[dict],
    note_id: int,
    bps_time: float,
    pitch: str,
) -> dict:
    matches = [
        note
        for note in notes
        if note.get("note_id") == note_id
        and abs(note["bps_time"] - bps_time) < 0.001
        and note["pitch_name"] == pitch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one endpoint note for id={note_id}, "
            f"time={bps_time:.3f}, pitch={pitch}; found {len(matches)}"
        )
    return matches[0]


def _measure_boundaries_for_page(
    image: Image.Image,
    xml_page: dict,
) -> tuple[list, list[list[int]]]:
    systems = detect_systems(image)
    measures_per_system = Counter(
        measure["system"] for measure in xml_page["measures"]
    )
    expected_counts = [
        measures_per_system[system.number] + 1 for system in systems
    ]
    return systems, detect_barlines(image, systems, expected_counts)


def _endpoint_geometry(
    note: dict,
    clean_image: Image.Image,
    scan_image: Image.Image,
    clean_systems: list,
    scan_systems: list,
    clean_boundaries: list[list[int]],
    scan_boundaries: list[list[dict]],
    first_measure_by_system: dict[int, int],
) -> dict:
    system_index = note["system"] - 1
    clean_system = clean_systems[system_index]
    scan_system = scan_systems[system_index]
    clean_x, clean_y = note_pixel_position(note, clean_system)
    _unused_scan_x, scan_y = note_pixel_position(note, scan_system)

    measure_index = (
        note["xml_measure"] - first_measure_by_system[note["system"]]
    )
    clean_left = clean_boundaries[system_index][measure_index]
    clean_right = clean_boundaries[system_index][measure_index + 1]
    scan_left = scan_boundaries[system_index][measure_index]["x"]
    scan_right = scan_boundaries[system_index][measure_index + 1]["x"]
    within_measure = (clean_x - clean_left) / (clean_right - clean_left)
    scan_x = scan_left + within_measure * (scan_right - scan_left)

    clean_snap = snap_notehead_x(
        clean_image,
        clean_x,
        clean_y,
        clean_system.upper if note["staff"] == 1 else clean_system.lower,
        search_radius=24,
    )
    scan_snap = snap_notehead_x(
        scan_image,
        scan_x,
        scan_y,
        scan_system.upper if note["staff"] == 1 else scan_system.lower,
        # Keep the search local to the predicted onset.  A wider radius can
        # jump to the same pitch at the preceding onset in dense piano music.
        search_radius=24,
    )
    return {
        "clean_left": clean_left,
        "clean_right": clean_right,
        "scan_left": scan_left,
        "scan_right": scan_right,
        "clean_rough_x": clean_x,
        "scan_rough_x": scan_x,
        "clean": clean_snap,
        "scan": scan_snap,
    }


def _crop_and_scale(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    target_width: int,
) -> tuple[Image.Image, float]:
    result = image.crop(crop)
    scale = target_width / result.width
    result = result.resize(
        (target_width, round(result.height * scale)),
        Image.Resampling.LANCZOS,
    )
    return result, scale


def _panel_point(
    point: dict,
    crop: tuple[int, int, int, int],
    scale: float,
    panel_x: int,
    panel_y: int,
) -> tuple[float, float]:
    return (
        panel_x + (point["x"] - crop[0]) * scale,
        panel_y + (point["y"] - crop[1]) * scale,
    )


def _draw_endpoint(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    color: str,
    label: str,
) -> None:
    x, y = point
    radius = 19
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=7,
    )
    draw.text(
        (x - 12, y - 48),
        label,
        fill=color,
        font=_font(29, bold=True),
        stroke_width=2,
        stroke_fill="white",
    )


def _draw_yolo_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: list[dict],
    target_line: int,
    image_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    scale: float,
    panel_x: int,
    panel_y: int,
    target_class: str = "slur",
) -> None:
    image_width, image_height = image_size
    for box in boxes:
        if box["class"] != target_class:
            continue
        x0 = (box["x"] - box["w"] / 2) * image_width
        y0 = (box["y"] - box["h"] / 2) * image_height
        x1 = (box["x"] + box["w"] / 2) * image_width
        y1 = (box["y"] + box["h"] / 2) * image_height
        if x1 < crop[0] or x0 > crop[2] or y1 < crop[1] or y0 > crop[3]:
            continue
        color = "#1261ff" if box["txt_line"] == target_line else "#888888"
        width = 2 if box["txt_line"] == target_line else 1
        draw.rectangle(
            (
                panel_x + (x0 - crop[0]) * scale,
                panel_y + (y0 - crop[1]) * scale,
                panel_x + (x1 - crop[0]) * scale,
                panel_y + (y1 - crop[1]) * scale,
            ),
            outline=color,
            width=width,
        )
        draw.text(
            (
                panel_x + (x0 - crop[0]) * scale,
                panel_y + (y0 - crop[1]) * scale - 30,
            ),
            f"Y{box['txt_line']}",
            fill=color,
            font=_font(25, bold=True),
            stroke_width=2,
            stroke_fill="white",
        )


def generate_check(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    candidate_id: str,
    yolo_line: int,
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
        raise ValueError(f"MusicXML contains unpaired slur endpoints: {issues}")
    candidate = next(
        (row for row in candidates if row["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Unknown slur candidate: {candidate_id}")

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

    clean_system = clean_systems[start_note["system"] - 1]
    scan_system = scan_systems[start_note["system"] - 1]
    clean_crop = (
        round(min(start_geometry["clean_left"], end_geometry["clean_left"]) - 50),
        round(clean_system.upper.lines[0] - 115),
        round(max(start_geometry["clean_right"], end_geometry["clean_right"]) + 50),
        round(clean_system.lower.lines[-1] + 75),
    )
    scan_crop = (
        round(min(start_geometry["scan_left"], end_geometry["scan_left"]) - 75),
        round(scan_system.upper.lines[0] - 145),
        round(max(start_geometry["scan_right"], end_geometry["scan_right"]) + 75),
        round(scan_system.lower.lines[-1] + 105),
    )

    panel_width = 505
    clean_panel, clean_scale = _crop_and_scale(
        clean_image,
        clean_crop,
        panel_width,
    )
    scan_panel, scan_scale = _crop_and_scale(
        scan_image,
        scan_crop,
        panel_width,
    )
    panel_y = 145
    left_x = 30
    right_x = 555
    canvas_height = max(clean_panel.height, scan_panel.height) + panel_y + 25
    canvas = Image.new("RGB", (1090, canvas_height), "white")
    canvas.paste(clean_panel, (left_x, panel_y))
    canvas.paste(scan_panel, (right_x, panel_y))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (30, 14),
        f"{candidate_id} / Y{yolo_line} endpoint check",
        fill="black",
        font=_font(28, bold=True),
    )
    draw.text(
        (30, 52),
        (
            f"A = {candidate['start_pitch']} start "
            f"{candidate['start_meas']}"
        ),
        fill="#00aa45",
        font=_font(21, bold=True),
    )
    draw.text(
        (350, 52),
        (
            f"B = {candidate['end_pitch']} stop "
            f"{candidate['end_meas']}"
        ),
        fill="#e53329",
        font=_font(21, bold=True),
    )
    draw.text(
        (30, 82),
        (
            "Blue = target YOLO slur; gray = other slurs; "
            "A/B = XML-selected chord tones"
        ),
        fill="#1261ff",
        font=_font(18, bold=True),
    )
    draw.text(
        (left_x, 112),
        "Sibelius / MusicXML",
        fill="black",
        font=_font(20, bold=True),
    )
    draw.text(
        (right_x, 112),
        "Scan + YOLO",
        fill="black",
        font=_font(20, bold=True),
    )

    categories = load_categories(notes_json_path)
    yolo_boxes = load_yolo(yolo_path, categories)
    target_boxes = [
        box for box in yolo_boxes if box["txt_line"] == yolo_line
    ]
    if len(target_boxes) != 1 or target_boxes[0]["class"] != "slur":
        raise ValueError(f"YOLO line {yolo_line} is not one slur box")
    _draw_yolo_boxes(
        draw,
        yolo_boxes,
        yolo_line,
        scan_image.size,
        scan_crop,
        scan_scale,
        right_x,
        panel_y,
    )

    for geometry, color, label in (
        (start_geometry, "#00aa45", "A"),
        (end_geometry, "#e53329", "B"),
    ):
        _draw_endpoint(
            draw,
            _panel_point(
                geometry["clean"],
                clean_crop,
                clean_scale,
                left_x,
                panel_y,
            ),
            color,
            label,
        )
        _draw_endpoint(
            draw,
            _panel_point(
                geometry["scan"],
                scan_crop,
                scan_scale,
                right_x,
                panel_y,
            ),
            color,
            label,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    rows = []
    for version, geometry in (
        ("sibelius", start_geometry),
        ("scan", start_geometry),
        ("sibelius", end_geometry),
        ("scan", end_geometry),
    ):
        role = "start" if geometry is start_geometry else "end"
        note = start_note if role == "start" else end_note
        point = geometry["clean"] if version == "sibelius" else geometry["scan"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "yolo_line": yolo_line,
                "version": version,
                "role": role,
                "pitch": note["pitch_name"],
                "bps_time": f"{note['bps_time']:.3f}",
                "snapped_x": point["x"],
                "pitch_y": f"{point['y']:.1f}",
                "ink_count": point["ink_count"],
            }
        )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw one MusicXML-slur/YOLO-slur endpoint review image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clean-image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--yolo-line", type=int, required=True)
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
        candidate_id=args.candidate,
        yolo_line=args.yolo_line,
        output_path=args.output,
        output_csv_path=args.output_csv,
        page=args.page,
    )


if __name__ == "__main__":
    main()
