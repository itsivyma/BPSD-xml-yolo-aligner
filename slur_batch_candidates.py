"""Rank MusicXML slur candidates for every YOLO slur on one score page.

The output is review-only.  It never updates the formal BPS-OMR CSV and never
turns a heuristic proposal into a confirmed mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
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


DETAIL_FIELDS = [
    "page_id",
    "yolo_line",
    "rank",
    "box_system",
    "candidate_id",
    "candidate_segment",
    "candidate_start_system",
    "candidate_end_system",
    "start_meas",
    "end_meas",
    "start_pitch",
    "end_pitch",
    "start_note_candidate",
    "end_note_candidate",
    "score",
    "x_center_score",
    "width_score",
    "vertical_score",
    "orientation_score",
    "x_center_error_px",
    "box_width_px",
    "predicted_span_px",
    "vertical_error_px",
    "mutual_best",
]

SUMMARY_FIELDS = [
    "page_id",
    "yolo_line",
    "class_id",
    "class",
    "box_system",
    "x",
    "y",
    "w",
    "h",
    "best_candidate_id",
    "best_candidate_segment",
    "best_score",
    "second_score",
    "margin",
    "start_meas_candidate",
    "end_meas_candidate",
    "start_pitch_candidate",
    "end_pitch_candidate",
    "start_note_candidate",
    "end_note_candidate",
    "status",
    "reason",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reviews(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _box_pixels(box: dict, image: Image.Image) -> dict:
    return {
        "center_x": box["x"] * image.width,
        "center_y": box["y"] * image.height,
        "width": box["w"] * image.width,
        "height": box["h"] * image.height,
    }


def assign_box_system(center_y: float, systems: list) -> int:
    """Assign a glyph to the nearest piano system vertical region."""

    scored = []
    for system in systems:
        spacing = (
            system.upper.line_spacing + system.lower.line_spacing
        ) / 2
        top = system.upper.lines[0] - 5 * spacing
        bottom = system.lower.lines[-1] + 5 * spacing
        if top <= center_y <= bottom:
            outside_distance = 0.0
        else:
            outside_distance = min(
                abs(center_y - top),
                abs(center_y - bottom),
            )
        center_distance = abs(center_y - system.center)
        scored.append((outside_distance, center_distance, system.number))
    return min(scored)[2]


def score_box_against_segment(
    box_pixels: dict,
    segment: dict,
) -> dict:
    """Return interpretable geometric compatibility features in [0, 1]."""

    predicted_center_x = (segment["x0"] + segment["x1"]) / 2
    predicted_span = max(8.0, abs(segment["x1"] - segment["x0"]))
    x_error = abs(box_pixels["center_x"] - predicted_center_x)
    x_tolerance = max(18.0, predicted_span * 0.35)
    x_center_score = math.exp(
        -0.5 * (x_error / x_tolerance) ** 2
    )

    width_ratio = max(0.05, box_pixels["width"] / predicted_span)
    width_score = math.exp(-abs(math.log(width_ratio)) / 0.65)

    spacing = max(1.0, segment["staff_spacing"])
    if segment["orientation"] == "under":
        note_side_y = max(segment["y0"], segment["y1"])
        expected_y = note_side_y + min(
            max(spacing * 1.1, predicted_span * 0.38),
            spacing * 3.5,
        )
        wrong_side = max(0.0, note_side_y - box_pixels["center_y"])
    else:
        note_side_y = min(segment["y0"], segment["y1"])
        expected_y = note_side_y - min(
            max(spacing * 1.1, predicted_span * 0.38),
            spacing * 3.5,
        )
        wrong_side = max(0.0, box_pixels["center_y"] - note_side_y)

    vertical_error = abs(box_pixels["center_y"] - expected_y)
    vertical_score = math.exp(
        -0.5 * (vertical_error / (spacing * 2.0)) ** 2
    )
    orientation_score = math.exp(-wrong_side / spacing)

    total = (
        0.42 * x_center_score
        + 0.28 * width_score
        + 0.23 * vertical_score
        + 0.07 * orientation_score
    )
    return {
        "score": total,
        "x_center_score": x_center_score,
        "width_score": width_score,
        "vertical_score": vertical_score,
        "orientation_score": orientation_score,
        "x_center_error_px": x_error,
        "box_width_px": box_pixels["width"],
        "predicted_span_px": predicted_span,
        "vertical_error_px": vertical_error,
    }


def classify_proposal(
    *,
    best_score: float,
    margin: float,
    mutual_best: bool,
    segment_type: str,
) -> tuple[str, str]:
    """Apply conservative abstention thresholds to a best candidate."""

    if best_score < 0.42:
        return (
            "possible_scan_only",
            "No XML candidate has sufficient geometric agreement.",
        )
    if (
        best_score >= 0.82
        and margin >= 0.12
        and mutual_best
        and segment_type == "full"
    ):
        return (
            "high_confidence_candidate",
            "Mutual best full-system match with score and margin above "
            "the conservative thresholds.",
        )
    reasons = []
    if best_score < 0.82:
        reasons.append("score below 0.82")
    if margin < 0.12:
        reasons.append("top-two margin below 0.12")
    if not mutual_best:
        reasons.append("candidate segment prefers another YOLO box")
    if segment_type != "full":
        reasons.append("cross-system slur segment requires review")
    return "needs_review", "; ".join(reasons)


def _candidate_segments(
    *,
    candidates: list[dict],
    xml_notes: list[dict],
    clean_image: Image.Image,
    scan_image: Image.Image,
    clean_systems: list,
    scan_systems: list,
    clean_boundaries: list[list[int]],
    scan_boundaries: list[list[dict]],
    first_measure_by_system: dict[int, int],
) -> list[dict]:
    segments = []
    for candidate in candidates:
        start_note = _find_endpoint_note(
            xml_notes,
            int(candidate["start_note_candidate"]),
            float(candidate["start_meas"]),
            candidate["start_pitch"],
        )
        end_note = _find_endpoint_note(
            xml_notes,
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
        start_point = start_geometry["scan"]
        end_point = end_geometry["scan"]
        orientation = candidate["orientation"] or "over"

        if start_note["system"] == end_note["system"]:
            system = scan_systems[start_note["system"] - 1]
            staff_spacing = (
                (
                    system.upper
                    if start_note["staff"] == 1
                    else system.lower
                ).line_spacing
                + (
                    system.upper
                    if end_note["staff"] == 1
                    else system.lower
                ).line_spacing
            ) / 2
            segments.append(
                {
                    "key": f"{candidate['candidate_id']}:full",
                    "candidate": candidate,
                    "segment_type": "full",
                    "system": start_note["system"],
                    "x0": start_point["x"],
                    "y0": start_point["y"],
                    "x1": end_point["x"],
                    "y1": end_point["y"],
                    "staff_spacing": staff_spacing,
                    "orientation": orientation,
                }
            )
        else:
            start_system = scan_systems[start_note["system"] - 1]
            start_staff = (
                start_system.upper
                if start_note["staff"] == 1
                else start_system.lower
            )
            segments.append(
                {
                    "key": f"{candidate['candidate_id']}:start",
                    "candidate": candidate,
                    "segment_type": "start",
                    "system": start_note["system"],
                    "x0": start_point["x"],
                    "y0": start_point["y"],
                    "x1": start_system.x_right,
                    "y1": start_point["y"],
                    "staff_spacing": start_staff.line_spacing,
                    "orientation": orientation,
                }
            )
            end_system = scan_systems[end_note["system"] - 1]
            end_staff = (
                end_system.upper
                if end_note["staff"] == 1
                else end_system.lower
            )
            segments.append(
                {
                    "key": f"{candidate['candidate_id']}:end",
                    "candidate": candidate,
                    "segment_type": "end",
                    "system": end_note["system"],
                    "x0": end_system.x_left,
                    "y0": end_point["y"],
                    "x1": end_point["x"],
                    "y1": end_point["y"],
                    "staff_spacing": end_staff.line_spacing,
                    "orientation": orientation,
                }
            )
    return segments


def _candidate_summary_values(row: dict) -> dict:
    return {
        "best_candidate_id": row["candidate_id"],
        "best_candidate_segment": row["candidate_segment"],
        "start_meas_candidate": row["start_meas"],
        "end_meas_candidate": row["end_meas"],
        "start_pitch_candidate": row["start_pitch"],
        "end_pitch_candidate": row["end_pitch"],
        "start_note_candidate": row["start_note_candidate"],
        "end_note_candidate": row["end_note_candidate"],
    }


def _draw_overlay(
    image: Image.Image,
    boxes: list[dict],
    summary_by_line: dict[int, dict],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = {
        "locked_xml_match": "#00a33d",
        "locked_scan_only": "#8d3fd1",
        "high_confidence_candidate": "#00a8a8",
        "needs_review": "#f08a00",
        "possible_scan_only": "#d93676",
    }
    for box in boxes:
        if box["class"] != "slur":
            continue
        summary = summary_by_line[box["txt_line"]]
        color = colors[summary["status"]]
        x0 = (box["x"] - box["w"] / 2) * image.width
        y0 = (box["y"] - box["h"] / 2) * image.height
        x1 = (box["x"] + box["w"] / 2) * image.width
        y1 = (box["y"] + box["h"] / 2) * image.height
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        candidate = summary["best_candidate_id"] or "scan"
        score = (
            ""
            if summary["best_score"] == ""
            else f" {float(summary['best_score']):.2f}"
        )
        label = f"Y{box['txt_line']}→{candidate}{score}"
        draw.text(
            (x0, y0 - 26),
            label,
            fill=color,
            font=_font(20, True),
            stroke_width=2,
            stroke_fill="white",
        )

    legend = [
        ("locked_xml_match", "confirmed XML match"),
        ("locked_scan_only", "confirmed scan-only"),
        ("high_confidence_candidate", "high-confidence candidate"),
        ("needs_review", "needs review"),
        ("possible_scan_only", "possible scan-only"),
    ]
    legend_x = 35
    legend_y = 35
    draw.rounded_rectangle(
        (20, 20, 570, 220),
        radius=12,
        fill="white",
        outline="black",
        width=2,
    )
    draw.text(
        (legend_x, legend_y),
        "First-page slur batch candidates",
        fill="black",
        font=_font(25, True),
    )
    for index, (status, label) in enumerate(legend):
        y = legend_y + 42 + index * 27
        draw.rectangle(
            (legend_x, y + 3, legend_x + 20, y + 20),
            outline=colors[status],
            width=4,
        )
        draw.text(
            (legend_x + 30, y),
            label,
            fill=colors[status],
            font=_font(18, True),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_batch(
    *,
    image_path: Path,
    clean_image_path: Path,
    yolo_path: Path,
    notes_json_path: Path,
    xml_path: Path,
    bps_notes_path: Path,
    reviews_path: Path,
    output_dir: Path,
    page: int = 1,
) -> dict:
    scan_image = Image.open(image_path).convert("RGB")
    clean_image = Image.open(clean_image_path).convert("RGB")
    categories = load_categories(notes_json_path)
    boxes = load_yolo(yolo_path, categories)
    slur_boxes = [box for box in boxes if box["class"] == "slur"]

    xml_page = parse_musicxml_page(xml_path, page)
    bps_notes = load_bps_notes(bps_notes_path)
    attach_bps_note_ids(xml_page["notes"], bps_notes)
    candidates, issues = build_slur_candidates(xml_page["notes"], bps_notes)
    if issues:
        raise ValueError(f"Unpaired MusicXML slur endpoints: {issues}")

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
    segments = _candidate_segments(
        candidates=candidates,
        xml_notes=xml_page["notes"],
        clean_image=clean_image,
        scan_image=scan_image,
        clean_systems=clean_systems,
        scan_systems=scan_systems,
        clean_boundaries=clean_boundaries,
        scan_boundaries=scan_boundaries,
        first_measure_by_system=first_measure_by_system,
    )
    reviews = _load_reviews(reviews_path)
    review_by_line = {int(row["yolo_line"]): row for row in reviews}
    locked_candidate_keys = {
        f"{row['xml_candidate_id']}:full"
        for row in reviews
        if row["match_status"] == "matched"
    }
    available_segments = [
        segment
        for segment in segments
        if segment["key"] not in locked_candidate_keys
    ]

    rankings: dict[int, list[dict]] = {}
    for box in slur_boxes:
        if box["txt_line"] in review_by_line:
            continue
        pixels = _box_pixels(box, scan_image)
        box_system = assign_box_system(pixels["center_y"], scan_systems)
        ranked = []
        for segment in available_segments:
            if segment["system"] != box_system:
                continue
            features = score_box_against_segment(pixels, segment)
            candidate = segment["candidate"]
            ranked.append(
                {
                    "page_id": image_path.stem,
                    "yolo_line": box["txt_line"],
                    "box_system": box_system,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_segment": segment["segment_type"],
                    "segment_key": segment["key"],
                    "candidate_start_system": candidate["start_system"],
                    "candidate_end_system": candidate["end_system"],
                    "start_meas": candidate["start_meas"],
                    "end_meas": candidate["end_meas"],
                    "start_pitch": candidate["start_pitch"],
                    "end_pitch": candidate["end_pitch"],
                    "start_note_candidate": candidate[
                        "start_note_candidate"
                    ],
                    "end_note_candidate": candidate["end_note_candidate"],
                    **features,
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        rankings[box["txt_line"]] = ranked

    segment_best_yolo: dict[str, int] = {}
    for segment in available_segments:
        options = [
            rows_item
            for rows in rankings.values()
            for rows_item in rows
            if rows_item["segment_key"] == segment["key"]
        ]
        if options:
            best = max(options, key=lambda row: row["score"])
            segment_best_yolo[segment["key"]] = int(best["yolo_line"])

    detail_rows = []
    summary_rows = []
    summary_by_line = {}
    candidate_by_id = {
        row["candidate_id"]: row for row in candidates
    }
    for box in slur_boxes:
        pixels = _box_pixels(box, scan_image)
        box_system = assign_box_system(pixels["center_y"], scan_systems)
        base_summary = {
            "page_id": image_path.stem,
            "yolo_line": box["txt_line"],
            "class_id": box["class_id"],
            "class": box["class"],
            "box_system": box_system,
            "x": f"{box['x']:.6f}",
            "y": f"{box['y']:.6f}",
            "w": f"{box['w']:.6f}",
            "h": f"{box['h']:.6f}",
        }
        review = review_by_line.get(box["txt_line"])
        if review is not None:
            if review["match_status"] == "matched":
                candidate = candidate_by_id[review["xml_candidate_id"]]
                summary = {
                    **base_summary,
                    "best_candidate_id": candidate["candidate_id"],
                    "best_candidate_segment": "full",
                    "best_score": "",
                    "second_score": "",
                    "margin": "",
                    "start_meas_candidate": candidate["start_meas"],
                    "end_meas_candidate": candidate["end_meas"],
                    "start_pitch_candidate": candidate["start_pitch"],
                    "end_pitch_candidate": candidate["end_pitch"],
                    "start_note_candidate": review["start_note_id"],
                    "end_note_candidate": review["end_note_id"],
                    "status": "locked_xml_match",
                    "reason": "Manually confirmed; excluded from ranking.",
                }
            else:
                summary = {
                    **base_summary,
                    "best_candidate_id": "",
                    "best_candidate_segment": "",
                    "best_score": "",
                    "second_score": "",
                    "margin": "",
                    "start_meas_candidate": "",
                    "end_meas_candidate": "",
                    "start_pitch_candidate": "",
                    "end_pitch_candidate": "",
                    "start_note_candidate": review["start_note_id"],
                    "end_note_candidate": review["end_note_id"],
                    "status": "locked_scan_only",
                    "reason": (
                        "Manually confirmed endpoints; MusicXML has no "
                        "corresponding slur."
                    ),
                }
            summary_rows.append(summary)
            summary_by_line[box["txt_line"]] = summary
            continue

        ranked = rankings[box["txt_line"]]
        if not ranked:
            summary = {
                **base_summary,
                "best_candidate_id": "",
                "best_candidate_segment": "",
                "best_score": "",
                "second_score": "",
                "margin": "",
                "start_meas_candidate": "",
                "end_meas_candidate": "",
                "start_pitch_candidate": "",
                "end_pitch_candidate": "",
                "start_note_candidate": "",
                "end_note_candidate": "",
                "status": "possible_scan_only",
                "reason": "No XML slur segment exists in this system.",
            }
            summary_rows.append(summary)
            summary_by_line[box["txt_line"]] = summary
            continue

        for rank, row in enumerate(ranked[:3], start=1):
            mutual = segment_best_yolo.get(row["segment_key"]) == box["txt_line"]
            detail_rows.append(
                {
                    field: (
                        rank
                        if field == "rank"
                        else (
                            f"{row[field]:.4f}"
                            if field
                            in {
                                "score",
                                "x_center_score",
                                "width_score",
                                "vertical_score",
                                "orientation_score",
                                "x_center_error_px",
                                "box_width_px",
                                "predicted_span_px",
                                "vertical_error_px",
                            }
                            else (
                                "yes"
                                if field == "mutual_best" and mutual
                                else "no"
                                if field == "mutual_best"
                                else row[field]
                            )
                        )
                    )
                    for field in DETAIL_FIELDS
                }
            )

        best = ranked[0]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = best["score"] - second_score
        mutual = (
            segment_best_yolo.get(best["segment_key"])
            == box["txt_line"]
        )
        status, reason = classify_proposal(
            best_score=best["score"],
            margin=margin,
            mutual_best=mutual,
            segment_type=best["candidate_segment"],
        )
        summary = {
            **base_summary,
            **_candidate_summary_values(best),
            "best_score": f"{best['score']:.4f}",
            "second_score": f"{second_score:.4f}",
            "margin": f"{margin:.4f}",
            "status": status,
            "reason": reason,
        }
        summary_rows.append(summary)
        summary_by_line[box["txt_line"]] = summary

    summary_rows.sort(key=lambda row: int(row["yolo_line"]))
    detail_rows.sort(
        key=lambda row: (int(row["yolo_line"]), int(row["rank"]))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{image_path.stem}_slur_batch_summary.csv"
    detail_path = output_dir / f"{image_path.stem}_slur_batch_top3.csv"
    overlay_path = output_dir / f"{image_path.stem}_slur_batch_overlay.png"
    report_path = output_dir / f"{image_path.stem}_slur_batch_report.json"
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(detail_path, DETAIL_FIELDS, detail_rows)
    _draw_overlay(scan_image, slur_boxes, summary_by_line, overlay_path)

    status_counts = Counter(row["status"] for row in summary_rows)
    report = {
        "page_id": image_path.stem,
        "yolo_slurs": len(slur_boxes),
        "xml_slur_pairs": len(candidates),
        "xml_rendered_segments": len(segments),
        "confirmed_reviews": len(reviews),
        "status_counts": dict(sorted(status_counts.items())),
        "thresholds": {
            "possible_scan_only_below": 0.42,
            "high_confidence_at_least": 0.82,
            "high_confidence_margin_at_least": 0.12,
            "high_confidence_requires_mutual_best": True,
            "cross_system_segments_require_review": True,
        },
        "outputs": {
            "summary_csv": str(summary_path),
            "top3_csv": str(detail_path),
            "overlay_png": str(overlay_path),
        },
        "warning": (
            "All non-locked matches are heuristic proposals and require "
            "review before entering the training table or formal BPS-OMR CSV."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--clean-image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    report = run_batch(
        image_path=args.image,
        clean_image_path=args.clean_image,
        yolo_path=args.yolo,
        notes_json_path=args.notes_json,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        page=args.page,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
