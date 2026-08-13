"""Normalize human ground truth and apply website review decisions."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_checkpoint import atomic_write_csv, atomic_write_json, emit_progress


REVIEW_ACTIONS = (
    "pending",
    "confirm",
    "correct",
    "scan_only",
    "wrong_class",
    "bad_bbox",
    "not_a_symbol",
    "uncertain",
    "skipped",
    "reject",
)
SEMANTIC_FIELDS = (
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
)
EVALUATION_FIELDS = (*SEMANTIC_FIELDS, "staff")
EDITOR_FIELDS = (
    "action",
    "page_id",
    "yolo_line",
    "class",
    "corrected_class_id",
    "corrected_class",
    "machine_status",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "staff",
    "comment",
)
GROUND_TRUTH_FIELDS = (
    "source_file",
    "page_id",
    "yolo_line",
    "class_id",
    "class",
    "review_status",
    "expected_start_meas",
    "expected_end_meas",
    "expected_start_note",
    "expected_end_note",
    "expected_connected_note",
    "expected_staff",
    "comment",
)


def _text(value: object) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value.upper() == "NA" else value


def _first(row: dict, *fields: str) -> str:
    for field in fields:
        value = _text(row.get(field, ""))
        if value:
            return value
    return ""


def _split_ids(value: object) -> list[str]:
    text = _text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [_text(item) for item in parsed if _text(item)]
    return [item for item in text.replace(",", "+").split("+") if item]


def _equivalent(field: str, actual: object, expected: object) -> bool:
    actual_text = _text(actual)
    expected_text = _text(expected)
    if field in {"start_meas", "end_meas"} and actual_text and expected_text:
        try:
            return abs(float(actual_text) - float(expected_text)) <= 0.0005
        except ValueError:
            pass
    if field == "connected_note":
        return _split_ids(actual_text) == _split_ids(expected_text)
    return actual_text == expected_text


def normalize_legacy_review_rows(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Convert heterogeneous class-specific human CSVs to one ground truth."""

    output: list[dict] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        for row_number, row in enumerate(rows, start=2):
            page_id = _text(row.get("page_id"))
            yolo_line = _text(row.get("yolo_line"))
            if not page_id or not yolo_line:
                errors.append(f"{path.name}:{row_number} missing page_id/yolo_line")
                continue
            key = (page_id, yolo_line)
            if key in seen:
                errors.append(f"duplicate ground-truth key {page_id}:Y{yolo_line}")
                continue
            seen.add(key)
            note_ids = _split_ids(row.get("note_ids"))
            start_note = _first(row, "start_note_id", "note_id")
            end_note = _first(row, "end_note_id", "note_id")
            if not start_note and note_ids:
                start_note = note_ids[0]
            if not end_note and note_ids:
                end_note = note_ids[-1]
            if not note_ids:
                note_ids = list(dict.fromkeys([item for item in (start_note, end_note) if item]))
            connected = json.dumps(note_ids, ensure_ascii=False) if note_ids else ""
            output.append(
                {
                    "source_file": path.name,
                    "page_id": page_id,
                    "yolo_line": yolo_line,
                    "class_id": _text(row.get("class_id")),
                    "class": _text(row.get("class")),
                    "review_status": _first(row, "review_status", "candidate_status"),
                    "expected_start_meas": _first(
                        row,
                        "corrected_start_meas",
                        "candidate_start_meas",
                        "bps_time",
                        "start_bps_time",
                    ),
                    "expected_end_meas": _first(
                        row,
                        "corrected_end_meas",
                        "candidate_end_meas",
                        "bps_time",
                        "end_note_bps_time",
                    ),
                    "expected_start_note": start_note,
                    "expected_end_note": end_note,
                    "expected_connected_note": connected,
                    "expected_staff": _text(row.get("staff")),
                    "comment": _text(row.get("comment")),
                }
            )
    output.sort(key=lambda row: (row["page_id"], int(row["yolo_line"])))
    return output, errors


def build_editor_rows(
    detailed_rows: list[dict], *, include_matched: bool = False
) -> list[dict]:
    """Return stable, editable review rows from alignment details."""

    rows = []
    for source in detailed_rows:
        status = _text(source.get("status")) or "unresolved"
        if status == "matched" and not include_matched:
            continue
        rows.append(
            {
                "action": "pending",
                "page_id": _text(source.get("page_id")),
                "yolo_line": _text(source.get("txt_line")),
                "class": _text(source.get("class")),
                "corrected_class_id": _text(source.get("class_id")),
                "corrected_class": _text(source.get("class")),
                "machine_status": status,
                "start_meas": _text(source.get("start_meas")),
                "end_meas": _text(source.get("end_meas")),
                "start_note": _text(source.get("start_note")),
                "end_note": _text(source.get("end_note")),
                "connected_note": _text(source.get("connected_note")),
                "staff": _text(source.get("xml_staff")),
                "comment": "",
            }
        )
    return rows


def review_row_key(row: dict) -> str:
    """Return a stable session/checkpoint key for one YOLO row."""

    page_id, yolo_line = _key(row)
    return f"{page_id}:Y{yolo_line}"


def build_review_checkpoint(
    decisions: dict[str, dict],
    reviewer: str,
    *,
    alignment_fingerprint: str,
    score_id: str,
    pipeline_version: str,
) -> dict:
    """Build a checkpoint bound to one exact alignment input batch."""

    fingerprint = _text(alignment_fingerprint)
    if not fingerprint:
        raise ValueError("alignment fingerprint is required for a review checkpoint")

    return {
        "schema_version": "2.0",
        "alignment_fingerprint": fingerprint,
        "score_id": _text(score_id),
        "pipeline_version": _text(pipeline_version),
        "reviewer": _text(reviewer) or "User",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            decisions[key] for key in sorted(decisions)
        ],
    }


def load_review_checkpoint(
    payload: dict,
    detailed_rows: list[dict],
    *,
    expected_fingerprint: str,
    expected_score_id: str,
    expected_pipeline_version: str,
) -> tuple[dict[str, dict], list[str]]:
    """Validate and restore portable decisions against the current alignment."""

    expected = _text(expected_fingerprint)
    checkpoint_fingerprint = _text(payload.get("alignment_fingerprint"))
    if not expected:
        return {}, ["current alignment is missing its fingerprint; rerun alignment"]
    if not checkpoint_fingerprint:
        return {}, [
            "review checkpoint is missing alignment_fingerprint and cannot be "
            "safely restored; create a new checkpoint from the current version"
        ]
    if checkpoint_fingerprint != expected:
        return {}, [
            "review checkpoint belongs to a different alignment input batch; "
            "use the same images, YOLO TXT, XML, BPSD CSV, notes.json, and settings"
        ]
    if _text(payload.get("schema_version")) != "2.0":
        return {}, [
            "unsupported review checkpoint schema_version; create a new "
            "checkpoint from the current version"
        ]

    metadata_errors = []
    checkpoint_score_id = _text(payload.get("score_id"))
    expected_score = _text(expected_score_id)
    if checkpoint_score_id != expected_score:
        metadata_errors.append(
            f"review checkpoint score_id mismatch: expected {expected_score or '(blank)'}, "
            f"got {checkpoint_score_id or '(blank)'}"
        )
    checkpoint_version = _text(payload.get("pipeline_version"))
    expected_version = _text(expected_pipeline_version)
    if checkpoint_version != expected_version:
        metadata_errors.append(
            "review checkpoint pipeline version mismatch: "
            f"expected {expected_version or '(blank)'}, "
            f"got {checkpoint_version or '(blank)'}"
        )
    if metadata_errors:
        return {}, metadata_errors

    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}, ["review checkpoint must contain an entries list"]
    available = {review_row_key(row) for row in detailed_rows}
    decisions: dict[str, dict] = {}
    errors: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"checkpoint entry {index} is not an object")
            continue
        action = _text(entry.get("action")) or "pending"
        key = review_row_key(entry)
        if action not in REVIEW_ACTIONS:
            errors.append(f"checkpoint entry {index} has invalid action: {action}")
        elif key not in available:
            errors.append(f"checkpoint row not found in current alignment: {key}")
        elif key in decisions:
            errors.append(f"duplicate checkpoint row: {key}")
        else:
            decisions[key] = dict(entry)
    return decisions, errors


def build_review_queue(
    detailed_rows: list[dict],
    *,
    page_id: str | None = None,
    class_name: str | None = None,
    machine_status: str | None = None,
    include_matched: bool = False,
    decisions: dict[str, dict] | None = None,
) -> list[dict]:
    """Build a risk-first, filterable queue for the item-by-item workspace."""

    decisions = decisions or {}
    queue = []
    for source in detailed_rows:
        status = _text(source.get("status")) or "unresolved"
        if not include_matched and status == "matched":
            continue
        if page_id and _text(source.get("page_id")) != page_id:
            continue
        if class_name and _text(source.get("class")) != class_name:
            continue
        if machine_status and status != machine_status:
            continue
        item = dict(source)
        item["review_key"] = review_row_key(source)
        item["saved_action"] = _text(
            decisions.get(item["review_key"], {}).get("action")
        ) or "pending"
        queue.append(item)

    def confidence(row: dict) -> float:
        try:
            return float(_text(row.get("confidence")))
        except ValueError:
            return -1.0

    queue.sort(
        key=lambda row: (
            row["saved_action"] not in {"pending", "uncertain", "skipped"},
            row["status"] == "matched",
            confidence(row),
            _text(row.get("page_id")),
            int(_text(row.get("txt_line")) or 0),
        )
    )
    return queue


def render_review_focus_images(
    image_data: bytes,
    row: dict,
    *,
    include_crop_geometry: bool = False,
) -> tuple[bytes, bytes] | tuple[bytes, bytes, dict]:
    """Render a full-page highlight and a context crop without resampling blur."""

    with Image.open(io.BytesIO(image_data)) as source:
        image = source.convert("RGB")
    width, height = image.size
    try:
        center_x = float(row.get("x", 0.5)) * width
        center_y = float(row.get("y", 0.5)) * height
        box_width = max(1.0, float(row.get("w", 0.0)) * width)
        box_height = max(1.0, float(row.get("h", 0.0)) * height)
    except (TypeError, ValueError) as error:
        raise ValueError("Review row has invalid normalized bbox values") from error
    left = max(0, round(center_x - box_width / 2))
    top = max(0, round(center_y - box_height / 2))
    right = min(width - 1, round(center_x + box_width / 2))
    bottom = min(height - 1, round(center_y + box_height / 2))
    if right <= left or bottom <= top:
        raise ValueError("Review row bbox falls outside the page image")

    def optional_point(x_field: str, y_field: str) -> tuple[float, float] | None:
        x_text = _text(row.get(x_field))
        y_text = _text(row.get(y_field))
        if not x_text or not y_text:
            return None
        try:
            point_x = float(x_text)
            point_y = float(y_text)
        except ValueError:
            return None
        if not (0 <= point_x < width and 0 <= point_y < height):
            return None
        return point_x, point_y

    start_target = optional_point("target_x_px", "target_y_px")
    end_target = optional_point("end_target_x_px", "end_target_y_px")
    review_target = optional_point("review_target_x_px", "review_target_y_px")
    review_start_target = optional_point(
        "review_start_target_x_px", "review_start_target_y_px"
    )
    review_end_target = optional_point(
        "review_end_target_x_px", "review_end_target_y_px"
    )
    try:
        raw_candidates = json.loads(_text(row.get("review_note_candidates_json")) or "[]")
    except json.JSONDecodeError:
        raw_candidates = []
    review_candidates = []
    if isinstance(raw_candidates, list):
        # Keep the crop legible even though the review form retains a wider
        # set of note choices for start/end-point correction.
        for index, candidate in enumerate(raw_candidates[:8], start=1):
            if not isinstance(candidate, dict):
                continue
            try:
                point = (float(candidate["x_px"]), float(candidate["y_px"]))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= point[0] < width and 0 <= point[1] < height:
                review_candidates.append((point, str(index)))
    if (
        start_target is not None
        and end_target is not None
        and abs(start_target[0] - end_target[0]) < 1
        and abs(start_target[1] - end_target[1]) < 1
    ):
        end_target = None

    full = image.copy()
    full_draw = ImageDraw.Draw(full)
    stroke = max(3, round(min(width, height) * 0.0025))
    full_draw.rectangle((left, top, right, bottom), outline="#e31a1c", width=stroke)

    marker_radius = max(14, round(min(width, height) * 0.012))
    marker_font = ImageFont.load_default()
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            marker_font = ImageFont.truetype(
                font_path, size=max(16, round(marker_radius * 1.25))
            )
            break
        except OSError:
            continue

    def connection_end(
        point: tuple[float, float], *, offset_x: float = 0, offset_y: float = 0
    ) -> tuple[float, float]:
        """Stop the connector at the circle edge so it does not cover the notehead."""

        origin_x = center_x - offset_x
        origin_y = center_y - offset_y
        point_x = point[0] - offset_x
        point_y = point[1] - offset_y
        delta_x = point_x - origin_x
        delta_y = point_y - origin_y
        distance = (delta_x**2 + delta_y**2) ** 0.5
        if distance <= marker_radius:
            return origin_x, origin_y
        scale = (distance - marker_radius) / distance
        return origin_x + delta_x * scale, origin_y + delta_y * scale

    def draw_target(
        drawing: ImageDraw.ImageDraw,
        point: tuple[float, float],
        *,
        color: str,
        label: str,
        offset_x: float = 0,
        offset_y: float = 0,
        radius_scale: float = 1.0,
        target_stroke: int | None = None,
    ) -> None:
        point_x = round(point[0] - offset_x)
        point_y = round(point[1] - offset_y)
        radius = round(marker_radius * radius_scale)
        outline_width = target_stroke or stroke
        drawing.ellipse(
            (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
            outline=color,
            width=outline_width,
        )
        drawing.text(
            (point_x + radius + stroke, point_y - radius),
            label,
            fill=color,
            font=marker_font,
            stroke_width=max(1, stroke // 3),
            stroke_fill="white",
        )

    targets = []
    if start_target is not None:
        targets.append((start_target, "#00a6d6", "S" if end_target else "N"))
    if end_target is not None:
        targets.append((end_target, "#a020f0", "E"))
    selected_targets = []
    if review_target is not None:
        selected_targets.append((review_target, "#00a651", "C"))
    if review_start_target is not None:
        selected_targets.append((review_start_target, "#00a651", "S✓"))
    if review_end_target is not None:
        selected_targets.append((review_end_target, "#7b2cbf", "E✓"))
    for point, label in review_candidates:
        draw_target(
            full_draw,
            point,
            color="#ff8c00",
            label=label,
            radius_scale=0.72,
            target_stroke=max(2, stroke // 2),
        )
    for target, color, label in targets:
        line_end_x, line_end_y = connection_end(target)
        full_draw.line(
            (center_x, center_y, line_end_x, line_end_y),
            fill=color,
            width=max(2, stroke // 2),
        )
        draw_target(full_draw, target, color=color, label=label)
    for target, color, label in selected_targets:
        line_end_x, line_end_y = connection_end(target)
        full_draw.line(
            (center_x, center_y, line_end_x, line_end_y),
            fill=color,
            width=max(2, stroke // 2),
        )
        draw_target(full_draw, target, color=color, label=label)

    all_focus_targets = [
        *(target for target, _color, _label in targets),
        *(target for target, _color, _label in selected_targets),
        *(target for target, _label in review_candidates),
    ]
    focus_x = [left, right, *(target[0] for target in all_focus_targets)]
    focus_y = [top, bottom, *(target[1] for target in all_focus_targets)]
    horizontal_padding = max(box_width * 2.5, width * 0.08)
    vertical_padding = max(box_height * 2.0, height * 0.055)
    crop_left = max(0, round(min(focus_x) - horizontal_padding))
    crop_top = max(0, round(min(focus_y) - vertical_padding))
    crop_right = min(width, round(max(focus_x) + horizontal_padding))
    crop_bottom = min(height, round(max(focus_y) + vertical_padding))
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_draw = ImageDraw.Draw(crop)
    crop_draw.rectangle(
        (
            left - crop_left,
            top - crop_top,
            right - crop_left,
            bottom - crop_top,
        ),
        outline="#e31a1c",
        width=stroke,
    )
    for point, label in review_candidates:
        draw_target(
            crop_draw,
            point,
            color="#ff8c00",
            label=label,
            offset_x=crop_left,
            offset_y=crop_top,
            radius_scale=0.72,
            target_stroke=max(2, stroke // 2),
        )
    for target, color, label in targets:
        line_end_x, line_end_y = connection_end(
            target, offset_x=crop_left, offset_y=crop_top
        )
        crop_draw.line(
            (
                center_x - crop_left,
                center_y - crop_top,
                line_end_x,
                line_end_y,
            ),
            fill=color,
            width=max(2, stroke // 2),
        )
        draw_target(
            crop_draw,
            target,
            color=color,
            label=label,
            offset_x=crop_left,
            offset_y=crop_top,
        )
    for target, color, label in selected_targets:
        line_end_x, line_end_y = connection_end(
            target, offset_x=crop_left, offset_y=crop_top
        )
        crop_draw.line(
            (
                center_x - crop_left,
                center_y - crop_top,
                line_end_x,
                line_end_y,
            ),
            fill=color,
            width=max(2, stroke // 2),
        )
        draw_target(
            crop_draw,
            target,
            color=color,
            label=label,
            offset_x=crop_left,
            offset_y=crop_top,
        )

    def png_bytes(rendered: Image.Image) -> bytes:
        target = io.BytesIO()
        rendered.save(target, format="PNG")
        return target.getvalue()

    rendered = (png_bytes(full), png_bytes(crop))
    if not include_crop_geometry:
        return rendered
    return (
        *rendered,
        {
            "left": crop_left,
            "top": crop_top,
            "width": crop_right - crop_left,
            "height": crop_bottom - crop_top,
            "page_width": width,
            "page_height": height,
        },
    )


def _key(row: dict) -> tuple[str, str]:
    return _text(row.get("page_id")), _text(row.get("yolo_line", row.get("txt_line")))


def _validate_time(value: str, label: str, key: tuple[str, str]) -> str | None:
    if not value:
        return None
    try:
        float(value)
    except ValueError:
        return f"{key[0]}:Y{key[1]} {label} must be numeric or blank"
    return None


def _machine_final_source(row: dict) -> dict:
    output = dict(row)
    output["alignment_status"] = {
        "matched": "matched",
        "inferred": "ambiguous",
        "review": "ambiguous",
    }.get(_text(row.get("status")), "unresolved")
    output["human_approved"] = "false"
    return output


def apply_review_decisions(
    detailed_rows: list[dict],
    editor_rows: list[dict],
    *,
    reviewer: str,
    reviewed_at: str | None = None,
    valid_note_ids: set[str] | None = None,
    class_map: dict[str, str] | None = None,
) -> tuple[list[dict], dict, dict, list[str]]:
    """Apply decisions without mutating inputs and return strict final rows."""

    from bpsd_aligner.web_pipeline import build_final_bps_rows

    reviewed_at = reviewed_at or datetime.now(timezone.utc).isoformat()
    original = {_key(row): row for row in detailed_rows}
    decisions: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    for editor in editor_rows:
        action = _text(editor.get("action")) or "pending"
        if action not in REVIEW_ACTIONS:
            errors.append(f"invalid review action: {action}")
            continue
        if action == "pending":
            continue
        key = _key(editor)
        if key not in original:
            errors.append(f"review row not found: {key[0]}:Y{key[1]}")
            continue
        if key in decisions:
            errors.append(f"duplicate review decision: {key[0]}:Y{key[1]}")
            continue
        for field in ("start_meas", "end_meas"):
            error = _validate_time(_text(editor.get(field)), field, key)
            if error:
                errors.append(error)
        start_time = _text(editor.get("start_meas"))
        end_time = _text(editor.get("end_meas"))
        if start_time and end_time:
            try:
                if float(start_time) > float(end_time):
                    errors.append(
                        f"{key[0]}:Y{key[1]} start_meas must not exceed end_meas"
                    )
            except ValueError:
                pass
        staff = _text(editor.get("staff"))
        if staff and staff not in {"1", "2"}:
            errors.append(f"{key[0]}:Y{key[1]} staff must be 1, 2, or blank")
        note_values = {
            "start_note": _text(editor.get("start_note")),
            "end_note": _text(editor.get("end_note")),
        }
        connected_ids = _split_ids(editor.get("connected_note"))
        for field, value in note_values.items():
            if value and not value.isdigit():
                errors.append(
                    f"{key[0]}:Y{key[1]} {field} must be an integer note ID or blank"
                )
            elif value and valid_note_ids is not None and value not in valid_note_ids:
                errors.append(f"{key[0]}:Y{key[1]} {field} note ID {value} does not exist")
        for value in connected_ids:
            if not value.isdigit():
                errors.append(
                    f"{key[0]}:Y{key[1]} connected_note contains non-integer ID {value}"
                )
            elif valid_note_ids is not None and value not in valid_note_ids:
                errors.append(
                    f"{key[0]}:Y{key[1]} connected_note ID {value} does not exist"
                )
        if connected_ids:
            for field, value in note_values.items():
                if value and value not in connected_ids:
                    errors.append(
                        f"{key[0]}:Y{key[1]} {field} must appear in connected_note"
                    )
        if action == "correct":
            source = original[key]
            changed = any(
                _text(editor.get(field)) != _text(source.get(field))
                for field in SEMANTIC_FIELDS
            ) or _text(editor.get("staff")) != _text(source.get("xml_staff"))
            if not changed:
                errors.append(
                    f"{key[0]}:Y{key[1]} action=correct but no value was changed; "
                    "use confirm instead"
                )
        if action == "wrong_class":
            corrected_class = _text(editor.get("corrected_class"))
            corrected_class_id = _text(editor.get("corrected_class_id"))
            if (
                not corrected_class
                or not corrected_class_id
                or (
                    corrected_class == _text(original[key].get("class"))
                    and corrected_class_id == _text(original[key].get("class_id"))
                )
            ):
                errors.append(
                    f"{key[0]}:Y{key[1]} wrong_class requires corrected class ID "
                    "and class name"
                )
            elif class_map is not None and class_map.get(corrected_class_id) != corrected_class:
                errors.append(
                    f"{key[0]}:Y{key[1]} class ID {corrected_class_id} maps to "
                    f"{class_map.get(corrected_class_id, 'no class')}, not {corrected_class}"
                )
        decisions[key] = dict(editor)
    if errors:
        return [], {}, {}, errors

    applied_sources: list[dict] = []
    correction_entries: list[dict] = []
    evaluation_items: list[dict] = []
    for source in detailed_rows:
        key = _key(source)
        decision = decisions.get(key)
        final_source = _machine_final_source(source)
        if decision is not None:
            action = decision["action"]
            machine_values = {
                **{field: _text(source.get(field)) for field in SEMANTIC_FIELDS},
                "staff": _text(source.get("xml_staff")),
            }
            if action in {"reject", "scan_only", "bad_bbox"}:
                ground_truth = {field: "" for field in EVALUATION_FIELDS}
                final_source["alignment_status"] = "unresolved"
                final_source["status"] = "unresolved"
                for field in SEMANTIC_FIELDS:
                    final_source[field] = ""
            elif action == "confirm":
                ground_truth = dict(machine_values)
                final_source["alignment_status"] = "matched"
                final_source["status"] = "matched"
                final_source["review_status"] = "confirmed"
                final_source["match_source"] = "human_confirmation"
                final_source["human_approved"] = "false"
            elif action == "correct":
                ground_truth = {
                    field: _text(decision.get(field, source.get(field, "")))
                    for field in SEMANTIC_FIELDS
                }
                ground_truth["staff"] = _text(decision.get("staff"))
                final_source.update(ground_truth)
                final_source["alignment_status"] = "matched"
                final_source["status"] = "matched"
                final_source["review_status"] = "corrected"
                final_source["match_source"] = "human_review"
                final_source["human_approved"] = "true"
                if _text(decision.get("review_target_x_px")):
                    final_source["target_x_px"] = _text(
                        decision.get("review_target_x_px")
                    )
                if _text(decision.get("review_target_y_px")):
                    final_source["target_y_px"] = _text(
                        decision.get("review_target_y_px")
                    )
                if _text(decision.get("review_start_target_x_px")):
                    final_source["target_x_px"] = _text(
                        decision.get("review_start_target_x_px")
                    )
                if _text(decision.get("review_start_target_y_px")):
                    final_source["target_y_px"] = _text(
                        decision.get("review_start_target_y_px")
                    )
                if _text(decision.get("review_end_target_x_px")):
                    final_source["end_target_x_px"] = _text(
                        decision.get("review_end_target_x_px")
                    )
                if _text(decision.get("review_end_target_y_px")):
                    final_source["end_target_y_px"] = _text(
                        decision.get("review_end_target_y_px")
                    )
            elif action == "wrong_class":
                ground_truth = {field: "" for field in EVALUATION_FIELDS}
                final_source["class_id"] = _text(
                    decision.get("corrected_class_id")
                )
                final_source["class"] = _text(decision.get("corrected_class"))
                for field in SEMANTIC_FIELDS:
                    final_source[field] = ""
                final_source["alignment_status"] = "unresolved"
                final_source["status"] = "unresolved"
                final_source["match_source"] = "human_wrong_class"
                final_source["human_approved"] = "true"
            else:
                # uncertain/skipped are preserved as explicit review decisions,
                # but they do not promote machine candidates into final values.
                ground_truth = {field: "" for field in EVALUATION_FIELDS}
            final_source["xml_staff"] = (
                _text(decision.get("staff"))
                if action == "correct"
                else _text(source.get("xml_staff"))
            )
            correction_entries.append(
                {
                    "page_id": key[0],
                    "yolo_line": key[1],
                    "class": _text(source.get("class")),
                    "corrected_class_id": _text(
                        decision.get("corrected_class_id")
                    ),
                    "corrected_class": _text(decision.get("corrected_class")),
                    "action": action,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "original": machine_values,
                    "corrected": ground_truth,
                    "staff": final_source["xml_staff"],
                    "review_target_x_px": _text(
                        decision.get("review_target_x_px")
                    ),
                    "review_target_y_px": _text(
                        decision.get("review_target_y_px")
                    ),
                    "review_target_pitch": _text(
                        decision.get("review_target_pitch")
                    ),
                    "review_start_target_x_px": _text(
                        decision.get("review_start_target_x_px")
                    ),
                    "review_start_target_y_px": _text(
                        decision.get("review_start_target_y_px")
                    ),
                    "review_end_target_x_px": _text(
                        decision.get("review_end_target_x_px")
                    ),
                    "review_end_target_y_px": _text(
                        decision.get("review_end_target_y_px")
                    ),
                    "comment": _text(decision.get("comment")),
                }
            )
            if action in {"confirm", "correct", "reject"}:
                evaluation_items.append(
                    {
                        "class": _text(source.get("class")),
                        "machine": machine_values,
                        "expected": ground_truth,
                        "action": action,
                    }
                )
        if decision is None or decision.get("action") not in {
            "not_a_symbol",
            "bad_bbox",
        }:
            applied_sources.append(final_source)

    final_rows = build_final_bps_rows(applied_sources)
    paired = list(zip(applied_sources, final_rows))
    paired.sort(
        key=lambda pair: (
            float(pair[1]["start_meas"])
            if _text(pair[1].get("start_meas"))
            else float("inf"),
            _text(pair[0].get("page_id")),
            int(_text(pair[0].get("txt_line")) or 0),
        )
    )
    final_rows = [pair[1] for pair in paired]
    payload = {
        "schema_version": "1.0",
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "entries": correction_entries,
    }
    report = build_accuracy_report(evaluation_items)
    return final_rows, payload, report, []


def apply_corrections_to_master_rows(
    master_rows: list[dict], corrections_payload: dict
) -> list[dict]:
    """Apply reviewed values to detailed YOLO master rows without losing provenance."""

    decisions = {
        (_text(entry.get("page_id")), _text(entry.get("yolo_line"))): entry
        for entry in corrections_payload.get("entries", [])
    }
    output = []
    for source in master_rows:
        row = dict(source)
        key = (_text(row.get("page_id")), _text(row.get("yolo_line")))
        decision = decisions.get(key)
        if decision is None:
            output.append(row)
            continue
        action = decision.get("action")
        if action in {"not_a_symbol", "bad_bbox"}:
            continue
        original = {
            field: row.get(field, "")
            for field in (*SEMANTIC_FIELDS, "staff", "class_id", "class")
        }
        if action == "confirm":
            row.update(
                {
                    "alignment_status": "matched",
                    "review_status": "confirmed",
                    "match_source": "human_confirmation",
                    "human_approved": "false",
                }
            )
        elif action == "correct":
            corrected = decision.get("corrected", {})
            for field in SEMANTIC_FIELDS:
                row[field] = _text(corrected.get(field))
            row["staff"] = _text(decision.get("staff"))
            row.update(
                {
                    "alignment_status": "matched",
                    "review_status": "corrected",
                    "match_source": "human_review",
                    "human_approved": "true",
                }
            )
            if _text(decision.get("review_target_x_px")):
                row["target_x_px"] = _text(decision.get("review_target_x_px"))
            if _text(decision.get("review_target_y_px")):
                row["target_y_px"] = _text(decision.get("review_target_y_px"))
            if _text(decision.get("review_start_target_x_px")):
                row["target_x_px"] = _text(decision.get("review_start_target_x_px"))
            if _text(decision.get("review_start_target_y_px")):
                row["target_y_px"] = _text(decision.get("review_start_target_y_px"))
            if _text(decision.get("review_end_target_x_px")):
                row["end_target_x_px"] = _text(
                    decision.get("review_end_target_x_px")
                )
            if _text(decision.get("review_end_target_y_px")):
                row["end_target_y_px"] = _text(
                    decision.get("review_end_target_y_px")
                )
        elif action == "wrong_class":
            row["class_id"] = _text(decision.get("corrected_class_id"))
            row["class"] = _text(decision.get("corrected_class"))
            for field in SEMANTIC_FIELDS:
                row[field] = ""
            row.update(
                {
                    "alignment_status": "unresolved",
                    "review_status": "corrected_class_needs_alignment",
                    "match_source": "human_wrong_class",
                    "human_approved": "true",
                }
            )
        elif action in {"reject", "scan_only"}:
            for field in SEMANTIC_FIELDS:
                row[field] = ""
            row.update(
                {
                    "alignment_status": "unresolved",
                    "review_status": action,
                    "match_source": f"human_{action}",
                    "human_approved": "true",
                }
            )
        elif action in {"uncertain", "skipped"}:
            row["review_status"] = "deferred"
        row["reviewer"] = corrections_payload.get("reviewer", "")
        row["reviewed_at"] = corrections_payload.get("reviewed_at", "")
        row["review_source"] = "website"
        row["original_candidate_json"] = json.dumps(
            original, ensure_ascii=False, sort_keys=True
        )
        row["corrected_value_json"] = json.dumps(
            decision.get("corrected", {}), ensure_ascii=False, sort_keys=True
        )
        row["comment"] = _text(decision.get("comment"))
        output.append(row)
    return output


def build_accuracy_report(items: list[dict]) -> dict:
    """Measure original machine values against reviewed ground truth."""

    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_class[item.get("class", "unknown")].append(item)

    def summarize(rows: list[dict]) -> dict:
        fields = {}
        exact = 0
        for field in EVALUATION_FIELDS:
            eligible = [
                row
                for row in rows
                if row.get("action") == "reject"
                or _text(row["expected"].get(field))
            ]
            correct = sum(
                _equivalent(
                    field,
                    row["machine"].get(field),
                    row["expected"].get(field),
                )
                for row in eligible
            )
            fields[field] = {
                "evaluated": len(eligible),
                "correct": correct,
                "accuracy": correct / len(eligible) if eligible else None,
            }
        for row in rows:
            eligible_fields = [
                field
                for field in EVALUATION_FIELDS
                if row.get("action") == "reject"
                or _text(row["expected"].get(field))
            ]
            if eligible_fields and all(
                _equivalent(
                    field,
                    row["machine"].get(field),
                    row["expected"].get(field),
                )
                for field in eligible_fields
            ):
                exact += 1
        return {
            "reviewed_rows": len(rows),
            "exact_rows": exact,
            "exact_accuracy": exact / len(rows) if rows else None,
            "fields": fields,
        }

    return {
        "reviewed_rows": len(items),
        "overall": summarize(items),
        "by_class": {name: summarize(rows) for name, rows in sorted(by_class.items())},
    }


def evaluate_ground_truth_rows(
    ground_truth_rows: list[dict], prediction_rows: list[dict]
) -> tuple[dict, list[str]]:
    """Evaluate a detailed alignment CSV against normalized human truth."""

    predictions = {_key(row): row for row in prediction_rows}
    items: list[dict] = []
    errors: list[str] = []
    missing: list[str] = []
    class_mismatches: list[dict] = []
    for truth in ground_truth_rows:
        key = _key(truth)
        prediction = predictions.get(key)
        if prediction is None:
            missing.append(f"{key[0]}:Y{key[1]}")
            continue
        expected = {
            "start_meas": _text(truth.get("expected_start_meas")),
            "end_meas": _text(truth.get("expected_end_meas")),
            "start_note": _text(truth.get("expected_start_note")),
            "end_note": _text(truth.get("expected_end_note")),
            "connected_note": _text(truth.get("expected_connected_note")),
            "stem_dir": "",
            "staff": _text(truth.get("expected_staff")),
        }
        machine = {
            **{field: _text(prediction.get(field)) for field in SEMANTIC_FIELDS},
            "staff": _first(prediction, "xml_staff", "staff"),
        }
        truth_class = _text(truth.get("class"))
        predicted_class = _text(prediction.get("class"))
        if truth_class and truth_class != predicted_class:
            class_mismatches.append(
                {
                    "row": f"{key[0]}:Y{key[1]}",
                    "expected": truth_class,
                    "actual": predicted_class,
                }
            )
        items.append(
            {
                "class": truth_class or predicted_class,
                "machine": machine,
                "expected": expected,
                "action": "ground_truth",
            }
        )
    report = build_accuracy_report(items)
    report.update(
        {
            "ground_truth_rows": len(ground_truth_rows),
            "matched_prediction_rows": len(items),
            "missing_predictions": missing,
            "class_mismatches": class_mismatches,
        }
    )
    if missing:
        errors.append(f"missing predictions for {len(missing)} ground-truth rows")
    if class_mismatches:
        errors.append(f"class mismatch for {len(class_mismatches)} ground-truth rows")
    return report, errors


def csv_bytes(rows: list[dict], fields: list[str] | tuple[str, ...]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize class-specific human review CSVs"
    )
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional alignment_detailed.csv to score against ground truth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.review_dir.glob("*_human_review*.csv"))
    emit_progress("review-evaluation", 0, 2, f"normalizing {len(paths)} files")
    rows, errors = normalize_legacy_review_rows(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "evaluation_ground_truth.csv"
    report_path = args.output_dir / "normalization_report.json"
    atomic_write_csv(csv_path, GROUND_TRUTH_FIELDS, rows)
    report = {
        "source_files": len(paths),
        "ground_truth_rows": len(rows),
        "class_counts": dict(Counter(row["class"] for row in rows)),
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {"ground_truth_csv": str(csv_path)},
    }
    if args.predictions is not None:
        with args.predictions.open(newline="", encoding="utf-8-sig") as file:
            predictions = list(csv.DictReader(file))
        inferred_page_id = args.predictions.name.removesuffix(
            "_alignment_detailed.csv"
        )
        for prediction in predictions:
            prediction["page_id"] = _text(prediction.get("page_id")) or inferred_page_id
        accuracy, accuracy_errors = evaluate_ground_truth_rows(rows, predictions)
        accuracy["prediction_file"] = args.predictions.name
        accuracy_path = args.output_dir / "accuracy_report.json"
        atomic_write_json(accuracy_path, accuracy)
        report["accuracy"] = accuracy
        report["outputs"]["accuracy_report_json"] = str(accuracy_path)
        errors.extend(accuracy_errors)
        report["validation_errors"] = errors
        report["passed"] = not errors
    atomic_write_json(report_path, report)
    emit_progress("review-evaluation", 2, 2, f"passed={report['passed']} rows={len(rows)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
