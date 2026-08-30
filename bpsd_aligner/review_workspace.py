"""Pure note-selection helpers for the Streamlit Review workspace."""

from __future__ import annotations

import json
import re


def candidate_printed_measure(candidate: dict) -> str:
    explicit = str(candidate.get("printed_measure", "") or "").strip()
    if explicit:
        return explicit
    xml_measure = str(candidate.get("xml_measure", "") or "").strip()
    try:
        timeline_measure = str(int(float(candidate.get("start_meas", ""))))
        if xml_measure and abs(int(xml_measure) - int(timeline_measure)) <= 1:
            return timeline_measure
    except (TypeError, ValueError):
        pass
    return xml_measure


def review_note_label(candidate: dict) -> str:
    staff = str(candidate.get("staff", ""))
    staff_label = {"1": "上 staff", "2": "下 staff"}.get(staff, f"staff {staff or '—'}")
    order = candidate.get("measure_note_order")
    order_label = f"第 {order} 個音" if str(order or "").strip() else "音符"
    return (
        f"第 {candidate_printed_measure(candidate) or '—'} 小節 · "
        f"{staff_label} · {candidate.get('pitch') or '音高不明'} · {order_label}"
    )


def fill_missing_note_orders(candidates: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(
            (
                str(candidate.get("page_id", "")),
                candidate_printed_measure(candidate),
                str(candidate.get("staff", "")),
            ),
            [],
        ).append(candidate)
    for group in groups.values():
        group.sort(
            key=lambda candidate: (
                float(candidate.get("x_px", 0) or 0),
                float(candidate.get("y_px", 0) or 0),
                str(candidate.get("note_id", "")),
            )
        )
        for order, candidate in enumerate(group, start=1):
            candidate.setdefault("measure_note_order", order)


def merge_page_note_candidates(
    current_candidates: list[dict], detailed_rows: list[dict], page_id: str
) -> list[dict]:
    merged = []
    seen = set()
    candidate_groups = [current_candidates]
    for row in detailed_rows:
        if str(row.get("page_id", "")) != str(page_id):
            continue
        try:
            row_candidates = json.loads(
                str(row.get("review_note_candidates_json", "") or "[]")
            )
        except json.JSONDecodeError:
            continue
        if isinstance(row_candidates, list):
            candidate_groups.append(row_candidates)
    for group in candidate_groups:
        for candidate in group:
            if not isinstance(candidate, dict):
                continue
            sequence = str(candidate.get("xml_note_sequence", "")).strip()
            identity = (
                ("sequence", sequence)
                if sequence
                else (
                    "geometry",
                    str(candidate.get("note_id", "")),
                    str(candidate.get("xml_measure", "")),
                    str(candidate.get("printed_measure", "")),
                    str(candidate.get("staff", "")),
                    str(candidate.get("pitch", "")),
                    str(candidate.get("x_px", "")),
                    str(candidate.get("y_px", "")),
                )
            )
            if identity in seen:
                continue
            seen.add(identity)
            prepared = dict(candidate)
            prepared.setdefault("page_id", str(page_id))
            merged.append(prepared)
    return merged


def endpoint_note_input_value(candidate: dict | None) -> str:
    if candidate is None:
        return ""
    staff = {"1": "上", "2": "下"}.get(
        str(candidate.get("staff", "")), str(candidate.get("staff", ""))
    )
    return ", ".join(
        [
            candidate_printed_measure(candidate),
            staff,
            str(candidate.get("pitch", "")),
            str(candidate.get("measure_note_order", "")),
        ]
    )


def endpoint_note_input_values(candidates: list[dict]) -> str:
    """Serialize one or more endpoint notes for the editable Review field."""

    values = [endpoint_note_input_value(candidate) for candidate in candidates]
    return " ; ".join(value for value in values if value)


def candidate_note_id(candidate: dict | None) -> str:
    if candidate is None or candidate.get("note_id") is None:
        return ""
    return str(candidate["note_id"])


def normalize_pitch(value: object) -> str:
    return str(value or "").strip().replace("♯", "#").replace("♭", "b").upper()


def resolve_endpoint_note_input(
    value: str, candidates: list[dict]
) -> tuple[dict | None, str]:
    compact = str(value).strip()
    if not compact:
        return None, ""
    parts = [part.strip() for part in re.split(r"[,，/|]", compact)]
    if len(parts) != 4:
        return None, "請輸入四項：小節, staff, 音高, 第幾個音"
    measure = re.sub(r"^(第)?|小節$", "", parts[0]).strip()
    staff_text = parts[1].lower().replace("staff", "").strip()
    staff = {"上": "1", "upper": "1", "下": "2", "lower": "2"}.get(
        staff_text, staff_text
    )
    pitch = normalize_pitch(parts[2])
    order = re.sub(r"^(第)?|個音$|音$", "", parts[3]).strip()
    matches = [
        candidate
        for candidate in candidates
        if candidate_printed_measure(candidate) == measure
        and str(candidate.get("staff", "")) == staff
        and normalize_pitch(candidate.get("pitch")) == pitch
        and str(candidate.get("measure_note_order", "")) == order
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, "找到多個相同音符，請確認該小節第幾個音"
    return None, "找不到這個音符，請檢查小節、staff、音高與順序"


def resolve_endpoint_note_inputs(
    value: str, candidates: list[dict]
) -> tuple[list[dict], str]:
    """Resolve a semicolon/newline-separated endpoint chord selection."""

    compact = str(value).strip()
    if not compact:
        return [], ""
    selections = [
        part.strip() for part in re.split(r"[;；\n]+", compact) if part.strip()
    ]
    resolved = []
    seen = set()
    for selection in selections:
        candidate, error = resolve_endpoint_note_input(selection, candidates)
        if error:
            return [], error
        identity = (
            str(candidate.get("candidate_id", "") or "").strip()
            or (str(candidate.get("xml_note_sequence", "") or "").strip())
            or endpoint_note_input_value(candidate)
        )
        if identity in seen:
            continue
        seen.add(identity)
        resolved.append(candidate)
    if len(resolved) > 1:
        onset_keys = {
            (
                candidate_printed_measure(candidate),
                str(candidate.get("start_meas", "") or "").strip(),
            )
            for candidate in resolved
        }
        if len(onset_keys) > 1:
            return [], "同一端點的多個音必須屬於同一小節與同一開始時間"
    return resolved, ""


def toggle_endpoint_note_input(
    value: str, candidate: dict, candidates: list[dict]
) -> tuple[str, bool]:
    """Toggle one clicked note in a chord selection.

    Returns the updated editable value and whether the note was added.
    """

    selected, error = resolve_endpoint_note_inputs(value, candidates)
    if error:
        selected = []
    clicked_id = candidate_note_id(candidate)

    def same_note(item: dict) -> bool:
        item_id = candidate_note_id(item)
        if clicked_id and item_id:
            return item_id == clicked_id
        return endpoint_note_input_value(item) == endpoint_note_input_value(candidate)

    if any(same_note(item) for item in selected):
        return endpoint_note_input_values(
            [item for item in selected if not same_note(item)]
        ), False
    selected.append(candidate)
    return endpoint_note_input_values(selected), True


def snap_click_to_note_candidate(
    click: dict,
    candidates: list[dict],
    crop_geometry: dict,
    *,
    max_display_distance: float = 42.0,
) -> tuple[dict | None, float | None]:
    try:
        click_x = float(click["x"])
        click_y = float(click["y"])
        display_width = float(click["width"])
        display_height = float(click["height"])
        crop_width = float(crop_geometry["width"])
        crop_height = float(crop_geometry["height"])
        crop_left = float(crop_geometry["left"])
        crop_top = float(crop_geometry["top"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if min(display_width, display_height, crop_width, crop_height) <= 0:
        return None, None
    closest = None
    closest_distance = None
    for candidate in candidates:
        try:
            displayed_x = (
                (float(candidate["x_px"]) - crop_left) * display_width / crop_width
            )
            displayed_y = (
                (float(candidate["y_px"]) - crop_top) * display_height / crop_height
            )
        except (KeyError, TypeError, ValueError):
            continue
        distance = ((click_x - displayed_x) ** 2 + (click_y - displayed_y) ** 2) ** 0.5
        if closest_distance is None or distance < closest_distance:
            closest = candidate
            closest_distance = distance
    if closest_distance is None or closest_distance > max_display_distance:
        return None, closest_distance
    return closest, closest_distance
