"""Link existing YOLO/BPSD rows to XML events and build a lossless master CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)
from xml_export import BPS_FIELDS, EVENT_FIELDS


PIPELINE_VERSION = "0.3.0"
LINK_FIELDS = [
    "link_id", "bbox_id", "xml_event_id", "xml_node_id", "score_id",
    "page_id", "yolo_line", "yolo_class", "xml_event_type",
    "xml_event_subtype", "relation_type", "bps_note_id", "link_method",
    "link_status", "is_primary", "confidence", "human_approved",
    "review_status", "source_alignment_status", "pipeline_version",
    "validation_status",
]
COMBINED_CONTROL_FIELDS = [
    "record_id", "row_origin", "combined_status", "source_alignment_status",
    "xml_event_ids_json", "xml_event_count", "xml_events_json",
]

NOTATION_CLASSES = {
    "articStaccatoAbove": {"staccato"},
    "articStaccatoBelow": {"staccato"},
    "articAccentAbove": {"accent"},
    "articAccentBelow": {"accent"},
    "articMarcatoAbove": {"strong-accent"},
    "articMarcatoBelow": {"strong-accent"},
    "fermataAbove": {"fermata"},
    "fermataBelow": {"fermata"},
    "slur": {"slur"},
    "tie": {"tied"},
    "tuplet3": {"tuplet"},
    "tuplet5": {"tuplet"},
    "tuplet6": {"tuplet"},
    "tupletBracket": {"tuplet"},
}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_list(value: str) -> list:
    if not value or value == "NA":
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _note_ids(row: dict) -> list[int]:
    for field in ("note_ids", "connected_note"):
        values = [_int(value) for value in _json_list(row.get(field, ""))]
        values = [value for value in values if value is not None]
        if values:
            return list(dict.fromkeys(values))
    values = [_int(row.get("start_note")), _int(row.get("end_note"))]
    return list(dict.fromkeys(value for value in values if value is not None))


def _page_values(row: dict) -> set[int]:
    values = set()
    direct = _int(row.get("xml_page"))
    if direct is not None:
        values.add(direct)
    for value in _json_list(row.get("xml_systems_json", "")):
        if isinstance(value, dict):
            page = _int(value.get("xml_page"))
            if page is not None:
                values.add(page)
    return values


def _select_note_event(
    candidates: list[tuple[dict, dict]], yolo_row: dict
) -> tuple[dict, dict] | None:
    if not candidates:
        return None
    pages = _page_values(yolo_row)
    if pages:
        on_page = [item for item in candidates if _int(item[0].get("page")) in pages]
        if on_page:
            candidates = on_page
    wanted_staff = str(yolo_row.get("staff", ""))
    wanted_measure = str(yolo_row.get("xml_measure", ""))
    wanted_time = _float(yolo_row.get("start_meas"))
    status_rank = {"exact": 0, "rounded_tolerance": 1, "within_tied_span": 2}
    return min(
        candidates,
        key=lambda item: (
            status_rank.get(item[1].get("note_match_status", ""), 3),
            0 if not wanted_staff or item[0].get("staff") == wanted_staff else 1,
            0 if not wanted_measure or item[0].get("xml_measure") == wanted_measure else 1,
            abs(float(item[1].get("start_meas", 0)) - wanted_time)
            if wanted_time is not None else 0,
            item[0]["xml_event_id"],
        ),
    )


def _select_dynamic(events: list[dict], yolo_row: dict) -> dict | None:
    symbol = yolo_row.get("xml_symbol", "")
    if not symbol or symbol == "NA":
        return None
    candidates = [
        event for event in events
        if event["event_type"] == "direction"
        and event["event_subtype"] == "dynamic"
        and event["dynamic"] == symbol
    ]
    measure = yolo_row.get("xml_measure", "")
    if measure:
        same_measure = [event for event in candidates if event["xml_measure"] == measure]
        if same_measure:
            candidates = same_measure
        else:
            return None
    pages = _page_values(yolo_row)
    if pages:
        same_page = [event for event in candidates if _int(event.get("page")) in pages]
        if same_page:
            candidates = same_page
    if not candidates:
        return None
    wanted_time = _float(yolo_row.get("start_meas"))
    wanted_staff = yolo_row.get("staff", "")
    return min(
        candidates,
        key=lambda event: (
            0 if not wanted_staff or event.get("staff") == wanted_staff else 1,
            abs(float(event.get("start_meas") or 0) - wanted_time)
            if wanted_time is not None else 0,
            event["xml_event_id"],
        ),
    )


def _select_rest(events: list[dict], yolo_row: dict) -> dict | None:
    candidates = [event for event in events if event["event_type"] == "rest"]
    measure = yolo_row.get("xml_measure", "")
    if measure:
        candidates = [event for event in candidates if event["xml_measure"] == measure]
    staff = yolo_row.get("staff", "")
    if staff:
        same_staff = [event for event in candidates if event["staff"] == staff]
        if same_staff:
            candidates = same_staff
    pages = _page_values(yolo_row)
    if pages:
        same_page = [event for event in candidates if _int(event.get("page")) in pages]
        if same_page:
            candidates = same_page
    if not candidates:
        return None
    wanted_time = _float(yolo_row.get("start_meas"))
    return min(
        candidates,
        key=lambda event: (
            abs(float(event.get("start_meas") or 0) - wanted_time)
            if wanted_time is not None else 0,
            event["xml_event_id"],
        ),
    )


def _relation_type(yolo_row: dict, note_id: int) -> str:
    if yolo_row.get("target_type") != "span":
        return "target_note"
    start_note = _int(yolo_row.get("start_note"))
    end_note = _int(yolo_row.get("end_note"))
    if start_note == note_id and end_note != note_id:
        return "span_start"
    if end_note == note_id and start_note != note_id:
        return "span_end"
    return "span_endpoint"


def build_score_links(yolo_rows: list[dict], xml_events: list[dict]) -> list[dict]:
    """Build evidence-backed links without rerunning image/XML alignment."""

    note_index: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    notation_by_anchor: dict[str, list[dict]] = defaultdict(list)
    event_by_id = {event["xml_event_id"]: event for event in xml_events}
    for event in xml_events:
        if event["event_type"] == "note":
            for occurrence in _json_list(event.get("repeat_json", "")):
                note_id = _int(occurrence.get("note_id")) if isinstance(occurrence, dict) else None
                if note_id is not None:
                    note_index[note_id].append((event, occurrence))
        if event["event_type"] == "notation":
            for anchor in _json_list(event.get("anchor_xml_event_ids_json", "")):
                notation_by_anchor[str(anchor)].append(event)

    pending = []
    allowed_statuses = {"matched", "candidate", "ambiguous"}
    for yolo_row in yolo_rows:
        if yolo_row.get("alignment_status") not in allowed_statuses:
            continue
        bbox_links = []
        for note_id in _note_ids(yolo_row):
            selected = _select_note_event(note_index.get(note_id, []), yolo_row)
            if selected is None:
                continue
            note_event, _occurrence = selected
            relation = _relation_type(yolo_row, note_id)
            bbox_links.append((note_event, relation, note_id, "bps_note_anchor"))

            allowed_notations = NOTATION_CLASSES.get(yolo_row.get("class", ""), set())
            notation_candidates = [
                event for event in notation_by_anchor.get(note_event["xml_event_id"], [])
                if event["event_subtype"] in allowed_notations
            ]
            if relation in {"span_start", "span_end"}:
                expected_type = "start" if relation == "span_start" else "stop"
                typed = [
                    event for event in notation_candidates
                    if json.loads(event.get("xml_attributes_json") or "{}").get("type") == expected_type
                ]
                if typed:
                    notation_candidates = typed
            for notation in notation_candidates:
                bbox_links.append(
                    (notation, "semantic_event", note_id, "xml_notation_anchor")
                )

        if yolo_row.get("class", "").startswith("dynamic"):
            dynamic = _select_dynamic(xml_events, yolo_row)
            if dynamic is not None:
                bbox_links.append(
                    (dynamic, "semantic_event", None, "xml_direction_measure_symbol")
                )
        if yolo_row.get("target_type") == "rest":
            rest = _select_rest(xml_events, yolo_row)
            if rest is not None:
                bbox_links.append((rest, "target_rest", None, "xml_rest_anchor"))
                allowed_notations = NOTATION_CLASSES.get(yolo_row.get("class", ""), set())
                for notation in notation_by_anchor.get(rest["xml_event_id"], []):
                    if notation["event_subtype"] in allowed_notations:
                        bbox_links.append(
                            (notation, "semantic_event", None, "xml_notation_anchor")
                        )

        deduplicated = {}
        for event, relation, note_id, method in bbox_links:
            key = (event["xml_event_id"], relation, note_id, method)
            deduplicated[key] = (event, relation, note_id, method)
        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                0 if item[3] in {"xml_notation_anchor", "xml_direction_measure_symbol"} else 1,
                item[0]["xml_event_id"],
                item[1],
                -1 if item[2] is None else item[2],
            ),
        )
        for index, (event, relation, note_id, method) in enumerate(ordered):
            pending.append(
                {
                    "bbox_id": yolo_row["bbox_id"],
                    "xml_event_id": event["xml_event_id"],
                    "xml_node_id": event["xml_node_id"],
                    "score_id": yolo_row["score_id"],
                    "page_id": yolo_row["page_id"],
                    "yolo_line": yolo_row["yolo_line"],
                    "yolo_class": yolo_row["class"],
                    "xml_event_type": event["event_type"],
                    "xml_event_subtype": event["event_subtype"],
                    "relation_type": relation,
                    "bps_note_id": "" if note_id is None else note_id,
                    "link_method": method,
                    "link_status": (
                        "confirmed" if yolo_row.get("human_approved") == "true"
                        else yolo_row["alignment_status"]
                    ),
                    "is_primary": "true" if index == 0 else "false",
                    "confidence": yolo_row.get("confidence", ""),
                    "human_approved": yolo_row.get("human_approved", "false"),
                    "review_status": yolo_row.get("review_status", ""),
                    "source_alignment_status": yolo_row["alignment_status"],
                    "pipeline_version": PIPELINE_VERSION,
                    "validation_status": (
                        "confirmed" if yolo_row.get("human_approved") == "true"
                        else "source_candidate_not_revalidated"
                    ),
                }
            )
    for index, link in enumerate(
        sorted(pending, key=lambda row: (row["bbox_id"], row["is_primary"] != "true", row["xml_event_id"], str(row["bps_note_id"]))),
        start=1,
    ):
        link["link_id"] = f"{link['score_id']}:L{index:08d}"
    return pending


def build_score_combined(
    yolo_fields: list[str],
    yolo_rows: list[dict],
    xml_events: list[dict],
    links: list[dict],
) -> tuple[list[str], list[dict]]:
    yolo_extra_fields = [field for field in yolo_fields if field not in BPS_FIELDS]
    event_extra_fields = [field for field in EVENT_FIELDS if field not in BPS_FIELDS]
    combined_fields = (
        BPS_FIELDS
        + COMBINED_CONTROL_FIELDS
        + yolo_extra_fields
        + [f"event_{field}" for field in event_extra_fields]
    )
    event_by_id = {event["xml_event_id"]: event for event in xml_events}
    links_by_bbox: dict[str, list[dict]] = defaultdict(list)
    for link in links:
        links_by_bbox[link["bbox_id"]].append(link)
    linked_event_ids = {link["xml_event_id"] for link in links}
    combined = []
    for yolo_row in yolo_rows:
        row = {field: "" for field in combined_fields}
        row.update(yolo_row)
        bbox_links = sorted(
            links_by_bbox.get(yolo_row["bbox_id"], []),
            key=lambda link: (link["is_primary"] != "true", link["xml_event_id"]),
        )
        linked_events = [event_by_id[link["xml_event_id"]] for link in bbox_links]
        row.update(
            {
                "record_id": yolo_row["bbox_id"],
                "row_origin": "yolo",
                "source_alignment_status": yolo_row["alignment_status"],
                "xml_event_ids_json": _json([event["xml_event_id"] for event in linked_events]),
                "xml_event_count": len(linked_events),
                "xml_events_json": _json(linked_events),
            }
        )
        if bbox_links:
            row["combined_status"] = (
                "aligned" if yolo_row["alignment_status"] == "matched"
                else yolo_row["alignment_status"]
            )
            primary = event_by_id[bbox_links[0]["xml_event_id"]]
            for field in event_extra_fields:
                row[f"event_{field}"] = primary.get(field, "")
        elif yolo_row["alignment_status"] == "xml_missing":
            row["combined_status"] = "yolo_only"
        else:
            row["combined_status"] = "unresolved"
        combined.append(row)

    for event in xml_events:
        if event["xml_event_id"] in linked_event_ids:
            continue
        row = {field: "" for field in combined_fields}
        for field in BPS_FIELDS:
            row[field] = event.get(field, "")
        row.update(
            {
                "record_id": event["xml_event_id"],
                "row_origin": "xml",
                "combined_status": "xml_only",
                "source_alignment_status": "xml_only",
                "score_id": event["score_id"],
                "xml_path": event["source_xml_path"],
                "xml_page": event["page"],
                "xml_measure": event["xml_measure"],
                "staff": event["staff"],
                "target_type": event["event_type"],
                "xml_event_ids_json": _json([event["xml_event_id"]]),
                "xml_event_count": 1,
                "xml_events_json": _json([event]),
            }
        )
        for field in event_extra_fields:
            row[f"event_{field}"] = event.get(field, "")
        combined.append(row)
    return combined_fields, combined


def combine_dataset(
    yolo_master_path: Path,
    xml_events_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    yolo_fields, yolo_rows = _read_csv(yolo_master_path)
    xml_fields, xml_events = _read_csv(xml_events_path)
    if yolo_fields[: len(BPS_FIELDS)] != BPS_FIELDS:
        raise ValueError("YOLO master does not begin with official BPS-OMR fields")
    if xml_fields[: len(BPS_FIELDS)] != BPS_FIELDS:
        raise ValueError("XML events do not begin with official BPS-OMR fields")

    yolo_by_score: dict[str, list[dict]] = defaultdict(list)
    xml_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in yolo_rows:
        yolo_by_score[row["score_id"]].append(row)
    for row in xml_events:
        xml_by_score[row["score_id"]].append(row)
    score_ids = sorted(set(yolo_by_score) | set(xml_by_score))
    all_links, all_combined = [], []
    combined_fields = []
    score_reports = []
    emit_progress("combine", 0, len(score_ids), f"starting (resume={resume})")
    for index, score_id in enumerate(score_ids, start=1):
        score_dir = output_dir / "per_scores" / score_id
        link_path = score_dir / "alignment_links.csv"
        combined_path = score_dir / "combined_master.csv"
        checkpoint_path = output_dir / "checkpoints" / f"{score_id}.json"
        fingerprint = stable_digest(
            {
                "version": PIPELINE_VERSION,
                "yolo_master": path_signature(yolo_master_path),
                "xml_events": path_signature(xml_events_path),
                "score_id": score_id,
                "yolo_rows": len(yolo_by_score[score_id]),
                "xml_event_rows": len(xml_by_score[score_id]),
            }
        )
        reused = False
        if resume and checkpoint_path.is_file() and link_path.is_file() and combined_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                _link_fields, links = _read_csv(link_path)
                saved_combined_fields, combined = _read_csv(combined_path)
                reused = (
                    checkpoint.get("fingerprint") == fingerprint
                    and checkpoint.get("link_rows") == len(links)
                    and checkpoint.get("combined_rows") == len(combined)
                )
                if reused:
                    combined_fields = saved_combined_fields
            except (OSError, csv.Error, json.JSONDecodeError):
                reused = False
        if not reused:
            links = build_score_links(yolo_by_score[score_id], xml_by_score[score_id])
            combined_fields, combined = build_score_combined(
                yolo_fields, yolo_by_score[score_id], xml_by_score[score_id], links
            )
            atomic_write_csv(link_path, LINK_FIELDS, links)
            atomic_write_csv(combined_path, combined_fields, combined)
            atomic_write_json(
                checkpoint_path,
                {
                    "score_id": score_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "fingerprint": fingerprint,
                    "link_rows": len(links),
                    "combined_rows": len(combined),
                },
            )
        all_links.extend(links)
        all_combined.extend(combined)
        score_reports.append(
            {
                "score_id": score_id,
                "yolo_rows": len(yolo_by_score[score_id]),
                "xml_event_rows": len(xml_by_score[score_id]),
                "link_rows": len(links),
                "combined_rows": len(combined),
                "resumed": reused,
            }
        )
        emit_progress(
            "combine", index, len(score_ids),
            f"{score_id} {'resumed' if reused else 'built'} links={len(links)} combined={len(combined)}",
        )

    atomic_write_csv(output_dir / "alignment_links.csv", LINK_FIELDS, all_links)
    atomic_write_csv(output_dir / "combined_master.csv", combined_fields, all_combined)
    yolo_combined = [row for row in all_combined if row["row_origin"] == "yolo"]
    xml_only = [row for row in all_combined if row["row_origin"] == "xml"]
    linked_event_ids = {row["xml_event_id"] for row in all_links}
    represented_event_ids = linked_event_ids | {
        _json_list(row["xml_event_ids_json"])[0] for row in xml_only
    }
    source_bbox_ids = {row["bbox_id"] for row in yolo_rows}
    combined_bbox_ids = {row["bbox_id"] for row in yolo_combined}
    source_event_ids = {row["xml_event_id"] for row in xml_events}
    combined_yolo_by_id = {row["bbox_id"]: row for row in yolo_combined}
    errors = []
    if len(yolo_rows) != len(source_bbox_ids):
        errors.append("source YOLO master contains duplicate bbox_id")
    if source_bbox_ids != combined_bbox_ids or len(yolo_combined) != len(yolo_rows):
        errors.append("combined master does not preserve every YOLO bbox exactly once")
    if source_event_ids != represented_event_ids:
        errors.append("combined master does not represent every XML event")
    if any(link["bbox_id"] not in source_bbox_ids for link in all_links):
        errors.append("alignment link references missing bbox")
    if any(link["xml_event_id"] not in source_event_ids for link in all_links):
        errors.append("alignment link references missing XML event")
    if len({link["link_id"] for link in all_links}) != len(all_links):
        errors.append("duplicate link_id")
    primary_counts = Counter(
        link["bbox_id"] for link in all_links if link["is_primary"] == "true"
    )
    linked_bbox_ids = {link["bbox_id"] for link in all_links}
    if any(primary_counts[bbox_id] != 1 for bbox_id in linked_bbox_ids):
        errors.append("linked bbox does not have exactly one primary link")
    source_field_changes = 0
    for source_row in yolo_rows:
        combined_row = combined_yolo_by_id.get(source_row["bbox_id"], {})
        if any(combined_row.get(field, "") != source_row[field] for field in yolo_fields):
            source_field_changes += 1
    if source_field_changes:
        errors.append(f"{source_field_changes} YOLO rows changed source field values")
    for row in all_combined:
        for field in ("xml_event_ids_json", "xml_events_json"):
            try:
                json.loads(row[field] or "[]")
            except json.JSONDecodeError:
                errors.append(f"invalid {field}: {row['record_id']}")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "scores": len(score_ids),
        "yolo_source_rows": len(yolo_rows),
        "unique_bbox_ids": len(source_bbox_ids),
        "xml_event_source_rows": len(xml_events),
        "xml_events_represented": len(represented_event_ids),
        "alignment_link_rows": len(all_links),
        "linked_bbox_rows": len({row["bbox_id"] for row in all_links}),
        "linked_xml_event_rows": len(linked_event_ids),
        "combined_rows": len(all_combined),
        "row_origin_counts": dict(Counter(row["row_origin"] for row in all_combined)),
        "combined_status_counts": dict(Counter(row["combined_status"] for row in all_combined)),
        "link_method_counts": dict(Counter(row["link_method"] for row in all_links)),
        "link_status_counts": dict(Counter(row["link_status"] for row in all_links)),
        "score_reports": score_reports,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "alignment_links": str(output_dir / "alignment_links.csv"),
            "combined_master": str(output_dir / "combined_master.csv"),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("combine-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-master", type=Path, required=True)
    parser.add_argument("--xml-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = combine_dataset(
        args.yolo_master,
        args.xml_events,
        args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
