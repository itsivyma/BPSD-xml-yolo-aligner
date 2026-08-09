"""Build a scan-system to repetition-MusicXML scope manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from bps_xml_alignment import (
    assign_system,
    detect_systems,
    load_categories,
    load_yolo,
    parse_musicxml_page,
)


FIELDS = [
    "score_id",
    "page_id",
    "scan_page",
    "scan_system_index",
    "bbox_count",
    "movement_scope_status",
    "page_mapping_status",
    "xml_page",
    "xml_systems_json",
    "written_measure_start",
    "written_measure_end",
    "mapping_source",
    "review_status",
]


def resolve_system_mapping(
    *,
    page_id: str,
    scan_page: int,
    scan_system: int,
    xml_system_count: int,
    overrides: dict,
) -> dict:
    page_override = overrides.get(page_id, {})
    override = page_override.get(str(scan_system))
    if override is not None:
        if override.get("movement_scope_status") == "outside_bpsd_scope":
            return {
                "movement_scope_status": "outside_bpsd_scope",
                "page_mapping_status": override.get("mapping_status", "xml_missing"),
                "xml_page": "",
                "xml_systems": [],
                "mapping_source": "scope_override",
                "review_status": "not_required",
            }
        return {
            "movement_scope_status": "in_bpsd_scope",
            "page_mapping_status": override.get("mapping_status", "direct"),
            "xml_page": int(override.get("xml_page", scan_page)),
            "xml_systems": [int(value) for value in override["xml_systems"]],
            "mapping_source": "scope_override",
            "review_status": (
                "needs_review"
                if override.get("mapping_status") != "direct"
                else "not_required"
            ),
        }
    if 1 <= scan_system <= xml_system_count:
        return {
            "movement_scope_status": "in_bpsd_scope",
            "page_mapping_status": "direct",
            "xml_page": scan_page,
            "xml_systems": [scan_system],
            "mapping_source": "page_and_system_identity",
            "review_status": "not_required",
        }
    return {
        "movement_scope_status": "outside_bpsd_scope",
        "page_mapping_status": "xml_missing",
        "xml_page": "",
        "xml_systems": [],
        "mapping_source": "scan_system_beyond_xml_scope",
        "review_status": "not_required",
    }


def _xml_page(xml_path: Path, page: int) -> dict | None:
    try:
        return parse_musicxml_page(xml_path, page_number=page)
    except ValueError as error:
        if "does not exist" in str(error):
            return None
        raise


def build_scope_manifest(
    *, manifest_path: Path, notes_json_path: Path, config_path: Path
) -> tuple[list[dict], dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = config.get("overrides", {})
    categories = load_categories(notes_json_path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        pages = list(csv.DictReader(file))

    rows = []
    total_boxes = 0
    outside_boxes = 0
    review_boxes = 0
    xml_cache: dict[tuple[str, int], dict | None] = {}
    for page in pages:
        scan_page = int(page["page"])
        image = Image.open(page["image_path"]).convert("RGB")
        systems = detect_systems(image)
        boxes = load_yolo(Path(page["yolo_path"]), categories=categories)
        counts = {system.number: 0 for system in systems}
        for box in boxes:
            system = assign_system(box, systems, image.height)
            counts[system] += 1
        total_boxes += len(boxes)

        cache_key = (page["xml_path"], scan_page)
        if cache_key not in xml_cache:
            xml_cache[cache_key] = _xml_page(Path(page["xml_path"]), scan_page)
        xml_page = xml_cache[cache_key]
        xml_system_count = (
            max((int(m["system"]) for m in xml_page["measures"]), default=0)
            if xml_page is not None
            else 0
        )
        measures_by_system: dict[int, list[int]] = {}
        if xml_page is not None:
            for measure in xml_page["measures"]:
                measures_by_system.setdefault(int(measure["system"]), []).append(
                    int(measure["measure"])
                )

        for system in systems:
            mapping = resolve_system_mapping(
                page_id=page["page_id"],
                scan_page=scan_page,
                scan_system=system.number,
                xml_system_count=xml_system_count,
                overrides=overrides,
            )
            mapped_measures = [
                measure
                for xml_system in mapping["xml_systems"]
                for measure in measures_by_system.get(xml_system, [])
            ]
            bbox_count = counts[system.number]
            if mapping["movement_scope_status"] == "outside_bpsd_scope":
                outside_boxes += bbox_count
            if mapping["review_status"] == "needs_review":
                review_boxes += bbox_count
            rows.append(
                {
                    "score_id": page["score_id"],
                    "page_id": page["page_id"],
                    "scan_page": scan_page,
                    "scan_system_index": system.number,
                    "bbox_count": bbox_count,
                    "movement_scope_status": mapping["movement_scope_status"],
                    "page_mapping_status": mapping["page_mapping_status"],
                    "xml_page": mapping["xml_page"],
                    "xml_systems_json": json.dumps(mapping["xml_systems"]),
                    "written_measure_start": min(mapped_measures) if mapped_measures else "",
                    "written_measure_end": max(mapped_measures) if mapped_measures else "",
                    "mapping_source": mapping["mapping_source"],
                    "review_status": mapping["review_status"],
                }
            )

    report = {
        "schema_version": config.get("schema_version"),
        "pages": len(pages),
        "scan_systems": len(rows),
        "total_boxes": total_boxes,
        "outside_bpsd_scope_boxes": outside_boxes,
        "mapping_review_boxes": review_boxes,
        "errors": [],
    }
    return rows, report


def write_scope_manifest(
    rows: list[dict], report: dict, output_csv: Path, output_json: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows, report = build_scope_manifest(
        manifest_path=args.manifest,
        notes_json_path=args.notes_json,
        config_path=args.config,
    )
    write_scope_manifest(rows, report, args.output_csv, args.output_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
