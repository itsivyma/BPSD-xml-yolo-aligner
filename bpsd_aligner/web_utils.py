"""Dependency-light helpers used by the Streamlit app and tests."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import PurePath


def filename_page_number(stem: str) -> int | None:
    match = re.search(r"(?:^|[-_])(\d+)$", stem)
    return int(match.group(1)) if match else None


def pair_page_uploads(
    image_uploads: list,
    yolo_uploads: list,
    *,
    first_page: int,
    infer_page_from_filename: bool,
) -> list[dict]:
    """Pair page images and YOLO TXT files by exact filename stem."""

    def unique_by_stem(uploads: list, kind: str) -> dict[str, object]:
        output = {}
        for uploaded in uploads:
            stem = PurePath(uploaded.name).stem
            if stem in output:
                raise ValueError(f"Duplicate {kind} filename stem: {stem}")
            output[stem] = uploaded
        return output

    images = unique_by_stem(image_uploads, "image")
    yolos = unique_by_stem(yolo_uploads, "YOLO")
    missing_yolo = sorted(set(images) - set(yolos))
    missing_image = sorted(set(yolos) - set(images))
    if missing_yolo or missing_image:
        messages = []
        if missing_yolo:
            messages.append("missing YOLO TXT for: " + ", ".join(missing_yolo))
        if missing_image:
            messages.append("missing image for: " + ", ".join(missing_image))
        raise ValueError("; ".join(messages))

    stems = sorted(
        images,
        key=lambda stem: [
            (0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", stem)
        ],
    )
    pages = []
    used_page_numbers = set()
    for index, stem in enumerate(stems):
        inferred = filename_page_number(stem) if infer_page_from_filename else None
        page_number = inferred if inferred is not None else first_page + index
        if page_number in used_page_numbers:
            raise ValueError(f"Duplicate MusicXML page number: {page_number}")
        used_page_numbers.add(page_number)
        pages.append(
            {
                "stem": stem,
                "page_number": page_number,
                "image": images[stem],
                "yolo": yolos[stem],
            }
        )
    return pages


def parse_system_start_measures(value: object) -> list[int]:
    """Parse optional comma/space-separated printed system measure anchors."""

    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    parts = [part for part in re.split(r"[,，、;；\s]+", text) if part]
    try:
        anchors = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("Scan system starts must contain measure numbers only") from error
    if any(anchor < 1 for anchor in anchors):
        raise ValueError("Scan system starts must be positive measure numbers")
    if anchors != sorted(set(anchors)):
        raise ValueError("Scan system starts must be strictly increasing")
    return anchors


def apply_page_mapping_edits(page_pairs: list[dict], edited_rows: list[dict]) -> list[dict]:
    """Apply page numbers and optional scanned-system anchors safely."""

    expected = {pair["stem"] for pair in page_pairs}
    provided = {str(row.get("page stem", "")) for row in edited_rows}
    if provided != expected:
        raise ValueError("Page pairing rows changed unexpectedly; rebuild the pairing table")
    page_by_stem = {}
    anchors_by_stem = {}
    page_end_by_stem = {}
    used = set()
    for row in edited_rows:
        stem = str(row["page stem"])
        raw = row.get("MusicXML page")
        try:
            numeric = float(raw)
            page = int(numeric)
        except (TypeError, ValueError):
            raise ValueError(f"MusicXML page for {stem} must be a positive integer")
        if numeric != page or page < 1:
            raise ValueError(f"MusicXML page for {stem} must be a positive integer")
        if page in used:
            raise ValueError(f"Duplicate MusicXML page number: {page}")
        used.add(page)
        page_by_stem[stem] = page
        try:
            anchors_by_stem[stem] = parse_system_start_measures(
                row.get("Scan system starts", "")
            )
        except ValueError as error:
            raise ValueError(f"{stem}: {error}") from error
        raw_end = row.get("Scan page end", "")
        if raw_end is None or str(raw_end).strip().lower() in {"", "nan"}:
            page_end_by_stem[stem] = None
        else:
            try:
                numeric_end = float(raw_end)
                page_end = int(numeric_end)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Scan page end for {stem} must be a positive integer"
                )
            if numeric_end != page_end or page_end < 1:
                raise ValueError(
                    f"Scan page end for {stem} must be a positive integer"
                )
            if anchors_by_stem[stem] and page_end < anchors_by_stem[stem][-1]:
                raise ValueError(
                    f"Scan page end for {stem} cannot precede its last system start"
                )
            page_end_by_stem[stem] = page_end

    output = [
        {
            **pair,
            "page_number": page_by_stem[pair["stem"]],
            "system_start_measures": anchors_by_stem[pair["stem"]],
            "page_end_measure": page_end_by_stem[pair["stem"]],
        }
        for pair in page_pairs
    ]
    ordered = sorted(output, key=lambda pair: pair["page_number"])
    for index, pair in enumerate(ordered[:-1]):
        following = ordered[index + 1]
        if (
            pair["page_end_measure"] is None
            and pair["system_start_measures"]
            and following["system_start_measures"]
            and following["page_number"] == pair["page_number"] + 1
        ):
            pair["page_end_measure"] = following["system_start_measures"][0] - 1
    return output


def apply_page_number_edits(page_pairs: list[dict], edited_rows: list[dict]) -> list[dict]:
    """Backward-compatible wrapper for callers editing only XML page numbers."""

    return apply_page_mapping_edits(page_pairs, edited_rows)


def read_csv_bytes(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "scores": len({row.get("score_id", "") for row in rows if row.get("score_id")}),
        "origins": dict(Counter(row.get("row_origin", "unknown") or "unknown" for row in rows)),
        "statuses": dict(Counter(row.get("combined_status", "unknown") or "unknown" for row in rows)),
    }


def group_review_overlays(
    overlays: dict[str, bytes],
    detailed_rows: list[dict[str, str]],
) -> dict[str, dict[str, object]]:
    """Group current multi-page overlay keys into a page-first UI model."""

    needs_review = Counter(
        row.get("page_id", "")
        for row in detailed_rows
        if row.get("page_id") and row.get("status") != "matched"
    )
    pages: dict[str, dict[str, object]] = {}

    def page_entry(page_id: str) -> dict[str, object]:
        return pages.setdefault(
            page_id,
            {
                "review_overlay": None,
                "classes": {},
                "needs_review": needs_review.get(page_id, 0),
            },
        )

    for name, data in overlays.items():
        if name.startswith("class_"):
            encoded = name.removeprefix("class_")
            if "__" not in encoded:
                continue
            page_id, overlay_name = encoded.split("__", 1)
            if not overlay_name.endswith("_overlay"):
                continue
            class_name = overlay_name.removesuffix("_overlay")
            classes = page_entry(page_id)["classes"]
            assert isinstance(classes, dict)
            classes[class_name] = data
            continue

        if "__" not in name:
            continue
        page_id, overlay_name = name.split("__", 1)
        if overlay_name == "review_overlay":
            page_entry(page_id)["review_overlay"] = data

    return pages


def sanitize_path_columns(data: bytes) -> bytes:
    fields, rows = read_csv_bytes(data)
    path_fields = [field for field in fields if field == "source_xml_path" or field.endswith("_path")]
    for row in rows:
        for field in path_fields:
            value = row.get(field, "")
            row[field] = PurePath(value).name if value else ""
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")
