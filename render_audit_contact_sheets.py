"""Render audit-sample review images into an overview and readable contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pipeline_checkpoint import (
    atomic_save_png,
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
    valid_png,
)


PIPELINE_VERSION = "0.1.0-audit-contact-sheets"
INDEX_FIELDS = [
    "contact_page", "tile_index", "sample_ids_json", "bbox_ids_json",
    "classes_json", "confidences_json", "source_review_png",
    "contact_sheet_png",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def group_samples(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["review_png"]].append(row)
    result = []
    for path, members in grouped.items():
        result.append(
            {
                "source_review_png": path,
                "sample_ids": [row["sample_id"] for row in members],
                "bbox_ids": [row["bbox_id"] for row in members],
                "classes": sorted({row["class"] for row in members}),
                "confidences": [row["confidence"] for row in members],
            }
        )
    return sorted(result, key=lambda item: item["sample_ids"])


def render_sheet(
    items: list[dict],
    path: Path,
    *,
    columns: int,
    tile_width: int,
    image_height: int,
    label_height: int,
    title: str,
) -> None:
    margin, gap, title_height = 24, 18, 58
    rows = max(1, math.ceil(len(items) / columns))
    tile_height = image_height + label_height
    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = margin * 2 + title_height + rows * tile_height + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill="#111827", font=_font(28))
    for index, item in enumerate(items):
        row_index, column_index = divmod(index, columns)
        left = margin + column_index * (tile_width + gap)
        top = margin + title_height + row_index * (tile_height + gap)
        image_box = (left, top, left + tile_width, top + image_height)
        draw.rectangle(image_box, outline="#94A3B8", width=2)
        with Image.open(item["source_review_png"]) as source:
            fitted = ImageOps.contain(
                source.convert("RGB"),
                (tile_width - 8, image_height - 8),
                method=Image.Resampling.LANCZOS,
            )
        image_left = left + (tile_width - fitted.width) // 2
        image_top = top + (image_height - fitted.height) // 2
        canvas.paste(fitted, (image_left, image_top))
        label_top = top + image_height
        sample_text = ", ".join(item["sample_ids"])
        bbox_text = ", ".join(item["bbox_ids"])
        detail_text = (
            f"{sample_text} | {', '.join(item['classes'])} | "
            f"confidence {', '.join(item['confidences'])}"
        )
        draw.text((left + 4, label_top + 5), detail_text, fill="#111827", font=_font(19))
        draw.text((left + 4, label_top + 32), bbox_text, fill="#334155", font=_font(17))
    atomic_save_png(canvas, path, optimize=True)


def render_audit_sheets(
    audit_csv: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    rows = _read_csv(audit_csv)
    items = group_samples(rows)
    source_signatures = [path_signature(Path(item["source_review_png"])) for item in items]
    fingerprint = stable_digest(
        {
            "version": PIPELINE_VERSION,
            "audit_csv": path_signature(audit_csv),
            "sources": source_signatures,
        }
    )
    overview_path = output_dir / "audit_sample_overview.png"
    pages_dir = output_dir / "contact_pages"
    index_path = output_dir / "image_index.csv"
    checkpoint_path = output_dir / "checkpoint.json"
    page_size = 6
    page_count = math.ceil(len(items) / page_size)
    page_paths = [pages_dir / f"audit_contact_part{index:02d}.png" for index in range(1, page_count + 1)]
    reused = False
    if resume and checkpoint_path.is_file() and index_path.is_file():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            index_rows = _read_csv(index_path)
            reused = (
                checkpoint.get("fingerprint") == fingerprint
                and checkpoint.get("source_images") == len(items)
                and checkpoint.get("sample_rows") == len(rows)
                and valid_png(overview_path)
                and all(valid_png(path) for path in page_paths)
                and len(index_rows) == len(items)
            )
        except (OSError, csv.Error, json.JSONDecodeError):
            reused = False
    if not reused:
        emit_progress("audit-contact", 0, page_count + 1, "rendering overview")
        render_sheet(
            items,
            overview_path,
            columns=5,
            tile_width=500,
            image_height=570,
            label_height=70,
            title=f"Audit sample overview: {len(rows)} samples / {len(items)} review images",
        )
        index_rows = []
        for page_index, page_path in enumerate(page_paths, start=1):
            page_items = items[(page_index - 1) * page_size: page_index * page_size]
            render_sheet(
                page_items,
                page_path,
                columns=2,
                tile_width=760,
                image_height=850,
                label_height=82,
                title=f"Audit contact sheet {page_index}/{page_count}",
            )
            for tile_index, item in enumerate(page_items, start=1):
                index_rows.append(
                    {
                        "contact_page": page_index,
                        "tile_index": tile_index,
                        "sample_ids_json": json.dumps(item["sample_ids"]),
                        "bbox_ids_json": json.dumps(item["bbox_ids"]),
                        "classes_json": json.dumps(item["classes"]),
                        "confidences_json": json.dumps(item["confidences"]),
                        "source_review_png": item["source_review_png"],
                        "contact_sheet_png": str(page_path.resolve()),
                    }
                )
            emit_progress(
                "audit-contact", page_index, page_count + 1,
                f"part {page_index}/{page_count} images={len(page_items)}",
            )
        atomic_write_csv(index_path, INDEX_FIELDS, index_rows)
        atomic_write_json(
            checkpoint_path,
            {
                "pipeline_version": PIPELINE_VERSION,
                "fingerprint": fingerprint,
                "sample_rows": len(rows),
                "source_images": len(items),
                "contact_pages": page_count,
            },
        )
        emit_progress("audit-contact", page_count + 1, page_count + 1, "render complete")

    covered_samples = {
        sample_id
        for index_row in index_rows
        for sample_id in json.loads(index_row["sample_ids_json"])
    }
    expected_samples = {row["sample_id"] for row in rows}
    errors = []
    if covered_samples != expected_samples:
        errors.append("contact sheets do not cover every audit sample")
    if not valid_png(overview_path):
        errors.append("overview PNG is invalid")
    if any(not valid_png(path) for path in page_paths):
        errors.append("one or more contact page PNGs are invalid")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "sample_rows": len(rows),
        "unique_source_images": len(items),
        "contact_pages": page_count,
        "resumed": reused,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "overview": str(overview_path),
            "contact_pages_dir": str(pages_dir),
            "image_index": str(index_path),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("audit-contact-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = render_audit_sheets(args.audit_csv, args.output_dir, resume=args.resume)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
