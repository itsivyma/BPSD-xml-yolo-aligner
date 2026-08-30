"""Render one lossless full-resolution annotated scan per audit sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_checkpoint import (
    atomic_save_png,
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
    valid_png,
)


PIPELINE_VERSION = "0.3.0-audit-fullres"
INDEX_FIELDS = [
    "sample_id", "selection_reason", "bbox_id", "class", "confidence", "source_image",
    "source_width", "source_height", "fullres_png", "render_status",
]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf") if bold else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def render_fullres(source_path: Path, row: dict, audit: dict, output_path: Path) -> tuple[int, int]:
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    width, height = source.size
    banner = 92
    canvas = Image.new("RGB", (width, height + banner), "white")
    canvas.paste(source, (0, banner))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 12),
        f"{audit['sample_id']} | {audit.get('selection_reason', '')} | {audit['bbox_id']} | {audit['class']} | confidence {audit['confidence']}",
        fill="#111827",
        font=_font(27, True),
    )
    draw.text(
        (20, 52),
        "Blue = YOLO box; green = proposed XML/BPSD target. Original scan pixels are not resized.",
        fill="#334155",
        font=_font(20),
    )
    x = float(row["x"]) * width
    y = float(row["y"]) * height + banner
    box_width = float(row["w"]) * width
    box_height = float(row["h"]) * height
    rectangle = (
        x - box_width / 2,
        y - box_height / 2,
        x + box_width / 2,
        y + box_height / 2,
    )
    draw.rectangle(rectangle, outline="#1261FF", width=5)
    draw.text(
        (max(0, rectangle[0]), max(banner, rectangle[1] - 30)),
        audit["sample_id"],
        fill="#1261FF",
        font=_font(22, True),
    )
    if row.get("target_x_px") not in {"", "NA"} and row.get("target_y_px") not in {"", "NA"}:
        target_x = float(row["target_x_px"])
        target_y = float(row["target_y_px"]) + banner
        draw.line((x, y, target_x, target_y), fill="#00A63C", width=4)
        radius = 18
        draw.ellipse(
            (target_x - radius, target_y - radius, target_x + radius, target_y + radius),
            outline="#00A63C",
            width=6,
        )
        draw.text(
            (target_x + 22, target_y - 25),
            "XML target",
            fill="#008A32",
            font=_font(20, True),
        )
    atomic_save_png(canvas, output_path, optimize=True)
    source.close()
    return width, height


def render_dataset(
    audit_csv: Path,
    combined_master_csv: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    audit_rows = _read(audit_csv)
    master_rows = _read(combined_master_csv)
    has_row_origin = any("row_origin" in row for row in master_rows)
    master = {
        row["bbox_id"]: row
        for row in master_rows
        if not has_row_origin or row.get("row_origin") == "yolo"
    }
    images_dir = output_dir / "images"
    checkpoints_dir = output_dir / "checkpoints"
    index_rows = []
    reused_count = 0
    emit_progress("audit-fullres", 0, len(audit_rows), f"starting (resume={resume})")
    for index, audit in enumerate(audit_rows, start=1):
        row = master[audit["bbox_id"]]
        source_path = Path(row["image_path"])
        safe_bbox = audit["bbox_id"].replace(":", "_")
        output_path = images_dir / f"{audit['sample_id']}_{safe_bbox}.png"
        checkpoint_path = checkpoints_dir / f"{audit['sample_id']}.json"
        fingerprint = stable_digest(
            {
                "version": PIPELINE_VERSION,
                "source": path_signature(source_path),
                "sample_id": audit["sample_id"],
                "bbox_id": audit["bbox_id"],
                "selection_reason": audit.get("selection_reason", ""),
                "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"],
                "target_x_px": row.get("target_x_px", ""),
                "target_y_px": row.get("target_y_px", ""),
            }
        )
        reused = False
        width = height = 0
        if resume and checkpoint_path.is_file() and valid_png(output_path):
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                reused = checkpoint.get("fingerprint") == fingerprint
                width = int(checkpoint.get("source_width", 0))
                height = int(checkpoint.get("source_height", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                reused = False
        if not reused:
            width, height = render_fullres(source_path, row, audit, output_path)
            atomic_write_json(
                checkpoint_path,
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "sample_id": audit["sample_id"],
                    "fingerprint": fingerprint,
                    "source_width": width,
                    "source_height": height,
                },
            )
        else:
            reused_count += 1
        index_rows.append(
            {
                "sample_id": audit["sample_id"],
                "selection_reason": audit.get("selection_reason", ""),
                "bbox_id": audit["bbox_id"],
                "class": audit["class"],
                "confidence": audit["confidence"],
                "source_image": str(source_path),
                "source_width": width,
                "source_height": height,
                "fullres_png": str(output_path.resolve()),
                "render_status": "resumed" if reused else "rendered",
            }
        )
        if index == len(audit_rows) or index % 5 == 0:
            emit_progress(
                "audit-fullres", index, len(audit_rows),
                f"{audit['sample_id']} {'resumed' if reused else 'rendered'}",
            )
    atomic_write_csv(output_dir / "image_index.csv", INDEX_FIELDS, index_rows)
    errors = []
    if len(index_rows) != len(audit_rows):
        errors.append("full-resolution index row count differs from audit sample")
    if any(not valid_png(Path(row["fullres_png"])) for row in index_rows):
        errors.append("one or more full-resolution PNGs are invalid")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "sample_rows": len(audit_rows),
        "fullres_pngs": len(index_rows),
        "resumed_pngs": reused_count,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "images_dir": str(images_dir),
            "image_index": str(output_dir / "image_index.csv"),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("audit-fullres-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--combined-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = render_dataset(
        args.audit_csv,
        args.combined_master,
        args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
