"""Render native-pixel audit crops and crisp nearest-neighbour detail views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_checkpoint import atomic_write_csv, emit_progress


PIPELINE_VERSION = "0.1.0-audit-crops"


def _font(size: int) -> ImageFont.ImageFont:
    for path in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bounded_crop(cx: int, cy: int, width: int, height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left = max(0, min(image_width - width, cx - width // 2))
    top = max(0, min(image_height - height, cy - height // 2))
    return left, top, min(image_width, left + width), min(image_height, top + height)


def render(audit_csv: Path, master_csv: Path, output_dir: Path, sample_ids: set[str], *, resume: bool) -> dict:
    with audit_csv.open(newline="", encoding="utf-8-sig") as file:
        audit = {row["bbox_id"]: row for row in csv.DictReader(file) if not sample_ids or row["sample_id"] in sample_ids}
    master: dict[str, dict[str, str]] = {}
    with master_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("bbox_id") in audit:
                master[row["bbox_id"]] = row
    missing = sorted(set(audit) - set(master))
    if missing:
        raise ValueError(f"bbox IDs missing from master CSV: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    ordered = sorted(audit.values(), key=lambda row: row["sample_id"])
    for index, item in enumerate(ordered, start=1):
        row = master[item["bbox_id"]]
        sample_id = item["sample_id"]
        context_path = output_dir / f"{sample_id}_context_native.png"
        detail_path = output_dir / f"{sample_id}_detail_4x_nearest.png"
        if resume and context_path.is_file() and detail_path.is_file():
            status = "resumed"
        else:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            iw, ih = image.size
            cx, cy = round(float(row["x"]) * iw), round(float(row["y"]) * ih)
            bw = max(2, round(float(row["w"]) * iw))
            bh = max(2, round(float(row["h"]) * ih))

            box = _bounded_crop(cx, cy, min(900, iw), min(650, ih), iw, ih)
            crop = image.crop(box)
            header = 72
            canvas = Image.new("RGB", (crop.width, crop.height + header), "white")
            canvas.paste(crop, (0, header))
            draw = ImageDraw.Draw(canvas)
            local_x, local_y = cx - box[0], cy - box[1] + header
            pad = 9
            draw.rectangle(
                (local_x - bw // 2 - pad, local_y - bh // 2 - pad, local_x + bw // 2 + pad, local_y + bh // 2 + pad),
                outline=(0, 90, 255), width=4,
            )
            draw.line((local_x, 55, local_x, max(header, local_y - bh // 2 - pad)), fill=(0, 90, 255), width=3)
            draw.text((14, 9), f"{sample_id} | {row['class']} | BLUE BOX = YOLO MARK", fill=(0, 35, 95), font=_font(27))
            draw.text((14, 42), "Native source pixels; open at 100% in VS Code.", fill=(20, 20, 20), font=_font(18))
            canvas.save(context_path)

            detail_box = _bounded_crop(cx, cy, min(180, iw), min(130, ih), iw, ih)
            detail = image.crop(detail_box).resize((720, 520), Image.Resampling.NEAREST)
            detail_canvas = Image.new("RGB", (720, 580), "white")
            detail_canvas.paste(detail, (0, 60))
            detail_draw = ImageDraw.Draw(detail_canvas)
            dx = (cx - detail_box[0]) * 4
            dy = (cy - detail_box[1]) * 4 + 60
            detail_draw.rectangle(
                (dx - bw * 2 - 12, dy - bh * 2 - 12, dx + bw * 2 + 12, dy + bh * 2 + 12),
                outline=(0, 90, 255), width=5,
            )
            detail_draw.text((14, 10), f"{sample_id} | 4x nearest-neighbour detail", fill=(0, 35, 95), font=_font(28))
            detail_canvas.save(detail_path)
            status = "rendered"
        records.append({
            "sample_id": sample_id,
            "bbox_id": item["bbox_id"],
            "context_native_png": str(context_path.resolve()),
            "detail_4x_nearest_png": str(detail_path.resolve()),
            "render_status": status,
        })
        emit_progress("audit-crops", index, len(ordered), f"{sample_id} {status}")
    atomic_write_csv(output_dir / "crop_index.csv", list(records[0]) if records else [], records)
    return {"pipeline_version": PIPELINE_VERSION, "rendered_samples": len(records), "missing": missing, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = render(args.audit_csv, args.master_csv, args.output_dir, set(args.sample_id), resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
