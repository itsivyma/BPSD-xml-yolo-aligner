"""Render batch human-review sheets from canonical alignment candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_checkpoint import (
    atomic_save_png,
    atomic_write_csv,
    atomic_write_text,
    emit_progress,
    valid_png,
)


INDEX_FIELDS = ["page_id", "class", "part", "candidates", "png", "csv"]
REVIEW_FIELDS = [
    "bbox_id", "page_id", "yolo_line", "class_id", "class",
    "alignment_status", "start_meas", "end_meas", "note_ids", "pitches",
    "confidence", "review_status", "human_approved", "corrected_value_json",
    "comment",
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else []
    ) + [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _json_summary(raw: str, empty: str = "") -> str:
    if not raw:
        return empty
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(values, list):
        return "+".join(str(value) for value in values)
    return str(values)


def _render_panel(image: Image.Image, row: dict) -> Image.Image:
    panel_width, panel_height, header = 720, 390, 125
    width, height = image.size
    bx0 = (float(row["x"]) - float(row["w"]) / 2) * width
    by0 = (float(row["y"]) - float(row["h"]) / 2) * height
    bx1 = (float(row["x"]) + float(row["w"]) / 2) * width
    by1 = (float(row["y"]) + float(row["h"]) / 2) * height
    has_target = row.get("target_x_px") not in {"", "NA"} and row.get("target_y_px") not in {"", "NA"}
    tx = float(row["target_x_px"]) if has_target else (bx0 + bx1) / 2
    ty = float(row["target_y_px"]) if has_target else (by0 + by1) / 2
    crop = (
        max(0, round(min(bx0, tx) - 210)),
        max(0, round(min(by0, ty) - 125)),
        min(width, round(max(bx1, tx) + 210)),
        min(height, round(max(by1, ty) + 125)),
    )
    cropped = image.crop(crop)
    content_height = panel_height - header
    scale = min(panel_width / cropped.width, content_height / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (panel_width, panel_height), "white")
    offset_x = (panel_width - resized.width) // 2
    offset_y = header + (content_height - resized.height) // 2
    panel.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(panel)

    pitches = _json_summary(row.get("pitches", ""), "-")
    note_ids = _json_summary(row.get("note_ids", ""), "-")
    repeat = row.get("repeat_occurrence_count") or "1"
    draw.text((16, 8), f"Y{row['yolo_line']} {row['class']}", fill="black", font=_font(24, True))
    draw.text(
        (16, 40),
        f"XML m.{row.get('xml_measure') or '?'}; {pitches}; BPSD {row.get('start_meas') or '?'}; staff {row.get('staff') or '?'}",
        fill="black", font=_font(17, True),
    )
    draw.text(
        (16, 68),
        f"note IDs {note_ids}; repeat occurrences {repeat}; confidence {row.get('confidence') or '?'}",
        fill="#1261ff", font=_font(15, True),
    )
    target_text = "green = proposed target" if has_target else "no point target; candidate is measure-position based"
    draw.text(
        (16, 94),
        f"blue = YOLO; {target_text}; status {row['alignment_status']}",
        fill="#1261ff" if has_target else "#d26a00", font=_font(14, True),
    )

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (x - crop[0]) * scale,
            offset_y + (y - crop[1]) * scale,
        )

    px0, py0 = point(bx0, by0)
    px1, py1 = point(bx1, by1)
    draw.rectangle((px0, py0, px1, py1), outline="#1261ff", width=3)
    if has_target:
        ptx, pty = point(tx, ty)
        draw.ellipse((ptx - 13, pty - 13, ptx + 13, pty + 13), outline="#00a63c", width=4)
        draw.line(((px0 + px1) / 2, (py0 + py1) / 2, ptx, pty), fill="#00a63c", width=2)
        draw.text((ptx - 10, pty - 40), "N", fill="#00a63c", font=_font(18, True))
    return panel


def _review_csv_matches(path: Path, rows: list[dict]) -> bool:
    if not path.is_file():
        return False
    try:
        existing = _read(path)
    except (OSError, csv.Error):
        return False
    if len(existing) != len(rows):
        return False
    return all(
        all(
            str(actual.get(field, "")) == str(expected.get(field, ""))
            for field in REVIEW_FIELDS
        )
        for actual, expected in zip(existing, rows, strict=True)
    )


def generate_review_sheets(
    master_csv: Path,
    output_dir: Path,
    chunk_size: int = 8,
    *,
    resume: bool = False,
    progress_every: int = 10,
) -> dict:
    rows = [
        row for row in _read(master_csv)
        if row["movement_scope_status"] == "in_bpsd_scope"
        and row["human_approved"].lower() != "true"
        and row["alignment_status"] in {"candidate", "ambiguous", "matched"}
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["page_id"], row["class"])].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    planned_sheets = sum(math.ceil(len(items) / chunk_size) for items in groups.values())
    emit_progress(
        "review-sheets",
        0,
        planned_sheets,
        f"starting ({len(rows)} candidate panels; resume={resume})",
    )
    index_rows = []
    rendered = 0
    reused_sheets = 0
    completed_sheets = 0
    for (page_id, class_name), items in sorted(groups.items()):
        items.sort(key=lambda row: int(row["yolo_line"]))
        source_image = None
        input_mtime_ns = max(
            master_csv.stat().st_mtime_ns,
            Path(items[0]["image_path"]).stat().st_mtime_ns,
        )
        for part, start in enumerate(range(0, len(items), chunk_size), start=1):
            chunk = items[start:start + chunk_size]
            stem = f"{page_id}_{class_name}_part{part:02d}"
            png_rel = Path(page_id) / f"{stem}.png"
            csv_rel = Path(page_id) / f"{stem}.csv"
            png_path = output_dir / png_rel
            csv_path = output_dir / csv_rel
            reused = (
                resume
                and valid_png(png_path)
                and _review_csv_matches(csv_path, chunk)
                and png_path.stat().st_mtime_ns >= input_mtime_ns
                and csv_path.stat().st_mtime_ns >= master_csv.stat().st_mtime_ns
            )
            if reused:
                reused_sheets += 1
            else:
                if source_image is None:
                    with Image.open(items[0]["image_path"]) as opened:
                        source_image = opened.convert("RGB")
                columns = 2
                sheet_rows = math.ceil(len(chunk) / columns)
                canvas = Image.new(
                    "RGB",
                    (1460, 20 + sheet_rows * 400),
                    "#d7d7d7",
                )
                for index, row in enumerate(chunk):
                    panel = _render_panel(source_image, row)
                    canvas.paste(
                        panel,
                        (10 + (index % 2) * 720, 10 + (index // 2) * 400),
                    )
                atomic_save_png(canvas, png_path)
                atomic_write_csv(csv_path, REVIEW_FIELDS, chunk)
            index_rows.append({
                "page_id": page_id, "class": class_name, "part": part,
                "candidates": len(chunk), "png": png_rel.as_posix(), "csv": csv_rel.as_posix(),
            })
            rendered += len(chunk)
            completed_sheets += 1
            if (
                completed_sheets == planned_sheets
                or completed_sheets % max(1, progress_every) == 0
            ):
                action = "reused" if reused else "rendered"
                emit_progress(
                    "review-sheets",
                    completed_sheets,
                    planned_sheets,
                    f"{page_id} {class_name} part {part:02d} {action}",
                )
        if source_image is not None:
            source_image.close()

    atomic_write_csv(
        output_dir / "review_sheet_index.csv",
        INDEX_FIELDS,
        index_rows,
    )
    cards = "\n".join(
        f'<article><h2>{html.escape(row["page_id"])} · {html.escape(row["class"])} · part {row["part"]}</h2><p>{row["candidates"]} candidates · <a href="{html.escape(row["csv"])}">review CSV</a></p><a href="{html.escape(row["png"])}"><img loading="lazy" src="{html.escape(row["png"])}"></a></article>'
        for row in index_rows
    )
    gallery = f"""<!doctype html><meta charset="utf-8"><title>Alignment review sheets</title>
<style>body{{font-family:system-ui;margin:22px;background:#eee}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}article{{background:white;padding:12px}}h2{{font-size:16px}}img{{width:100%;height:auto}}</style>
<h1>Alignment review sheets</h1><p>Blue = YOLO; green = proposed target. Machine candidates are never human-approved automatically.</p><main>{cards}</main>"""
    atomic_write_text(output_dir / "index.html", gallery)
    return {
        "candidate_rows": len(rows),
        "rendered_panels": rendered,
        "sheets": len(index_rows),
        "reused_sheets": reused_sheets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete PNG/CSV sheet pairs and rebuild only missing/corrupt sheets.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_review_sheets(
                args.master_csv,
                args.output_dir,
                resume=args.resume,
                progress_every=args.progress_every,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
