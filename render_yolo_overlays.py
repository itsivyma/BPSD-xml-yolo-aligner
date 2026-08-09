"""Render every YOLO bounding box back onto its source score image."""

from __future__ import annotations

import argparse
import html
from colorsys import hsv_to_rgb
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bps_xml_alignment import load_categories, load_yolo
from pipeline_checkpoint import (
    atomic_save_png,
    atomic_write_csv,
    atomic_write_text,
    emit_progress,
    valid_png,
)


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _color(class_id: int) -> tuple[int, int, int]:
    # The golden-ratio step keeps neighboring class IDs visually distinct.
    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = hsv_to_rgb(hue, 0.82, 0.72)
    return round(red * 255), round(green * 255), round(blue * 255)


def render_overlay(
    image_path: Path,
    yolo_path: Path,
    categories: dict[int, str],
    output_path: Path,
    *,
    show_class: bool,
) -> int:
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    boxes = load_yolo(yolo_path, categories=categories)
    label_font = _font(13 if show_class else 15)

    for box in boxes:
        x0 = round((box["x"] - box["w"] / 2) * width)
        y0 = round((box["y"] - box["h"] / 2) * height)
        x1 = round((box["x"] + box["w"] / 2) * width)
        y1 = round((box["y"] + box["h"] / 2) * height)
        color = _color(int(box["class_id"]))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)

        label = f"Y{box['txt_line']}"
        if show_class:
            label += f" {box['class']}"
        left = max(0, min(x0, width - 1))
        text_box = draw.textbbox((left, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = y0 - text_height - 5
        if top < 0:
            top = min(height - text_height - 4, y1 + 2)
        right = min(width, left + text_width + 5)
        draw.rectangle((left, top, right, top + text_height + 4), fill="white")
        draw.text((left + 2, top + 1), label, fill=color, font=label_font)

    atomic_save_png(image, output_path)
    image.close()
    return len(boxes)


def _write_gallery(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    cards = []
    for row in rows:
        page = html.escape(row["page_id"])
        original = Path(row["image_path"]).as_uri()
        id_only = html.escape(row["id_only_png"])
        id_class = html.escape(row["id_class_png"])
        cards.append(
            f"""<article><h2>{page} ({row['yolo_boxes']} boxes)</h2>
<p><a href="{original}">original</a> · <a href="{id_only}">Y IDs</a> · <a href="{id_class}">Y IDs + classes</a></p>
<a href="{id_class}"><img loading="lazy" src="{id_class}" alt="{page} overlay"></a></article>"""
        )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO score overlays</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f4f4;color:#222}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
article{background:white;padding:14px;border-radius:8px;box-shadow:0 1px 5px #bbb}
h1{margin-bottom:4px}h2{font-size:16px;margin:0 0 6px}p{margin:6px 0 10px}
img{width:100%;height:auto;border:1px solid #ddd}a{color:#0758b8}
</style></head><body><h1>YOLO score overlays</h1>
<p>Each colored box is an original YOLO annotation. Y numbers are 1-based TXT line numbers.</p>
<main>""" + "\n".join(cards) + "</main></body></html>\n"
    gallery_path = output_dir / "index.html"
    atomic_write_text(gallery_path, document)
    return gallery_path


def _current_overlay_pair(
    id_only_path: Path,
    id_class_path: Path,
    input_paths: tuple[Path, ...],
) -> bool:
    if not valid_png(id_only_path) or not valid_png(id_class_path):
        return False
    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    return min(
        id_only_path.stat().st_mtime_ns,
        id_class_path.stat().st_mtime_ns,
    ) >= newest_input


def render_dataset(
    xia_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    progress_every: int = 10,
) -> list[dict[str, str]]:
    categories = load_categories(xia_dir / "notes.json")
    image_dir = xia_dir / "images"
    label_dir = xia_dir / "labels"
    rows: list[dict[str, str]] = []
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    emit_progress(
        "yolo-overlays",
        0,
        len(images),
        f"starting (resume={resume})",
    )
    for index, image_path in enumerate(images, start=1):
        yolo_path = label_dir / f"{image_path.stem}.txt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"Missing YOLO label: {yolo_path}")
        id_only_rel = Path("id_only") / f"{image_path.stem}.png"
        id_class_rel = Path("id_class") / f"{image_path.stem}.png"
        id_only_path = output_dir / id_only_rel
        id_class_path = output_dir / id_class_rel
        reused = resume and _current_overlay_pair(
            id_only_path,
            id_class_path,
            (image_path, yolo_path, xia_dir / "notes.json"),
        )
        if reused:
            count = len(load_yolo(yolo_path, categories=categories))
        else:
            count = render_overlay(
                image_path,
                yolo_path,
                categories,
                id_only_path,
                show_class=False,
            )
            render_overlay(
                image_path,
                yolo_path,
                categories,
                id_class_path,
                show_class=True,
            )
        rows.append(
            {
                "page_id": image_path.stem,
                "image_path": str(image_path),
                "yolo_path": str(yolo_path),
                "yolo_boxes": str(count),
                "id_only_png": id_only_rel.as_posix(),
                "id_class_png": id_class_rel.as_posix(),
            }
        )
        if index == len(images) or index % max(1, progress_every) == 0:
            action = "reused" if reused else "rendered"
            emit_progress(
                "yolo-overlays",
                index,
                len(images),
                f"{image_path.stem} {action}",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_fields = [
        "page_id",
        "image_path",
        "yolo_path",
        "yolo_boxes",
        "id_only_png",
        "id_class_png",
    ]
    atomic_write_csv(output_dir / "overlay_index.csv", index_fields, rows)
    _write_gallery(output_dir, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xia-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete overlay PNG pairs and render only missing/corrupt pages.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    rows = render_dataset(
        args.xia_dir,
        args.output_dir,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    print(f"Rendered {len(rows)} pages to {args.output_dir}")


if __name__ == "__main__":
    main()
