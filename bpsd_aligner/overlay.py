"""Pure rendering helpers for alignment review overlays."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_alignment_overlay(
    image: Image.Image,
    rows: list[dict],
    mode: str,
    dynamic_classes: set[str] | frozenset[str],
) -> Image.Image:
    """Render an alignment overlay in memory."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font = _load_font(15 if mode == "dynamics" else 12)

    for row in rows:
        is_dynamic = row["class"] in dynamic_classes
        is_fingering = row["class"].startswith("fingering")
        if mode == "dynamics" and not is_dynamic:
            continue
        if mode == "fingerings" and not is_fingering:
            continue

        x = float(row["x"])
        y = float(row["y"])
        box_width = float(row["w"])
        box_height = float(row["h"])
        rectangle = (
            round((x - box_width / 2) * width),
            round((y - box_height / 2) * height),
            round((x + box_width / 2) * width),
            round((y + box_height / 2) * height),
        )

        status = row["status"]
        color = {
            "matched": "green",
            "inferred": "blue",
            "review": "darkorange",
            "unresolved": "gray",
        }.get(status, "red")
        draw.rectangle(rectangle, outline=color, width=2)

        if mode == "all":
            label = (
                f"Y{row['txt_line']} {row['class']} "
                f"{row['start_meas']}-{row['end_meas']} "
                f"{row['status']}"
            )
        elif mode == "class":
            start_written = row.get("start_xml_measure") or row.get("xml_measure", "")
            end_written = row.get("end_xml_measure") or start_written
            written_range = (
                start_written
                if str(start_written) == str(end_written)
                else f"{start_written}-{end_written}"
            )
            label = (
                f"Y{row['txt_line']} m{written_range} "
                f"t={row['start_meas']}-{row['end_meas']} {row['status']}"
            )
        elif is_dynamic:
            label = (
                f"{row['class']} m{row['xml_measure']} "
                f"t={row['start_meas']}"
            )
        else:
            if status == "unresolved":
                label = f"{row['class'][-1]} unresolved"
            else:
                label = (
                    f"{row['class'][-1]} n={row['start_note']} "
                    f"t={row['start_meas']} c={row['confidence']}"
                )

        if (
            row["target_x_px"] not in {"", "NA"}
            and row["target_y_px"] not in {"", "NA"}
        ):
            box_center = (round(x * width), round(y * height))
            start_center = (
                round(float(row["target_x_px"])),
                round(float(row["target_y_px"])),
            )
            end_x = row.get("end_target_x_px")
            end_y = row.get("end_target_y_px")
            if end_x in {None, "", "NA"} or end_y in {None, "", "NA"}:
                end_x, end_y = row["target_x_px"], row["target_y_px"]
            end_center = (
                round(float(end_x)),
                round(float(end_y)),
            )
            for target_center in dict.fromkeys((start_center, end_center)):
                draw.line((box_center, target_center), fill=color, width=1)
                radius = 3
                draw.ellipse(
                    (
                        target_center[0] - radius,
                        target_center[1] - radius,
                        target_center[0] + radius,
                        target_center[1] + radius,
                    ),
                    outline=color,
                    width=1,
                )
            if start_center != end_center:
                draw.line((start_center, end_center), fill=color, width=2)

        text_y = max(0, rectangle[1] - (18 if is_dynamic else 14))
        draw.text(
            (rectangle[0], text_y),
            label,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )

    return output


def write_alignment_overlay(
    image: Image.Image,
    rows: list[dict],
    output_path: Path,
    mode: str,
    dynamic_classes: set[str] | frozenset[str],
) -> None:
    """Render an alignment overlay to a file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_alignment_overlay(image, rows, mode, dynamic_classes).save(output_path)
