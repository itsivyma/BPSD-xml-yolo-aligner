import csv
from pathlib import Path

from PIL import Image

from pipeline_checkpoint import valid_png
from render_audit_contact_sheets import render_audit_sheets


def test_render_audit_sheets_groups_duplicate_sources_and_resumes(tmp_path: Path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (240, 180), "white").save(first_image)
    Image.new("RGB", (240, 180), "lightblue").save(second_image)
    audit_csv = tmp_path / "audit.csv"
    fields = ["sample_id", "bbox_id", "class", "confidence", "review_png"]
    with audit_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "S001", "bbox_id": "page:Y1", "class": "fingering1", "confidence": "0.95", "review_png": str(first_image)},
                {"sample_id": "S002", "bbox_id": "page:Y2", "class": "fingering2", "confidence": "0.96", "review_png": str(first_image)},
                {"sample_id": "S003", "bbox_id": "page:Y3", "class": "fingering3", "confidence": "0.97", "review_png": str(second_image)},
            ]
        )

    output = tmp_path / "output"
    first = render_audit_sheets(audit_csv, output, resume=True)
    second = render_audit_sheets(audit_csv, output, resume=True)

    assert first["passed"] is True
    assert first["sample_rows"] == 3
    assert first["unique_source_images"] == 2
    assert first["contact_pages"] == 1
    assert first["resumed"] is False
    assert second["resumed"] is True
    assert valid_png(output / "audit_sample_overview.png")
    assert valid_png(output / "contact_pages" / "audit_contact_part01.png")
