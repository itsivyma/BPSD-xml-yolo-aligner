from pathlib import Path

import fitz
import pytest
from PIL import Image

from bpsd_aligner.pdf_utils import pdf_page_count, render_pdf_page


def _write_pdf(path: Path) -> None:
    with fitz.open() as document:
        first = document.new_page(width=300, height=400)
        first.insert_text((40, 60), "page one")
        second = document.new_page(width=360, height=480)
        second.insert_text((40, 60), "page two")
        document.save(path)


def test_render_pdf_page_is_one_based_and_resumable(tmp_path):
    pdf_path = tmp_path / "clean.pdf"
    output_path = tmp_path / "pages" / "page-0002.png"
    _write_pdf(pdf_path)

    assert pdf_page_count(pdf_path) == 2
    assert render_pdf_page(pdf_path, 2, output_path, dpi=144) == output_path
    first_mtime = output_path.stat().st_mtime_ns
    with Image.open(output_path) as image:
        assert image.size == (720, 960)
        assert image.mode == "RGB"

    assert render_pdf_page(pdf_path, 2, output_path, dpi=144) == output_path
    assert output_path.stat().st_mtime_ns == first_mtime


def test_render_pdf_page_rejects_page_outside_pdf(tmp_path):
    pdf_path = tmp_path / "clean.pdf"
    _write_pdf(pdf_path)

    with pytest.raises(ValueError, match="has 2 pages"):
        render_pdf_page(pdf_path, 3, tmp_path / "page.png")
