"""Small, page-oriented PDF helpers used by the web upload workflow."""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image


def pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a readable PDF."""

    try:
        with fitz.open(pdf_path) as document:
            if not document.is_pdf:
                raise ValueError("The uploaded clean score is not a PDF")
            return document.page_count
    except (fitz.FileDataError, fitz.EmptyFileError) as error:
        raise ValueError(f"Unable to read clean repetition PDF: {error}") from error


def render_pdf_page(
    pdf_path: Path,
    page_number: int,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Render one one-based PDF page, reusing a valid checkpoint if present."""

    if page_number < 1:
        raise ValueError("Clean repetition PDF page numbers start at 1")
    if output_path.is_file():
        try:
            with Image.open(output_path) as image:
                image.verify()
            return output_path
        except (OSError, ValueError):
            output_path.unlink(missing_ok=True)

    try:
        with fitz.open(pdf_path) as document:
            if not document.is_pdf:
                raise ValueError("The uploaded clean score is not a PDF")
            if page_number > document.page_count:
                raise ValueError(
                    f"Clean repetition PDF has {document.page_count} pages, "
                    f"but MusicXML/scan page {page_number} was requested"
                )
            page = document.load_page(page_number - 1)
            scale = dpi / 72.0
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_bytes(pixmap.tobytes("png"))
            temporary_path.replace(output_path)
    except (fitz.FileDataError, fitz.EmptyFileError) as error:
        raise ValueError(f"Unable to read clean repetition PDF: {error}") from error
    return output_path
