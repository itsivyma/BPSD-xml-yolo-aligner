"""Small, page-oriented PDF helpers used by the web upload workflow."""

from __future__ import annotations

import math
import os
from pathlib import Path

import fitz
from PIL import Image


DEFAULT_MAX_PDF_PAGES = 500
DEFAULT_MAX_PDF_RENDER_PIXELS = 100_000_000


def _pdf_page_limit() -> int:
    limit = int(os.environ.get("BPSD_ALIGNER_MAX_PDF_PAGES", DEFAULT_MAX_PDF_PAGES))
    if limit < 1:
        raise ValueError("BPSD_ALIGNER_MAX_PDF_PAGES must be at least 1")
    return limit


def _validate_page_count(count: int) -> int:
    limit = _pdf_page_limit()
    if count > limit:
        raise ValueError(
            f"Clean repetition PDF has {count} pages; the configured limit is {limit}."
        )
    return count


def pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a readable PDF."""

    try:
        with fitz.open(pdf_path) as document:
            if not document.is_pdf:
                raise ValueError("The uploaded clean score is not a PDF")
            return _validate_page_count(document.page_count)
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
    if dpi < 72 or dpi > 600:
        raise ValueError("Clean repetition PDF render DPI must be between 72 and 600")
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
            _validate_page_count(document.page_count)
            if page_number > document.page_count:
                raise ValueError(
                    f"Clean repetition PDF has {document.page_count} pages, "
                    f"but MusicXML/scan page {page_number} was requested"
                )
            page = document.load_page(page_number - 1)
            scale = dpi / 72.0
            width = math.ceil(page.rect.width * scale)
            height = math.ceil(page.rect.height * scale)
            max_pixels = int(
                os.environ.get(
                    "BPSD_ALIGNER_MAX_PDF_RENDER_PIXELS",
                    DEFAULT_MAX_PDF_RENDER_PIXELS,
                )
            )
            if max_pixels < 1:
                raise ValueError(
                    "BPSD_ALIGNER_MAX_PDF_RENDER_PIXELS must be at least 1"
                )
            if width * height > max_pixels:
                raise ValueError(
                    "Clean repetition PDF page would render to "
                    f"{width * height:,} pixels; the configured limit is "
                    f"{max_pixels:,}."
                )
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_bytes(pixmap.tobytes("png"))
            os.chmod(temporary_path, 0o600)
            temporary_path.replace(output_path)
    except (fitz.FileDataError, fitz.EmptyFileError) as error:
        raise ValueError(f"Unable to read clean repetition PDF: {error}") from error
    return output_path
