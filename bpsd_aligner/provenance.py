"""Stable code provenance used to invalidate derived alignment artifacts."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


ROOT_ALIGNMENT_MODULES = (
    "bps_xml_alignment.py",
    "combine_yolo_xml.py",
    "dataset_dry_run.py",
    "pipeline_checkpoint.py",
    "repeat_mapping.py",
    "xml_export.py",
)
PACKAGE_ALIGNMENT_MODULES = (
    "bps_omr_schema.py",
    "candidate_scoring.py",
    "class_registry.py",
    "csv_io.py",
    "geometry.py",
    "musicxml.py",
    "pdf_utils.py",
    "review_candidates.py",
    "schema.py",
    "span_semantics.py",
    "thresholds.py",
    "web_pipeline.py",
)


@lru_cache(maxsize=1)
def pipeline_code_signature() -> str:
    """Hash executable alignment sources, independent of package version bumps."""

    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent
    source_paths = [
        package_dir / name
        for name in PACKAGE_ALIGNMENT_MODULES
        if (package_dir / name).is_file()
    ]
    source_paths.extend(
        project_dir / name
        for name in ROOT_ALIGNMENT_MODULES
        if (project_dir / name).is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(source_paths), key=lambda value: str(value)):
        relative = path.relative_to(project_dir)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def alignment_runtime_identity() -> dict:
    """Return every non-input setting that can change alignment semantics."""

    from bpsd_aligner.thresholds import configured_thresholds

    return {
        "pipeline_code_signature": pipeline_code_signature(),
        "thresholds": configured_thresholds(),
    }
