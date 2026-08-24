"""BPS-OMR bounding-box semantics shared by CLI and website exports."""

from __future__ import annotations

from bpsd_aligner.class_registry import (
    IN_TIMELINE_CLASSES,
    IN_TIMELINE_PREFIXES,
    OUTSIDE_TIMELINE_CLASSES,
    OUTSIDE_TIMELINE_PREFIXES,
    musical_time_for_class,
)

__all__ = [
    "IN_TIMELINE_CLASSES",
    "IN_TIMELINE_PREFIXES",
    "OUTSIDE_TIMELINE_CLASSES",
    "OUTSIDE_TIMELINE_PREFIXES",
    "musical_time_for_class",
]
