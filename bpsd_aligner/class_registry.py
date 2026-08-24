"""Single source of truth for stable YOLO class-family policies.

Keep only rules that are independent of page geometry and MusicXML contents
here.  Matching strategies remain in the alignment engine because their
answer depends on the actual score evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_THRESHOLDS = {
    "fingering": 0.95,
    "dynamic": 0.85,
    "slur": 0.85,
    "tie": 0.80,
    "articulation": 0.80,
    "fermata": 0.80,
    "ornament": 0.80,
    "tuplet": 0.80,
    "notehead": 0.92,
    "accidental": 0.92,
    "stem": 0.92,
    "beam": 0.90,
    "flag": 0.92,
    "ottava": 0.90,
    "pedal": 0.90,
    "tempo": 0.90,
    "term": 0.90,
}

FAMILY_PREFIXES = (
    ("fingering", "fingering"),
    ("dynamic", "dynamic"),
    ("tuplet", "tuplet"),
    ("fermata", "fermata"),
    ("artic", "articulation"),
    ("ornament", "ornament"),
    ("notehead", "notehead"),
    ("accidental", "accidental"),
    ("stem", "stem"),
    ("beam", "beam"),
    ("flag", "flag"),
    ("ottava", "ottava"),
    ("keyboardPed", "pedal"),
    ("tempo", "tempo"),
    ("term", "term"),
)

# BPS-OMR musical_time=1 means the symbol is outside the musical timeline.
OUTSIDE_TIMELINE_CLASSES = frozenset(
    {
        "IlFine",
        "LangsamUndSehnsuchtvoll",
        "MarciaDaCapoAlFineSenzaRepetizione",
        "coda",
        "segno",
        "keyboardMitEinerSaite",
        "keyboardSulUnaCorda",
    }
)
OUTSIDE_TIMELINE_PREFIXES = ("tempo", "term", "clef", "key", "timeSig")

# musical_time=0 means the glyph attaches to a musical instant or duration.
IN_TIMELINE_CLASSES = frozenset(
    {
        "arpeggiato",
        "caesura",
        "keyboardPed",
        "keyboardPedalPed",
        "keyboardPedalUp",
        "ottavaBracket",
        "slur",
        "tie",
    }
)
IN_TIMELINE_PREFIXES = (
    "accidental",
    "artic",
    "beam",
    "dynamic",
    "fermata",
    "fingering",
    "flag",
    "notehead",
    "ornament",
    "rest",
    "stem",
    "tremolo",
    "tuplet",
)


@dataclass(frozen=True)
class ClassPolicy:
    """Stable policy values known without opening a score page."""

    class_name: str
    family: str
    musical_time: int | None
    default_threshold: float | None


def threshold_family(class_name: object) -> str:
    """Return the deployment/calibration family for one YOLO class."""

    name = str(class_name or "").strip()
    if name in {"slur", "tie"}:
        return name
    for prefix, family in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return name or "unknown"


def musical_time_for_class(class_name: object) -> int | None:
    """Return the BPS-OMR timeline flag, or ``None`` for ambiguous roles."""

    name = str(class_name or "").strip()
    if not name:
        return None
    if name in OUTSIDE_TIMELINE_CLASSES:
        return 1
    if name in IN_TIMELINE_CLASSES:
        return 0
    if name.startswith(OUTSIDE_TIMELINE_PREFIXES):
        return 1
    if name.startswith(IN_TIMELINE_PREFIXES):
        return 0
    return None


def class_policy(class_name: object) -> ClassPolicy:
    """Return one compact, inspectable policy used by exports and review."""

    name = str(class_name or "").strip()
    family = threshold_family(name)
    return ClassPolicy(
        class_name=name,
        family=family,
        musical_time=musical_time_for_class(name),
        default_threshold=DEFAULT_THRESHOLDS.get(family),
    )
