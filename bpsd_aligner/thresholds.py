"""Per-class auto-accept thresholds with an optional deployment override."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "fingering": 0.95,
    "slur": 0.82,
    "tie": 0.60,
    "articulation": 0.65,
    "fermata": 0.65,
    "tuplet": 0.60,
}


@lru_cache(maxsize=1)
def configured_thresholds() -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    configured = os.environ.get("BPSD_ALIGNER_THRESHOLDS", "").strip()
    if not configured:
        return thresholds
    payload = json.loads(Path(configured).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold configuration must be a JSON object")
    for name, value in payload.items():
        numeric = float(value)
        if not 0 <= numeric <= 1:
            raise ValueError(f"threshold for {name} must be between 0 and 1")
        thresholds[str(name)] = numeric
    return thresholds


def auto_accept_threshold(class_name: str, default: float) -> float:
    thresholds = configured_thresholds()
    if class_name in thresholds:
        return thresholds[class_name]
    if class_name.startswith("fingering"):
        return thresholds["fingering"]
    if class_name.startswith("tuplet"):
        return thresholds["tuplet"]
    if class_name.startswith("fermata"):
        return thresholds["fermata"]
    if class_name.startswith("artic"):
        return thresholds["articulation"]
    return default
