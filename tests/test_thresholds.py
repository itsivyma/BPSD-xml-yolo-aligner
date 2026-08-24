import json

import pytest

from bpsd_aligner.thresholds import (
    auto_accept_threshold,
    configured_thresholds,
    threshold_family,
)


def test_thresholds_have_class_families_and_support_valid_override(tmp_path, monkeypatch):
    assert auto_accept_threshold("fingering4", 0.5) == 0.95
    assert auto_accept_threshold("dynamicF", 0.5) == 0.85
    assert auto_accept_threshold("tie", 0.5) == 0.80
    assert auto_accept_threshold("articStaccatoAbove", 0.5) == 0.80
    assert auto_accept_threshold("fermataBelow", 0.5) == 0.80
    assert auto_accept_threshold("ornamentTrill", 0.5) == 0.80
    assert auto_accept_threshold("tuplet3", 0.5) == 0.80
    assert auto_accept_threshold("noteheadBlackOnLineSmall", 0.5) == 0.92
    assert auto_accept_threshold("accidentalSharpSmall", 0.5) == 0.92
    assert auto_accept_threshold("stemSmall", 0.5) == 0.92
    assert auto_accept_threshold("beamSmall", 0.5) == 0.90
    assert auto_accept_threshold("flag8thUpSmall", 0.5) == 0.92
    assert auto_accept_threshold("ottavaBracket", 0.5) == 0.90
    assert auto_accept_threshold("keyboardPedalUp", 0.5) == 0.90
    assert auto_accept_threshold("tempoAllegro", 0.5) == 0.90
    assert auto_accept_threshold("termDolce", 0.5) == 0.90
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"fingering": 0.9, "slur": 0.88}))
    monkeypatch.setenv("BPSD_ALIGNER_THRESHOLDS", str(path))
    configured_thresholds.cache_clear()
    assert auto_accept_threshold("fingering4", 0.5) == 0.9
    assert auto_accept_threshold("slur", 0.5) == 0.88
    configured_thresholds.cache_clear()


def test_threshold_override_rejects_out_of_range_values(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"slur": 2}))
    monkeypatch.setenv("BPSD_ALIGNER_THRESHOLDS", str(path))
    configured_thresholds.cache_clear()
    with pytest.raises(ValueError, match="between 0 and 1"):
        configured_thresholds()
    configured_thresholds.cache_clear()


def test_threshold_family_is_stable_for_calibration_and_deployment():
    assert threshold_family("fingeringSubstitution") == "fingering"
    assert threshold_family("dynamicCrescendoHairpin") == "dynamic"
    assert threshold_family("articStaccatoAbove") == "articulation"
    assert threshold_family("keyboardPedalUp") == "pedal"
    assert threshold_family("slur") == "slur"
    assert threshold_family("unknownThing") == "unknownThing"
