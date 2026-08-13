import json

import pytest

from bpsd_aligner.thresholds import auto_accept_threshold, configured_thresholds


def test_thresholds_have_class_families_and_support_valid_override(tmp_path, monkeypatch):
    assert auto_accept_threshold("fingering4", 0.5) == 0.95
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
