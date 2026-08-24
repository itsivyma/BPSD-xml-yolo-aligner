from bpsd_aligner.class_registry import class_policy
from bpsd_aligner.bps_omr_schema import musical_time_for_class
from bpsd_aligner.thresholds import auto_accept_threshold, threshold_family


def test_registry_centralizes_timeline_and_threshold_policy():
    slur = class_policy("slur")
    assert slur.family == "slur"
    assert slur.musical_time == 0
    assert slur.default_threshold == 0.85

    tempo = class_policy("tempoRitardando")
    assert tempo.family == "tempo"
    assert tempo.musical_time == 1
    assert tempo.default_threshold == 0.90


def test_public_compatibility_modules_use_registry_values():
    assert threshold_family("fingering4") == "fingering"
    assert auto_accept_threshold("fingering4", 0.1) == 0.95
    assert musical_time_for_class("fingering4") == 0
    assert musical_time_for_class("numeral3") is None


def test_unknown_class_remains_unforced():
    policy = class_policy("newFutureSymbol")
    assert policy.family == "newFutureSymbol"
    assert policy.musical_time is None
    assert policy.default_threshold is None
