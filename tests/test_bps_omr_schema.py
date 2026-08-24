from bpsd_aligner.bps_omr_schema import musical_time_for_class


def test_musical_time_policy_covers_timeline_and_direction_families():
    for class_name in (
        "accidentalSharpSmall",
        "articStaccatoAbove",
        "beamSmall",
        "dynamicCrescendoLong",
        "fermataBelow",
        "fingering5",
        "noteheadBlackOnLineSmall",
        "restQuarter",
        "keyboardPedalPed",
        "ornamentTrill",
        "ottavaBracket",
        "slur",
        "stemSmall",
        "tie",
        "tremolo3",
        "tuplet5",
    ):
        assert musical_time_for_class(class_name) == 0

    for class_name in (
        "tempoAllegro",
        "termDolce",
        "clefG",
        "keyFlat",
        "timeSigCutCommon",
        "IlFine",
        "coda",
        "segno",
        "keyboardSulUnaCorda",
        "MarciaDaCapoAlFineSenzaRepetizione",
    ):
        assert musical_time_for_class(class_name) == 1


def test_musical_time_policy_leaves_context_dependent_or_unknown_classes_blank():
    assert musical_time_for_class("numeral3") is None
    assert musical_time_for_class("futureUnknownClass") is None
    assert musical_time_for_class("") is None
