from repeat_mapping import align_fingerprints


def test_align_fingerprints_maps_inserted_repeat_occurrence() -> None:
    written = ["m1", "m2", "m3", "m4", "m5"]
    unfolded = ["m1", "m2", "m3", "m2", "m3", "m4", "m5"]

    mapping, _evidence = align_fingerprints(written, unfolded)

    assert mapping == [0, 1, 2, 1, 2, 3, 4]


def test_align_fingerprints_fills_gap_between_exact_contiguous_anchors() -> None:
    written = ["m1", "m2", "m3", "m4"]
    unfolded = ["m1", "exported-m2", "exported-m3", "m4"]

    mapping, evidence = align_fingerprints(written, unfolded)

    assert mapping == [0, 1, 2, 3]
    assert evidence[-1]["method"] == "bounded_contiguous_interpolation"
