from bpsd_aligner.candidate_scoring import greedy_pairs, mutual_geometry_pairs


def _distance(box: dict, target: dict) -> float:
    return abs(box["x"] - target["x"])


def test_greedy_pairs_selects_non_conflicting_lowest_cost_pairs():
    boxes = [{"id": "b1", "x": 0.0}, {"id": "b2", "x": 10.0}]
    targets = [{"id": "t1", "x": 1.0}, {"id": "t2", "x": 9.0}]

    pairs = greedy_pairs(boxes, targets, _distance)

    assert [(box["id"], target["id"], value) for box, target, value in pairs] == [
        ("b1", "t1", 1.0),
        ("b2", "t2", 1.0),
    ]


def test_greedy_pairs_skips_forbidden_costs():
    assert greedy_pairs([{"x": 0.0}], [{"x": 1.0}], lambda _box, _target: 1_000_000) == []


def test_mutual_geometry_pairs_reports_margin_and_mutual_best():
    boxes = [{"id": "b1", "x": 0.0}, {"id": "b2", "x": 10.0}]
    targets = [{"id": "t1", "x": 1.0}, {"id": "t2", "x": 8.0}]

    pairs = mutual_geometry_pairs(boxes, targets, _distance)

    assert [
        (box["id"], target["id"], value, margin, mutual)
        for box, target, value, margin, mutual in pairs
    ] == [
        ("b1", "t1", 1.0, 7.0, True),
        ("b2", "t2", 2.0, 6.0, True),
    ]


def test_mutual_geometry_pairs_handles_empty_inputs():
    assert mutual_geometry_pairs([], [{"x": 1.0}], _distance) == []
    assert mutual_geometry_pairs([{"x": 1.0}], [], _distance) == []
