"""Generic one-to-one candidate pairing helpers.

The alignment engine supplies class-specific cost functions.  This module only
selects non-conflicting pairs and reports how unambiguous each selected pair
is; it intentionally contains no MusicXML or YOLO class policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


Candidate = dict[str, Any]
CostFunction = Callable[[Candidate, Candidate], float]


def greedy_pairs(
    boxes: list[Candidate],
    targets: list[Candidate],
    cost: CostFunction,
) -> list[tuple[Candidate, Candidate, float]]:
    """Select the lowest-cost non-conflicting box/target pairs."""

    scored = sorted(
        (cost(box, target), box_index, target_index)
        for box_index, box in enumerate(boxes)
        for target_index, target in enumerate(targets)
    )
    used_boxes: set[int] = set()
    used_targets: set[int] = set()
    output = []
    for value, box_index, target_index in scored:
        if box_index in used_boxes or target_index in used_targets:
            continue
        if value >= 1_000_000:
            continue
        used_boxes.add(box_index)
        used_targets.add(target_index)
        output.append((boxes[box_index], targets[target_index], value))
    return output


def mutual_geometry_pairs(
    boxes: list[Candidate],
    targets: list[Candidate],
    cost: CostFunction,
) -> list[tuple[Candidate, Candidate, float, float, bool]]:
    """Return one-to-one pairs with a mutual-best flag and candidate margin."""

    if not boxes or not targets:
        return []
    costs = [[float(cost(box, target)) for target in targets] for box in boxes]
    pairs = greedy_pairs(boxes, targets, cost)
    output = []
    for box, target, value in pairs:
        box_index = boxes.index(box)
        target_index = targets.index(target)
        box_costs = sorted(
            candidate for candidate in costs[box_index] if candidate < 1_000_000
        )
        target_costs = sorted(
            row[target_index]
            for row in costs
            if row[target_index] < 1_000_000
        )
        second_box = box_costs[1] if len(box_costs) > 1 else value + 1.0
        second_target = (
            target_costs[1] if len(target_costs) > 1 else value + 1.0
        )
        margin = max(0.0, min(second_box, second_target) - value)
        mutual = (
            target_index
            == min(range(len(targets)), key=lambda index: costs[box_index][index])
            and box_index
            == min(range(len(boxes)), key=lambda index: costs[index][target_index])
        )
        output.append((box, target, value, margin, mutual))
    return output
