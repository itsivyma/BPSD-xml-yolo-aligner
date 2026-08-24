"""Score-page staff, system, and barline geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class StaffGeometry:
    center: float
    line_spacing: float
    lines: list[float]


@dataclass
class SystemGeometry:
    number: int
    upper: StaffGeometry
    lower: StaffGeometry
    x_left: float
    x_right: float

    @property
    def center(self) -> float:
        return (self.upper.center + self.lower.center) / 2



def _consecutive_groups(values: Iterable[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _staff_pattern_candidates(
    row_ink: np.ndarray,
    crop_width: int,
) -> list[list[float]]:
    """Find non-overlapping five-line patterns in a horizontal ink profile.

    Dense chords can make several neighboring rows look like long horizontal
    lines, while noteheads can interrupt a genuine staff line.  Scoring a
    complete, approximately equally spaced five-line pattern is therefore more
    stable than thresholding each row independently.
    """

    candidates = []
    height = len(row_ink)
    # Scale the minimum spacing with page height: the rendered reference score
    # is smaller than the scan, while seven-pixel repeated strokes in the
    # larger Op90 scan come from dense chord beams rather than staff lines.
    minimum_spacing = max(7, round(height * 0.0043))
    for spacing_twice in range(2 * minimum_spacing, 49):
        spacing = spacing_twice / 2
        final_offset = round(4 * spacing)
        for top in range(2, height - final_offset - 2):
            lines = []
            strengths = []
            for line_index in range(5):
                predicted = round(top + line_index * spacing)
                search = range(predicted - 2, predicted + 3)
                row = max(search, key=lambda y: row_ink[y])
                lines.append(float(row))
                strengths.append(float(row_ink[row]) / crop_width)

            average_strength = sum(strengths) / 5
            minimum_strength = min(strengths)
            if average_strength < 0.38 or minimum_strength < 0.25:
                continue

            # Prefer strong complete patterns, then patterns whose selected
            # rows stay closest to the proposed equal spacing.
            spacing_error = sum(
                abs((lines[index] - lines[index - 1]) - spacing)
                for index in range(1, 5)
            ) / 4
            score = average_strength + 0.15 * minimum_strength - 0.02 * spacing_error
            candidates.append((score, spacing, lines))

    selected = []
    for score, spacing, lines in sorted(candidates, reverse=True):
        top = min(lines)
        bottom = max(lines)
        overlaps = any(
            not (
                bottom < min(existing) - spacing
                or top > max(existing) + spacing
            )
            for existing in selected
        )
        if not overlaps:
            selected.append(lines)

    return sorted(selected, key=lambda lines: sum(lines) / len(lines))


def detect_systems(image: Image.Image) -> list[SystemGeometry]:
    """Detect paired piano systems from five-line horizontal patterns."""

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    x0 = round(width * 0.05)
    x1 = round(width * 0.95)
    crop_width = x1 - x0

    row_ink = (gray[:, x0:x1] < 170).sum(axis=1)
    raw_staff_groups = _staff_pattern_candidates(row_ink, crop_width)
    strict_staff_groups = [
        group
        for group in raw_staff_groups
        if min(
            _longest_true_run(gray[round(line), x0:x1] < 190)
            for line in group
        )
        >= crop_width * 0.20
    ]
    # Most pages are best served by the long-line filter because it rejects
    # beams and footer text.  A scan can, however, contain a genuine staff line
    # broken by damage or dense notation.  If strict filtering leaves an
    # impossible odd count, fall back only when the complete five-line pattern
    # set itself forms valid piano pairs.  This preserves the strict result on
    # normal pages while recovering damaged staves such as Op110 p6 and Op111
    # p1.
    if len(strict_staff_groups) >= 2 and len(strict_staff_groups) % 2 == 0:
        staff_groups = strict_staff_groups
    elif len(raw_staff_groups) >= 2 and len(raw_staff_groups) % 2 == 0:
        staff_groups = raw_staff_groups
    else:
        staff_groups = strict_staff_groups
    if len(staff_groups) % 2 != 0 or len(staff_groups) < 2:
        raise ValueError(
            "Could not detect paired five-line staves "
            f"(strict={len(strict_staff_groups)}, raw={len(raw_staff_groups)})"
        )

    systems = []
    for system_index in range(0, len(staff_groups), 2):
        upper_lines = staff_groups[system_index]
        lower_lines = staff_groups[system_index + 1]
        all_lines = upper_lines + lower_lines

        votes = np.zeros(width, dtype=int)
        for center in all_lines:
            row = round(center)
            neighborhood = gray[max(0, row - 1) : min(height, row + 2), :]
            votes += (neighborhood < 190).any(axis=0)

        # A genuine staff can be interrupted by dense chords, accidentals,
        # damage, or compression. Requiring six of ten sampled staff rows made
        # the left half of Op90 p2 disappear and shifted every XML notehead to
        # the right. Three agreeing rows over a sustained horizontal run still
        # reject isolated stems while preserving partially obscured staves.
        candidate_columns = votes >= 3
        smoothed = np.convolve(
            candidate_columns.astype(int),
            np.ones(31, dtype=int),
            mode="same",
        )
        long_staff_columns = np.where(smoothed >= 20)[0]
        if not len(long_staff_columns):
            raise ValueError(f"Could not detect x range for system {system_index // 2 + 1}")

        upper_spacing = median(
            b - a for a, b in zip(upper_lines, upper_lines[1:])
        )
        lower_spacing = median(
            b - a for a, b in zip(lower_lines, lower_lines[1:])
        )

        systems.append(
            SystemGeometry(
                number=system_index // 2 + 1,
                upper=StaffGeometry(
                    center=sum(upper_lines) / 5,
                    line_spacing=float(upper_spacing),
                    lines=upper_lines,
                ),
                lower=StaffGeometry(
                    center=sum(lower_lines) / 5,
                    line_spacing=float(lower_spacing),
                    lines=lower_lines,
                ),
                x_left=float(long_staff_columns[0]),
                x_right=float(long_staff_columns[-1]),
            )
        )

    return systems


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def detect_barlines(
    image: Image.Image,
    systems: list[SystemGeometry],
    expected_boundary_counts: list[int],
) -> list[list[int]]:
    """Detect piano-system barlines using continuous vertical ink.

    Note stems and dense chords can have a high total ink count, but unlike a
    barline they do not form one continuous line from the upper staff to the
    lower staff.  System-edge estimates from horizontal staff lines are
    refined by searching for the actual vertical start/end barline nearby.
    """

    if len(systems) != len(expected_boundary_counts):
        raise ValueError(
            "One expected boundary count is required for each system"
        )

    gray = np.asarray(image.convert("L"))
    output = []
    for system, expected_count in zip(systems, expected_boundary_counts):
        y0 = round(system.upper.lines[0])
        y1 = round(system.lower.lines[-1])
        x0 = round(system.x_left)
        x1 = round(system.x_right)
        band_height = y1 - y0 + 1

        longest_runs = []
        for x in range(x0, x1 + 1):
            dark = gray[y0 : y1 + 1, x] < 170
            longest_runs.append(_longest_true_run(dark))

        candidate_columns = [
            x0 + index
            for index, run_length in enumerate(longest_runs)
            if run_length >= band_height * 0.85
        ]
        groups = _consecutive_groups(candidate_columns)
        boundaries = [
            round((group[0] + group[-1]) / 2)
            for group in groups
        ]

        edge_window = max(20, round((x1 - x0) * 0.04))
        refined_edges = []
        for estimated_edge in (x0, x1):
            search_left = max(0, estimated_edge - edge_window)
            search_right = min(
                gray.shape[1] - 1,
                estimated_edge + edge_window,
            )
            scored = []
            for x in range(search_left, search_right + 1):
                dark = gray[y0 : y1 + 1, x] < 170
                longest_run = _longest_true_run(dark)
                ink_count = int(dark.sum())
                score = (
                    2 * longest_run
                    + ink_count
                    - 0.4 * abs(x - estimated_edge)
                )
                scored.append((score, x))
            refined_edges.append(max(scored)[1])

        left_edge, right_edge = refined_edges
        edge_tolerance = 8
        interior_boundaries = [
            boundary
            for boundary in boundaries
            if boundary > left_edge + edge_tolerance
            and boundary < right_edge - edge_tolerance
        ]
        boundaries = [
            left_edge,
            *interior_boundaries,
            right_edge,
        ]

        if len(boundaries) != expected_count:
            raise ValueError(
                f"System {system.number}: expected {expected_count} "
                f"measure boundaries, detected {len(boundaries)} "
                f"({boundaries})"
            )
        output.append(boundaries)

    return output


def align_barlines_from_reference(
    image: Image.Image,
    systems: list[SystemGeometry],
    reference_systems: list[SystemGeometry],
    reference_boundaries: list[list[int]],
) -> list[list[dict]]:
    """Find scan barlines near normalized boundaries from a clean score.

    The reference supplies only a search position.  The selected x coordinate
    still comes from vertical ink in the target scan.  Low-continuity lines are
    retained but explicitly marked for visual review because printed symbols
    can occlude an otherwise genuine barline.
    """

    if not (
        len(systems)
        == len(reference_systems)
        == len(reference_boundaries)
    ):
        raise ValueError("Target and reference must have the same system count")

    gray = np.asarray(image.convert("L"))
    aligned = []
    for system, reference_system, boundaries in zip(
        systems,
        reference_systems,
        reference_boundaries,
    ):
        predicted = [
            system.x_left
            + (
                (boundary - reference_system.x_left)
                / (reference_system.x_right - reference_system.x_left)
            )
            * (system.x_right - system.x_left)
            for boundary in boundaries
        ]

        y0 = round(system.upper.lines[0])
        y1 = round(system.lower.lines[-1])
        band_height = y1 - y0 + 1
        system_output = []
        for index, predicted_x in enumerate(predicted):
            if index in {0, len(predicted) - 1}:
                edge_window = max(
                    20,
                    round((system.x_right - system.x_left) * 0.04),
                )
                search_left = max(0, round(predicted_x) - edge_window)
                search_right = min(
                    gray.shape[1] - 1,
                    round(predicted_x) + edge_window,
                )
            else:
                search_left = round((predicted[index - 1] + predicted_x) / 2)
                search_right = round((predicted_x + predicted[index + 1]) / 2)

            scored = []
            for x in range(search_left, search_right + 1):
                dark = gray[y0 : y1 + 1, x] < 170
                longest_run = _longest_true_run(dark)
                ink_count = int(dark.sum())
                score = (
                    2 * longest_run
                    + ink_count
                    - 0.4 * abs(x - predicted_x)
                )
                scored.append((score, x, longest_run, ink_count))

            _score, x, longest_run, ink_count = max(scored)
            vertical_coverage = longest_run / band_height
            ink_coverage = ink_count / band_height
            status = (
                "system_edge"
                if index in {0, len(predicted) - 1}
                else (
                    "detected"
                    if vertical_coverage >= 0.85
                    else "review_occluded"
                )
            )
            system_output.append(
                {
                    "x": x,
                    "predicted_x": predicted_x,
                    "vertical_coverage": vertical_coverage,
                    "ink_coverage": ink_coverage,
                    "status": status,
                }
            )
        aligned.append(system_output)

    return aligned


def assign_system(box: dict, systems: list[SystemGeometry], image_height: int) -> int:
    y_pixel = box["y"] * image_height
    return min(systems, key=lambda system: abs(system.center - y_pixel)).number
