from bps_xml_alignment import StaffGeometry, SystemGeometry
from slur_batch_candidates import (
    assign_box_system,
    classify_proposal,
    score_box_against_segment,
)


def _system(number: int, upper_center: float, lower_center: float):
    return SystemGeometry(
        number=number,
        upper=StaffGeometry(
            center=upper_center,
            line_spacing=10.0,
            lines=[
                upper_center - 20,
                upper_center - 10,
                upper_center,
                upper_center + 10,
                upper_center + 20,
            ],
        ),
        lower=StaffGeometry(
            center=lower_center,
            line_spacing=10.0,
            lines=[
                lower_center - 20,
                lower_center - 10,
                lower_center,
                lower_center + 10,
                lower_center + 20,
            ],
        ),
        x_left=0,
        x_right=1000,
    )


def test_assign_box_system_uses_nearest_vertical_region():
    systems = [_system(1, 100, 180), _system(2, 360, 440)]

    assert assign_box_system(70, systems) == 1
    assert assign_box_system(410, systems) == 2


def test_geometric_score_prefers_centered_box_with_matching_width():
    segment = {
        "x0": 100,
        "y0": 200,
        "x1": 160,
        "y1": 205,
        "staff_spacing": 10,
        "orientation": "over",
    }
    aligned = {
        "center_x": 130,
        "center_y": 180,
        "width": 58,
        "height": 8,
    }
    displaced = {
        "center_x": 300,
        "center_y": 250,
        "width": 12,
        "height": 8,
    }

    assert (
        score_box_against_segment(aligned, segment)["score"]
        > score_box_against_segment(displaced, segment)["score"]
    )


def test_confidence_classification_abstains_conservatively():
    assert classify_proposal(
        best_score=0.90,
        margin=0.20,
        mutual_best=True,
        segment_type="full",
    )[0] == "high_confidence_candidate"
    assert classify_proposal(
        best_score=0.90,
        margin=0.20,
        mutual_best=True,
        segment_type="start",
    )[0] == "needs_review"
    assert classify_proposal(
        best_score=0.30,
        margin=0.20,
        mutual_best=True,
        segment_type="full",
    )[0] == "possible_scan_only"
