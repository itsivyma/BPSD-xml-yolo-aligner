from bpsd_aligner.calibrate_thresholds import calibrate_groups
from bpsd_aligner.review_dataset import build_review_samples


def test_build_review_samples_joins_machine_evidence_and_human_labels():
    predictions = [
        {
            "page_id": "page-01",
            "txt_line": "1",
            "class_id": "56",
            "class": "slur",
            "status": "review",
            "match_source": "xml_span",
            "confidence": "0.81",
            "geometry_score": "0.9",
            "candidate_margin": "0.2",
            "xml_time_confirmed": "true",
            "start_meas": "1",
            "end_meas": "2",
            "start_note": "10",
            "end_note": "20",
            "connected_note": "[10, 20]",
            "xml_staff": "1",
        },
        {
            "page_id": "page-01",
            "txt_line": "2",
            "class_id": "45",
            "class": "fingeringSubstitution",
            "status": "review",
            "confidence": "0.7",
            "xml_time_confirmed": "false",
        },
    ]
    corrections = {
        "alignment_fingerprint": "abc",
        "score_id": "score-01",
        "pipeline_version": "1",
        "entries": [
            {
                "page_id": "page-01",
                "yolo_line": "1",
                "class": "slur",
                "action": "confirm",
                "original": {"start_meas": "1", "end_meas": "2"},
                "corrected": {"start_meas": "1", "end_meas": "2"},
            },
            {
                "page_id": "page-01",
                "yolo_line": "2",
                "class": "fingeringSubstitution",
                "action": "wrong_class",
                "corrected_class_id": "4",
                "corrected_class": "fingering4",
                "corrected": {},
            },
        ],
    }

    rows, errors = build_review_samples(predictions, corrections)

    assert errors == []
    assert rows[0]["threshold_family"] == "slur"
    assert rows[0]["alignment_correct"] == "1"
    assert rows[0]["calibration_eligible"] == "1"
    assert rows[0]["review_full_image"] == ""
    assert rows[0]["review_crop_image"] == ""
    assert rows[1]["threshold_family"] == "fingering"
    assert rows[1]["calibration_eligible"] == "0"
    assert rows[1]["requires_human_or_model"] == "1"


def test_calibration_recommends_only_reviewed_family_thresholds():
    rows = [
        {
            "calibration_eligible": "1",
            "threshold_family": "slur",
            "class": "slur",
            "confidence": score,
            "alignment_correct": correct,
            "current_threshold": "0.85",
        }
        for score, correct in (
            ("0.90", "1"),
            ("0.80", "1"),
            ("0.75", "1"),
            ("0.60", "0"),
        )
    ]

    results, recommendations, errors = calibrate_groups(
        rows,
        target_precision=1.0,
        min_reviewed_per_group=4,
        min_accepted=2,
        min_reviewed_overall=4,
    )

    assert errors == []
    assert recommendations == {"slur": 0.75}
    family = next(row for row in results if row["scope"] == "family")
    assert family["recommended_precision"] == "1.000000"
    assert family["recommended_recall"] == "1.000000"
    assert family["ready"] == "1"


def test_calibration_refuses_small_samples():
    rows = [
        {
            "calibration_eligible": "1",
            "threshold_family": "tie",
            "class": "tie",
            "confidence": "0.9",
            "alignment_correct": "1",
            "current_threshold": "0.8",
        }
    ]

    results, recommendations, errors = calibrate_groups(rows)

    assert errors == []
    assert recommendations == {}
    assert all(row["ready"] == "0" for row in results)
