import csv
import json
from pathlib import Path

from apply_human_audit_feedback import apply_human_audit_feedback


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_apply_feedback_preserves_source_and_resumes(tmp_path: Path):
    audit_path = tmp_path / "audit.csv"
    rows = [
        {
            "sample_id": "S001", "bbox_id": "score:Y1", "score_id": "score",
            "page_id": "score-01", "class": "fingering1", "start_meas": "1.0",
            "note_ids": "[1]", "pitches": '["C4"]',
            "xml_event_ids_json": '["score:E1"]', "proposal_status": "pending",
            "proposed_combined_status": "pending", "audit_decision": "",
            "audited_by": "", "audited_at": "", "audit_comment": "",
            "current_review_status": "needs_review", "human_approved": "false",
            "decision": "", "corrected_start_meas": "", "corrected_end_meas": "",
            "corrected_note_ids": "", "reviewer": "", "reviewed_at": "", "comment": "",
        },
        {
            "sample_id": "S002", "bbox_id": "score:Y2", "score_id": "score",
            "page_id": "score-01", "class": "fingering2", "start_meas": "2.0",
            "note_ids": "[2]", "pitches": '["D4"]',
            "xml_event_ids_json": '["score:E2"]', "proposal_status": "pending",
            "proposed_combined_status": "pending", "audit_decision": "",
            "audited_by": "", "audited_at": "", "audit_comment": "",
            "current_review_status": "needs_review", "human_approved": "false",
            "decision": "", "corrected_start_meas": "", "corrected_end_meas": "",
            "corrected_note_ids": "", "reviewer": "", "reviewed_at": "", "comment": "",
        },
    ]
    _write_csv(audit_path, rows)
    original = audit_path.read_bytes()
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "recorded_date": "2026-08-09", "source": "test", "entries": [
                    {
                        "sample_id": "S001", "bbox_id": "score:Y1",
                        "resolution_status": "resolved_unique_bps_note",
                        "user_correction": {"measure": 3, "staff": 2, "pitch": "E4"},
                        "corrected_start_meas": 3.25, "corrected_note_ids": [3],
                        "corrected_xml_event_ids": ["score:E3"], "evidence": "checked",
                    },
                    {
                        "sample_id": "S001", "bbox_id": "score:Y1",
                        "resolution_status": "withdrawn_by_user", "apply_to_audit": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    first = apply_human_audit_feedback(audit_path, feedback_path, output, resume=True)
    second = apply_human_audit_feedback(audit_path, feedback_path, output, resume=True)

    assert audit_path.read_bytes() == original
    assert first["passed"] is True
    assert first["applied_corrections"] == 1
    assert first["withdrawn_feedback_entries"] == 1
    assert first["resumed"] is False
    assert second["resumed"] is True
    with (output / "auto_accept_audit_reviewed.csv").open(newline="", encoding="utf-8") as file:
        reviewed = list(csv.DictReader(file))
    assert reviewed[0]["human_approved"] == "true"
    assert reviewed[0]["corrected_note_ids"] == "[3]"
    assert reviewed[0]["corrected_xml_event_ids_json"] == '["score:E3"]'
    assert reviewed[1]["human_approved"] == "false"
