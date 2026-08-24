import json
from pathlib import Path

import pytest

from bpsd_aligner.review_evidence import validate_review_evidence


def test_archived_review_evidence_is_valid_and_withdrawals_are_not_truth():
    directory = Path(__file__).parents[1] / "regression" / "review_evidence"

    report = validate_review_evidence(directory)

    assert report == {
        "status": "valid",
        "file_count": 3,
        "resolved_entry_count": 26,
        "withdrawn_identity_count": 1,
        "active_row_decision_count": 24,
        "batch_spot_check_count": 1,
        "batch_rows_promoted_to_human_approved": 0,
    }


def test_withdrawal_must_be_explicitly_excluded(tmp_path: Path):
    (tmp_path / "invalid.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "resolved_human_review",
                "entries": [
                    {
                        "sample_id": "S1",
                        "bbox_id": "Y1",
                        "audit_decision": "withdrawn_typo",
                        "resolution_status": "withdrawn_by_user",
                        "apply_to_audit": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="apply_to_audit=false"):
        validate_review_evidence(tmp_path)


def test_batch_spot_check_cannot_become_per_row_approval(tmp_path: Path):
    (tmp_path / "invalid.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "spot_check_batch_passed",
                "verification_level": "sampled_not_individually_verified",
                "per_row_human_approved": True,
                "preserve_automatic_alignment": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot approve every row"):
        validate_review_evidence(tmp_path)
