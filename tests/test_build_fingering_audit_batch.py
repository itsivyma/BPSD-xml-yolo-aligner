from build_fingering_audit_batch import select_audit_rows


def _row(index: int, status: str, confidence: str, page: str, cls: str) -> dict:
    return {
        "bbox_id": f"{page}:Y{index}", "page_id": page, "score_id": "score",
        "class": cls, "alignment_status": status, "confidence": confidence,
        "movement_scope_status": "in_bpsd_scope", "error_code": "",
    }


def test_select_audit_rows_balances_reasons_and_excludes_reviewed():
    rows = [
        _row(1, "unresolved", "", "p1", "fingering1"),
        _row(2, "unresolved", "", "p2", "fingering2"),
        _row(3, "candidate", "0.99", "p1", "fingering1"),
        _row(4, "candidate", "0.98", "p2", "fingering2"),
        _row(5, "candidate", "0.97", "p1", "fingering1"),
    ]

    selected = select_audit_rows(
        rows, {"p1:Y3"}, unresolved_count=2, risk_count=2
    )

    assert len(selected) == 4
    assert [row["sample_id"] for row in selected] == ["R001", "R002", "R003", "R004"]
    assert {row["selection_reason"] for row in selected} == {
        "unresolved", "high_risk_link"
    }
    assert "p1:Y3" not in {row["bbox_id"] for row in selected}
