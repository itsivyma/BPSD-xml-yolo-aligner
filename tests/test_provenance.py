from bpsd_aligner.provenance import PACKAGE_ALIGNMENT_MODULES


def test_span_semantics_changes_invalidate_alignment_artifacts() -> None:
    assert "span_semantics.py" in PACKAGE_ALIGNMENT_MODULES
