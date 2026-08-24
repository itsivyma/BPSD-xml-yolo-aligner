import csv
import json
from pathlib import Path

import pytest

from bpsd_aligner.regression_smoke import (
    _compare_baseline,
    _semantic_fingerprints,
    load_regression_manifest,
    run_regression_smoke,
)


def _write_manifest(tmp_path: Path) -> Path:
    for name in ("notes.json", "page.png", "page.txt", "score.xml", "score-unfolded.xml", "notes.csv"):
        (tmp_path / name).write_text("test", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "name": "test suite",
        "dataset_root": ".",
        "paths": {
            "notes_json": "notes.json",
            "image": "page.png",
            "yolo": "page.txt",
            "xml_repetitions": "score.xml",
            "xml_unfolded": "score-unfolded.xml",
            "bps_notes": "notes.csv",
        },
        "pages": [
            {
                "page_id": "page-01",
                "score_id": "score-01",
                "page_number": 1,
                "tags": ["slur"],
                "required_classes": ["slur"],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_regression_manifest_resolves_paths_relative_to_dataset_root(tmp_path):
    manifest = _write_manifest(tmp_path)

    payload, pages = load_regression_manifest(manifest)

    assert payload["name"] == "test suite"
    assert pages[0]["image"] == (tmp_path / "page.png").resolve()
    assert pages[0]["required_classes"] == ["slur"]


def test_regression_manifest_rejects_wrong_class_map_profile(tmp_path):
    manifest = _write_manifest(tmp_path)
    (tmp_path / "notes.json").write_text(
        json.dumps({"categories": []}), encoding="utf-8"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset_profile"] = "test-profile"
    payload["expected_class_count"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 1 classes"):
        load_regression_manifest(manifest)


def test_regression_runner_checkpoints_and_resumes(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "output"
    calls = []

    monkeypatch.setattr(
        "bpsd_aligner.regression_smoke.prepare_score_sources",
        lambda **_kwargs: {"prepared": True},
    )

    def fake_alignment(**kwargs):
        calls.append(kwargs["page_number"])
        page_output = Path(kwargs["output_dir"])
        detailed = page_output / "detailed.csv"
        detailed.parent.mkdir(parents=True, exist_ok=True)
        with detailed.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["class", "status", "match_source"],
            )
            writer.writeheader()
            writer.writerow(
                {"class": "slur", "status": "matched", "match_source": "xml"}
            )
        validation = page_output / "validation.json"
        validation.write_text("{}", encoding="utf-8")
        return {
            "passed": True,
            "validation_errors": [],
            "final_rows_with_blank_time": 0,
            "outputs": {
                "detailed_csv": str(detailed),
                "validation_json": str(validation),
            },
        }

    monkeypatch.setattr(
        "bpsd_aligner.regression_smoke.run_uploaded_alignment", fake_alignment
    )

    first = run_regression_smoke(
        manifest_path=manifest,
        output_dir=output,
        update_baseline=True,
    )
    second = run_regression_smoke(
        manifest_path=manifest,
        output_dir=output,
        resume=True,
    )

    assert first["passed"] is True
    assert second["passed"] is True
    assert second["resumed_pages"] == 1
    assert calls == [1]
    assert (output / "baseline.json").is_file()
    assert (output / "regression_summary.csv").is_file()
    baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["pages"]["page-01"]["required_class_status_counts"] == (
        '{"slur": {"matched": 1}}'
    )
    assert "class_status_counts" not in baseline["pages"]["page-01"]


def test_page_fingerprint_changes_when_pipeline_code_changes(tmp_path, monkeypatch):
    from bpsd_aligner import regression_smoke

    manifest = _write_manifest(tmp_path)
    _, pages = load_regression_manifest(manifest)
    monkeypatch.setattr(
        regression_smoke,
        "alignment_runtime_identity",
        lambda: {"pipeline_code_signature": "a", "thresholds": {}},
    )
    first = regression_smoke._page_fingerprint(pages[0])
    monkeypatch.setattr(
        regression_smoke,
        "alignment_runtime_identity",
        lambda: {"pipeline_code_signature": "b", "thresholds": {}},
    )

    assert regression_smoke._page_fingerprint(pages[0]) != first


def test_page_fingerprint_changes_when_thresholds_change(tmp_path, monkeypatch):
    from bpsd_aligner import regression_smoke

    manifest = _write_manifest(tmp_path)
    _, pages = load_regression_manifest(manifest)
    monkeypatch.setattr(
        regression_smoke,
        "alignment_runtime_identity",
        lambda: {"pipeline_code_signature": "same", "thresholds": {"slur": 0.8}},
    )
    first = regression_smoke._page_fingerprint(pages[0])
    monkeypatch.setattr(
        regression_smoke,
        "alignment_runtime_identity",
        lambda: {"pipeline_code_signature": "same", "thresholds": {"slur": 0.9}},
    )

    assert regression_smoke._page_fingerprint(pages[0]) != first


def test_semantic_fingerprint_detects_endpoint_change_with_same_row_counts():
    original = [
        {
            "txt_line": "1",
            "class_id": "56",
            "class": "slur",
            "status": "matched",
            "start_meas": "52.000",
            "end_meas": "53.000",
            "start_note": "100",
            "end_note": "110",
        }
    ]
    changed = [{**original[0], "end_note": "111"}]

    original_digest, original_classes = _semantic_fingerprints(original)
    changed_digest, changed_classes = _semantic_fingerprints(changed)

    assert changed_digest != original_digest
    assert changed_classes != original_classes
    current = {
        "page_id": "page-1",
        **{field: "1" for field in (
            "alignment_rows",
            "matched_rows",
            "inferred_rows",
            "review_rows",
            "blank_time_rows",
            "cross_page_candidates",
            "required_class_status_counts",
        )},
        "semantic_digest": changed_digest,
        "semantic_class_digests": changed_classes,
    }
    expected = {
        key: value
        for key, value in current.items()
        if key != "page_id"
    }
    expected["semantic_digest"] = original_digest
    expected["semantic_class_digests"] = original_classes
    differences = _compare_baseline(
        [current], {"pages": {"page-1": expected}}
    )

    assert any("semantic_digest changed" in item for item in differences)
    assert any("classes slur" in item for item in differences)


def test_regression_runner_scores_optional_human_ground_truth(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "output"
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "page_id,yolo_line,class,expected_start_meas,expected_end_meas,"
        "expected_start_note,expected_end_note,expected_connected_note,expected_staff\n"
        "page-01,1,slur,1.000,2.000,10,20,\"[10, 20]\",1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "bpsd_aligner.regression_smoke.prepare_score_sources",
        lambda **_kwargs: {"prepared": True},
    )

    def fake_alignment(**kwargs):
        page_output = Path(kwargs["output_dir"])
        detailed = page_output / "detailed.csv"
        detailed.parent.mkdir(parents=True, exist_ok=True)
        with detailed.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "txt_line", "class", "status", "match_source", "start_meas",
                    "end_meas", "start_note", "end_note", "connected_note", "xml_staff",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "txt_line": "1", "class": "slur", "status": "matched",
                    "match_source": "xml", "start_meas": "1.000", "end_meas": "2.000",
                    "start_note": "10", "end_note": "20", "connected_note": "[10, 20]",
                    "xml_staff": "1",
                }
            )
        validation = page_output / "validation.json"
        validation.write_text("{}", encoding="utf-8")
        return {
            "passed": True,
            "validation_errors": [],
            "final_rows_with_blank_time": 0,
            "outputs": {"detailed_csv": str(detailed), "validation_json": str(validation)},
        }

    monkeypatch.setattr(
        "bpsd_aligner.regression_smoke.run_uploaded_alignment", fake_alignment
    )

    report = run_regression_smoke(
        manifest_path=manifest,
        output_dir=output,
        update_baseline=True,
        ground_truth_path=truth,
    )

    assert report["passed"] is True
    assert report["ground_truth"]["overall"]["exact_accuracy"] == 1.0
    assert (output / "ground_truth_accuracy.json").is_file()
