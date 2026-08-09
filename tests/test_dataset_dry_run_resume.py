import csv
import json
from pathlib import Path

from dataset_dry_run import (
    FIELDS,
    PIPELINE_VERSION,
    _load_resumable_page,
    bootstrap_existing_checkpoints,
    select_manifest_pages,
)


def test_select_manifest_pages_preserves_order_and_rejects_unknown():
    import pytest

    pages = [{"page_id": "score-01"}, {"page_id": "score-02"}]

    assert select_manifest_pages(pages, {"score-02"}) == [pages[1]]
    assert select_manifest_pages(pages, None) == pages
    with pytest.raises(ValueError, match="missing-01"):
        select_manifest_pages(pages, {"missing-01"})


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_resumable_page_requires_matching_checkpoint_and_rows(
    tmp_path: Path,
) -> None:
    page_id = "score-01"
    page_csv = tmp_path / "pages" / f"{page_id}.csv"
    page_csv.parent.mkdir()
    row = {field: "" for field in FIELDS}
    row.update({"page_id": page_id, "bbox_id": f"{page_id}:Y1"})
    with page_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    checkpoint = tmp_path / "checkpoints" / f"{page_id}.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps(
            {
                "fingerprint": "current",
                "status": "completed",
                "row_count": 1,
                "validation_errors": [],
            }
        ),
        encoding="utf-8",
    )

    resumed = _load_resumable_page(
        checkpoint_path=checkpoint,
        page_csv_path=page_csv,
        page_id=page_id,
        expected_rows=1,
        fingerprint="current",
    )
    assert resumed is not None
    assert resumed[1]["resumed"] is True

    stale = _load_resumable_page(
        checkpoint_path=checkpoint,
        page_csv_path=page_csv,
        page_id=page_id,
        expected_rows=1,
        fingerprint="changed",
    )
    assert stale is None


def test_bootstrap_existing_checkpoints_does_not_run_alignment(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = {}
    for name in ["image.png", "labels.txt", "score.xml", "bps.csv", "notes.json"]:
        path = inputs / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    repeat_dir = tmp_path / "repeat"
    repeat_dir.mkdir()
    (repeat_dir / "score_repeat_mapping.csv").write_text("x\n", encoding="utf-8")
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()

    page = {
        "page_id": "score-01",
        "score_id": "score",
        "yolo_rows": "1",
        "image_path": str(paths["image.png"]),
        "yolo_path": str(paths["labels.txt"]),
        "xml_path": str(paths["score.xml"]),
        "bps_notes_path": str(paths["bps.csv"]),
        "image_sha256": "image-hash",
        "yolo_sha256": "yolo-hash",
    }
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [page])
    scope = tmp_path / "scope.csv"
    _write_csv(scope, [{"page_id": "score-01", "scan_system_index": "1"}])

    output = tmp_path / "output"
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "page_id": "score-01",
            "bbox_id": "score-01:Y1",
            "pipeline_version": PIPELINE_VERSION,
            "image_sha256": "image-hash",
            "yolo_sha256": "yolo-hash",
        }
    )
    _write_csv(output / "pages" / "score-01.csv", [row])
    _write_csv(output / "Xia_BPSD_alignment_master.csv", [row])
    (output / "validation_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "validation_errors": [],
                "pipeline_version": PIPELINE_VERSION,
                "canonical_rows": 1,
                "run_id": "old-run",
                "page_runs": [
                    {"page_id": "score-01", "status": "completed", "error": ""}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = bootstrap_existing_checkpoints(
        manifest_path=manifest,
        scope_path=scope,
        notes_json_path=paths["notes.json"],
        repeat_mapping_dir=repeat_dir,
        review_dir=review_dir,
        output_dir=output,
    )

    assert result["checkpoints_created"] == 1
    assert result["alignment_pages_executed"] == 0
    assert (output / "checkpoints" / "score-01.json").exists()
