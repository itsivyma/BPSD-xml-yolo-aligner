import json
import os
import zipfile
from pathlib import Path

import pytest

import bpsd_aligner.job_store as job_store

from bpsd_aligner.job_store import (
    acquire_job_lease,
    build_job_checkpoint_archive,
    job_directory,
    load_page_checkpoint,
    publish_completed_job,
    prune_job_store,
    restore_job_checkpoint_archive,
    release_job_lease,
    validate_upload_batch,
    validate_page_count,
    write_job_status,
    write_page_checkpoint,
    write_job_manifest,
)


def test_validate_page_count_rejects_oversized_job():
    assert validate_page_count(2, max_pages=2) == 2
    with pytest.raises(ValueError, match="3 score pages"):
        validate_page_count(3, max_pages=2)


class Upload:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size


def test_upload_batch_limits_count_and_total_bytes():
    assert validate_upload_batch(
        [Upload("a", 4), Upload("b", 5)],
        max_files=2,
        max_file_bytes=8,
        max_batch_bytes=9,
    ) == {"files": 2, "bytes": 9}
    with pytest.raises(ValueError, match="total limit"):
        validate_upload_batch(
            [Upload("a", 5), Upload("b", 5)], max_batch_bytes=9
        )
    with pytest.raises(ValueError, match="per-file"):
        validate_upload_batch([Upload("large", 10)], max_file_bytes=9)


def test_page_checkpoint_requires_matching_inputs_and_existing_outputs(tmp_path: Path):
    fingerprint = "a" * 64
    job_dir = job_directory(fingerprint, root=tmp_path)
    output = job_dir / "output.csv"
    output.write_text("a\n1\n", encoding="utf-8")
    image = job_dir / "page.png"
    image.write_bytes(b"png")
    report = {"outputs": {"csv": str(output)}}
    checkpoint = write_page_checkpoint(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version="v1",
        page_id="page-1",
        page_number=1,
        report=report,
        page_image=image,
    )
    assert json.loads(checkpoint.read_text())["page_id"] == "page-1"
    assert load_page_checkpoint(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version="v1",
        page_id="page-1",
        page_number=1,
    ) is not None
    assert load_page_checkpoint(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version="v2",
        page_id="page-1",
        page_number=1,
    ) is None
    output.unlink()
    assert load_page_checkpoint(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version="v1",
        page_id="page-1",
        page_number=1,
    ) is None


def test_portable_checkpoint_archive_validates_fingerprint_and_restores(tmp_path: Path):
    fingerprint = "b" * 64
    source = job_directory(fingerprint, root=tmp_path / "source")
    write_job_manifest(
        source,
        fingerprint=fingerprint,
        pipeline_version="v1",
        inputs=[],
    )
    write_job_status(
        source,
        state="running",
        stage="aligning",
        completed_pages=1,
        total_pages=2,
    )
    artifact = source / "outputs" / "pages" / "page.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("a\n1\n")
    (source / "outputs" / "all_outputs.zip").write_bytes(b"duplicate")
    archive = build_job_checkpoint_archive(source, tmp_path / "checkpoint.zip")
    with zipfile.ZipFile(archive) as packaged:
        assert "outputs/all_outputs.zip" not in packaged.namelist()
    target = job_directory(fingerprint, root=tmp_path / "target")
    restored = restore_job_checkpoint_archive(
        archive.read_bytes(),
        target,
        expected_fingerprint=fingerprint,
        expected_pipeline_version="v1",
    )
    assert restored == 3
    assert (target / "outputs" / "pages" / "page.csv").is_file()
    assert json.loads((target / "job_status.json").read_text())["stage"] == "aligning"
    with pytest.raises(ValueError, match="different uploaded inputs"):
        restore_job_checkpoint_archive(
            archive.read_bytes(), target, expected_fingerprint="c" * 64
        )
    with pytest.raises(ValueError, match="different pipeline version"):
        restore_job_checkpoint_archive(
            archive.read_bytes(),
            target,
            expected_fingerprint=fingerprint,
            expected_pipeline_version="v2",
        )


def test_prune_job_store_removes_only_old_inactive_jobs(tmp_path: Path):
    old = job_directory("1" * 64, root=tmp_path)
    current = job_directory("2" * 64, root=tmp_path)
    active = job_directory("3" * 64, root=tmp_path)
    old_time = 1_000.0
    os.utime(old, (old_time, old_time))
    os.utime(active, (old_time, old_time))
    lease_dir = tmp_path / ".leases"
    lease_dir.mkdir()
    (lease_dir / f"job-{active.name}.lock").write_text("active")

    removed = prune_job_store(
        root=tmp_path, retention_hours=1, now=old_time + 7_200
    )

    assert removed == [old.name]
    assert not old.exists()
    assert current.exists()
    assert active.exists()


def test_job_status_preserves_creation_time_and_validates_state(tmp_path: Path):
    job = job_directory("4" * 64, root=tmp_path)
    path = write_job_status(job, state="queued", stage="upload")
    created = json.loads(path.read_text())["created_at"]
    write_job_status(
        job,
        state="running",
        stage="alignment",
        completed_pages=2,
        total_pages=5,
    )
    updated = json.loads(path.read_text())
    assert updated["created_at"] == created
    assert updated["completed_pages"] == 2
    with pytest.raises(ValueError, match="unsupported job state"):
        write_job_status(job, state="unknown", stage="bad")


def test_completed_status_is_published_only_after_result_exists(
    tmp_path: Path, monkeypatch
):
    job = job_directory("5" * 64, root=tmp_path)
    original_write_status = job_store.write_job_status

    def checked_write_status(job_dir, **kwargs):
        if kwargs.get("state") == "completed":
            assert (job_dir / "job_result.json").is_file()
        return original_write_status(job_dir, **kwargs)

    monkeypatch.setattr(job_store, "write_job_status", checked_write_status)
    result_path = publish_completed_job(
        job,
        result={"schema_version": "1.0", "fingerprint": job.name},
        completed_pages=2,
        total_pages=2,
    )

    assert json.loads(result_path.read_text())["fingerprint"] == job.name
    status = json.loads((job / "job_status.json").read_text())
    assert status["state"] == "completed"
    assert status["stage"] == "outputs_ready"


def test_job_lease_limits_concurrent_workers(tmp_path: Path):
    job = job_directory("d" * 64, root=tmp_path)
    lease = acquire_job_lease(job, max_concurrent=1)
    other_job = job_directory("f" * 64, root=tmp_path)
    with pytest.raises(RuntimeError, match="worker slots are busy"):
        acquire_job_lease(other_job, max_concurrent=1)
    release_job_lease(lease)
    second = acquire_job_lease(job, max_concurrent=1)
    release_job_lease(second)


def test_job_lease_prevents_same_job_using_a_second_slot(tmp_path: Path):
    job = job_directory("e" * 64, root=tmp_path)
    lease = acquire_job_lease(job, max_concurrent=2)
    with pytest.raises(RuntimeError, match="exact alignment job"):
        acquire_job_lease(job, max_concurrent=2)
    release_job_lease(lease)
