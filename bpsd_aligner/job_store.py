"""Persistent, fingerprinted storage for resumable website jobs."""

from __future__ import annotations

import json
import io
import os
import shutil
import zipfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pipeline_checkpoint import atomic_write_json


DEFAULT_MAX_FILES = 500
DEFAULT_MAX_FILE_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_BATCH_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_CONCURRENT_JOBS = 2
DEFAULT_MAX_PAGES = 200


@dataclass(frozen=True)
class JobLease:
    job_lock: Path
    worker_lock: Path


def job_store_root() -> Path:
    """Return a configurable persistent job root.

    Production deployments should mount this directory on persistent storage.
    The default intentionally stays outside the repository and uploaded data.
    """

    configured = os.environ.get("BPSD_ALIGNER_JOB_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("TMPDIR", "/tmp")) / "bpsd-aligner-jobs"


def job_directory(fingerprint: str, *, root: Path | None = None) -> Path:
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint.lower()
    ):
        raise ValueError("job fingerprint must be a SHA-256 hexadecimal digest")
    target = (root or job_store_root()) / fingerprint.lower()
    target.mkdir(parents=True, exist_ok=True)
    return target


def validate_upload_batch(
    uploads: Iterable,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> dict[str, int]:
    """Validate count, per-file size, and total bytes without reading payloads."""

    present = [uploaded for uploaded in uploads if uploaded is not None]
    if len(present) > max_files:
        raise ValueError(
            f"Upload contains {len(present)} files; the limit is {max_files}."
        )
    oversized = [
        str(getattr(uploaded, "name", "unnamed"))
        for uploaded in present
        if int(getattr(uploaded, "size", 0)) > max_file_bytes
    ]
    if oversized:
        raise ValueError(
            "Files exceed the per-file upload limit: " + ", ".join(oversized)
        )
    total = sum(int(getattr(uploaded, "size", 0)) for uploaded in present)
    if total > max_batch_bytes:
        raise ValueError(
            f"Upload batch is {total / (1024 ** 2):.1f} MB; "
            f"the total limit is {max_batch_bytes / (1024 ** 2):.0f} MB."
        )
    return {"files": len(present), "bytes": total}


def validate_page_count(page_count: int, *, max_pages: int | None = None) -> int:
    """Reject accidentally huge browser jobs before any alignment work starts."""

    limit = max_pages or int(
        os.environ.get("BPSD_ALIGNER_MAX_PAGES", DEFAULT_MAX_PAGES)
    )
    if limit < 1:
        raise ValueError("BPSD_ALIGNER_MAX_PAGES must be at least 1")
    if page_count > limit:
        raise ValueError(
            f"This job has {page_count} score pages; the configured limit is {limit}. "
            "Split the upload or increase BPSD_ALIGNER_MAX_PAGES."
        )
    return page_count


def write_job_manifest(
    job_dir: Path, *, fingerprint: str, pipeline_version: str, inputs: list[dict]
) -> Path:
    path = job_dir / "job_manifest.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "job_id": fingerprint,
            "pipeline_version": pipeline_version,
            "inputs": inputs,
        },
    )
    os.utime(job_dir, None)
    return path


def write_job_status(
    job_dir: Path,
    *,
    state: str,
    stage: str,
    completed_pages: int = 0,
    total_pages: int = 0,
    message: str = "",
    error: str = "",
) -> Path:
    """Atomically persist user-readable job progress across browser restarts."""

    if state not in {"queued", "running", "completed", "failed", "cancelled"}:
        raise ValueError(f"unsupported job state: {state}")
    path = job_dir / "job_status.json"
    created_at = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        try:
            created_at = json.loads(path.read_text(encoding="utf-8")).get(
                "created_at", created_at
            )
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "job_id": job_dir.name,
            "state": state,
            "stage": stage,
            "completed_pages": int(completed_pages),
            "total_pages": int(total_pages),
            "message": message,
            "error": error,
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    os.utime(job_dir, None)
    return path


def publish_completed_job(
    job_dir: Path,
    *,
    result: dict,
    completed_pages: int,
    total_pages: int,
    message: str = "Background alignment outputs are ready.",
) -> Path:
    """Publish a durable result before exposing the completed state.

    The Streamlit status fragment may read ``job_status.json`` immediately
    after any atomic status update.  Writing the result first prevents a
    completed job from briefly pointing at a missing ``job_result.json``.
    """

    result_path = job_dir / "job_result.json"
    atomic_write_json(result_path, result)
    write_job_status(
        job_dir,
        state="completed",
        stage="outputs_ready",
        completed_pages=completed_pages,
        total_pages=total_pages,
        message=message,
    )
    return result_path


def request_job_cancellation(job_dir: Path) -> Path:
    """Persist a cooperative cancellation request for a background worker."""

    path = job_dir / "cancel_requested.json"
    atomic_write_json(
        path,
        {"requested_at": datetime.now(timezone.utc).isoformat()},
    )
    return path


def job_cancellation_requested(job_dir: Path) -> bool:
    return (job_dir / "cancel_requested.json").is_file()


def write_page_checkpoint(
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    page_id: str,
    page_number: int,
    report: dict,
    page_image: Path,
) -> Path:
    path = job_dir / "checkpoints" / f"{page_id}.json"
    portable_report = dict(report)
    portable_report["outputs"] = dict(report.get("outputs", {}))
    output_relpaths = {}
    for name, value in portable_report["outputs"].items():
        try:
            output_relpaths[name] = str(Path(value).resolve().relative_to(job_dir.resolve()))
        except ValueError:
            output_relpaths[name] = ""
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "fingerprint": fingerprint,
            "pipeline_version": pipeline_version,
            "page_id": page_id,
            "page_number": page_number,
            "page_image": str(page_image),
            "page_image_relpath": str(page_image.resolve().relative_to(job_dir.resolve())),
            "output_relpaths": output_relpaths,
            "report": portable_report,
        },
    )
    os.utime(job_dir, None)
    return path


def prune_job_store(
    *,
    root: Path | None = None,
    retention_hours: float,
    now: float | None = None,
) -> list[str]:
    """Remove inactive fingerprinted jobs older than an explicit retention limit."""

    if retention_hours <= 0:
        raise ValueError("job retention hours must be greater than zero")
    store = root or job_store_root()
    if not store.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - retention_hours * 60 * 60
    lease_dir = store / ".leases"
    removed = []
    for candidate in sorted(store.iterdir()):
        name = candidate.name.lower()
        if (
            not candidate.is_dir()
            or len(name) != 64
            or any(character not in "0123456789abcdef" for character in name)
        ):
            continue
        if candidate.stat().st_mtime >= cutoff:
            continue
        if (lease_dir / f"job-{name}.lock").exists():
            continue
        trash = store / f".deleting-{name}-{os.getpid()}"
        try:
            candidate.rename(trash)
        except FileNotFoundError:
            continue
        shutil.rmtree(trash)
        removed.append(name)
    return removed


def load_page_checkpoint(
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    page_id: str,
    page_number: int,
) -> dict | None:
    """Load a valid page checkpoint; stale or incomplete entries are ignored."""

    path = job_dir / "checkpoints" / f"{page_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload["report"]
        if payload.get("output_relpaths"):
            report["outputs"] = {
                name: str(job_dir / relative)
                for name, relative in payload["output_relpaths"].items()
                if relative
            }
        output_paths = [Path(value) for value in report["outputs"].values()]
        page_image = (
            job_dir / payload["page_image_relpath"]
            if payload.get("page_image_relpath")
            else Path(payload["page_image"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        payload.get("fingerprint") != fingerprint
        or payload.get("pipeline_version") != pipeline_version
        or payload.get("page_id") != page_id
        or int(payload.get("page_number", -1)) != page_number
        or not page_image.is_file()
        or not output_paths
        or not all(path.is_file() for path in output_paths)
    ):
        return None
    payload["page_image"] = str(page_image)
    payload["report"] = report
    return payload


def build_job_checkpoint_archive(job_dir: Path, destination: Path) -> Path:
    """Package resumable derived artifacts without original uploaded inputs."""

    allowed_roots = [job_dir / "checkpoints", job_dir / "outputs"]
    manifest = job_dir / "job_manifest.json"
    if not manifest.is_file():
        raise ValueError("job manifest is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest, arcname="job_manifest.json")
        status = job_dir / "job_status.json"
        if status.is_file():
            archive.write(status, arcname="job_status.json")
        for root in allowed_roots:
            if not root.is_dir():
                continue
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                # The final download archive is reproducible from the individual
                # outputs and otherwise duplicates nearly the entire checkpoint.
                if path.name in {"all_outputs.zip", "alignment_job_checkpoint.zip"}:
                    continue
                archive.write(path, arcname=str(path.relative_to(job_dir)))
    return destination


def restore_job_checkpoint_archive(
    data: bytes,
    job_dir: Path,
    *,
    expected_fingerprint: str,
    expected_pipeline_version: str | None = None,
    max_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024,
) -> int:
    """Safely restore a portable checkpoint archive into its fingerprinted job."""

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        try:
            manifest = json.loads(archive.read("job_manifest.json"))
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("checkpoint ZIP has no valid job_manifest.json") from error
        if manifest.get("job_id") != expected_fingerprint:
            raise ValueError("checkpoint ZIP belongs to different uploaded inputs")
        if (
            expected_pipeline_version is not None
            and manifest.get("pipeline_version") != expected_pipeline_version
        ):
            raise ValueError(
                "checkpoint ZIP was created by a different pipeline version"
            )
        members = [member for member in archive.infolist() if not member.is_dir()]
        total = sum(member.file_size for member in members)
        if total > max_uncompressed_bytes:
            raise ValueError("checkpoint ZIP expands beyond the allowed size")
        restored = 0
        for member in members:
            relative = Path(member.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.parts[0]
                not in {
                    "job_manifest.json",
                    "job_status.json",
                    "checkpoints",
                    "outputs",
                }
            ):
                raise ValueError(f"unsafe checkpoint ZIP path: {member.filename}")
            destination = job_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            restored += 1
    return restored


def acquire_job_lease(
    job_dir: Path,
    *,
    max_concurrent: int | None = None,
    stale_after_seconds: int = 6 * 60 * 60,
) -> JobLease:
    """Acquire one per-job lock and one bounded global worker slot."""

    limit = max_concurrent or int(
        os.environ.get("BPSD_ALIGNER_MAX_CONCURRENT_JOBS", DEFAULT_MAX_CONCURRENT_JOBS)
    )
    if limit < 1:
        raise ValueError("BPSD_ALIGNER_MAX_CONCURRENT_JOBS must be at least 1")
    lease_dir = job_dir.parent / ".leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()

    def lock_is_live(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
            age = now - path.stat().st_mtime
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            try:
                return now - path.stat().st_mtime <= stale_after_seconds
            except OSError:
                return False
        if age <= stale_after_seconds:
            return True
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    for path in lease_dir.glob("slot-*.lock"):
        try:
            if not lock_is_live(path):
                path.unlink()
        except FileNotFoundError:
            pass
    payload = json.dumps({"pid": os.getpid(), "job": job_dir.name})

    job_lock = lease_dir / f"job-{job_dir.name}.lock"
    try:
        descriptor = os.open(job_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if lock_is_live(job_lock):
            raise RuntimeError("This exact alignment job is already running.")
        job_lock.unlink(missing_ok=True)
        descriptor = os.open(job_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(payload)

    for slot in range(limit):
        path = lease_dir / f"slot-{slot}.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(payload)
        return JobLease(job_lock=job_lock, worker_lock=path)
    job_lock.unlink(missing_ok=True)
    raise RuntimeError(
        f"All {limit} alignment worker slots are busy. Try again after a current job finishes."
    )


def release_job_lease(lease: JobLease | Path | None) -> None:
    if isinstance(lease, JobLease):
        lease.worker_lock.unlink(missing_ok=True)
        lease.job_lock.unlink(missing_ok=True)
    elif lease is not None:
        lease.unlink(missing_ok=True)
