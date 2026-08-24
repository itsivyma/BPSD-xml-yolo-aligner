"""Persistent, fingerprinted storage for resumable website jobs."""

from __future__ import annotations

import json
import io
import os
import hashlib
import hmac
import secrets
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
DEFAULT_MAX_CHECKPOINT_MEMBERS = 10_000
DEFAULT_MAX_CHECKPOINT_MEMBER_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 200
CHECKPOINT_ARCHIVE_SCHEMA = "2.0"


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
    store = root or job_store_root()
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(store, 0o700)
    target = store / fingerprint.lower()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    return target


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _checkpoint_secret(job_dir: Path) -> bytes:
    """Return the deployment checkpoint signing secret.

    Set ``BPSD_ALIGNER_CHECKPOINT_SECRET`` to the same high-entropy value on
    every instance that must exchange checkpoint ZIPs. Local installations get
    a private persistent key in the job-store root.
    """

    configured = os.environ.get("BPSD_ALIGNER_CHECKPOINT_SECRET", "").strip()
    if configured:
        if len(configured.encode("utf-8")) < 32:
            raise ValueError(
                "BPSD_ALIGNER_CHECKPOINT_SECRET must contain at least 32 bytes"
            )
        return configured.encode("utf-8")
    key_path = job_dir.parent / ".checkpoint-secret"
    _private_directory(job_dir.parent)
    if not key_path.exists():
        try:
            descriptor = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as file:
                file.write(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)
    secret = key_path.read_bytes()
    if len(secret) < 32:
        raise ValueError("checkpoint signing key is invalid")
    return secret


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_signature(payload: dict, secret: bytes) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret, encoded, hashlib.sha256).hexdigest()


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
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    code_signature: str = "",
    inputs: list[dict],
    owner_id: str = "",
) -> Path:
    path = job_dir / "job_manifest.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "job_id": fingerprint,
            "owner_id": owner_id,
            "pipeline_version": pipeline_version,
            "code_signature": code_signature,
            "inputs": inputs,
        },
    )
    os.utime(job_dir, None)
    return path


def _validated_job_manifest(
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    owner_id: str,
    code_signature: str = "",
) -> dict:
    """Load the immutable identity fields used to protect persisted review data."""

    path = job_dir / "job_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("job manifest is missing or invalid") from error
    if manifest.get("job_id") != fingerprint or job_dir.name != fingerprint:
        raise ValueError("review state belongs to different uploaded inputs")
    if manifest.get("pipeline_version") != pipeline_version:
        raise ValueError("review state belongs to a different pipeline version")
    if code_signature and manifest.get("code_signature") != code_signature:
        raise ValueError("review state belongs to different alignment code")
    if str(manifest.get("owner_id", "")) != str(owner_id):
        raise ValueError("review state belongs to a different authenticated user")
    return manifest


def write_review_state(
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    owner_id: str,
    score_id: str,
    reviewer: str,
    decisions: dict,
) -> Path:
    """Atomically persist workspace decisions after every explicit save action."""

    _validated_job_manifest(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version=pipeline_version,
        owner_id=owner_id,
    )
    if not isinstance(decisions, dict):
        raise ValueError("review decisions must be an object keyed by review item")
    path = job_dir / "review_state.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "alignment_fingerprint": fingerprint,
            "owner_id": owner_id,
            "score_id": score_id,
            "pipeline_version": pipeline_version,
            "reviewer": reviewer.strip() or "User",
            "decisions": decisions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    os.utime(job_dir, None)
    return path


def load_review_state(
    job_dir: Path,
    *,
    fingerprint: str,
    pipeline_version: str,
    owner_id: str,
    score_id: str,
) -> dict | None:
    """Restore authenticated review data, rejecting stale or cross-user state."""

    path = job_dir / "review_state.json"
    if not path.is_file():
        return None
    _validated_job_manifest(
        job_dir,
        fingerprint=fingerprint,
        pipeline_version=pipeline_version,
        owner_id=owner_id,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("persisted review state is invalid JSON") from error
    expected = {
        "schema_version": "1.0",
        "alignment_fingerprint": fingerprint,
        "owner_id": owner_id,
        "score_id": score_id,
        "pipeline_version": pipeline_version,
    }
    mismatched = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatched:
        raise ValueError(
            "persisted review state identity mismatch: " + ", ".join(mismatched)
        )
    if not isinstance(payload.get("decisions"), dict):
        raise ValueError("persisted review decisions must be an object")
    return payload


def append_job_event(job_dir: Path, event: dict) -> Path:
    """Append one machine-readable event without exposing uploaded contents."""

    path = job_dir / "job_events.jsonl"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_dir.name,
        **event,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
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
    try:
        append_job_event(
            job_dir,
            {
                "event": "status",
                "state": state,
                "stage": stage,
                "completed_pages": int(completed_pages),
                "total_pages": int(total_pages),
                "message": message,
                "error": error,
            },
        )
    except OSError:
        # A readable status remains more important than optional event history.
        pass
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
    """Package signed resumable artifacts without original uploaded inputs."""

    allowed_roots = [job_dir / "checkpoints", job_dir / "outputs"]
    manifest = job_dir / "job_manifest.json"
    if not manifest.is_file():
        raise ValueError("job manifest is missing")
    try:
        job_identity = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("job manifest is invalid") from error
    members = [manifest]
    for optional_name in ("job_status.json", "review_state.json"):
        optional = job_dir / optional_name
        if optional.is_file():
            members.append(optional)
    for root in allowed_roots:
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            # The final download archive is reproducible from the individual
            # outputs and otherwise duplicates nearly the entire checkpoint.
            if path.name in {"all_outputs.zip", "alignment_job_checkpoint.zip"}:
                continue
            members.append(path)
    member_metadata = [
        {
            "path": path.relative_to(job_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in members
    ]
    checkpoint_manifest = {
        "schema_version": CHECKPOINT_ARCHIVE_SCHEMA,
        "job_id": job_identity.get("job_id", ""),
        "owner_id": job_identity.get("owner_id", ""),
        "pipeline_version": job_identity.get("pipeline_version", ""),
        "code_signature": job_identity.get("code_signature", ""),
        "members": member_metadata,
    }
    envelope = {
        **checkpoint_manifest,
        "hmac_sha256": _checkpoint_signature(
            checkpoint_manifest,
            _checkpoint_secret(job_dir),
        ),
    }
    _private_directory(destination.parent)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "checkpoint_manifest.json",
            json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        )
        for path in members:
            archive.write(path, arcname=path.relative_to(job_dir).as_posix())
    os.chmod(destination, 0o600)
    return destination


def restore_job_checkpoint_archive(
    data: bytes,
    job_dir: Path,
    *,
    expected_fingerprint: str,
    expected_pipeline_version: str | None = None,
    expected_code_signature: str | None = None,
    expected_owner_id: str | None = None,
    max_uncompressed_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    max_member_bytes: int = DEFAULT_MAX_CHECKPOINT_MEMBER_BYTES,
    max_members: int = DEFAULT_MAX_CHECKPOINT_MEMBERS,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> int:
    """Safely restore a portable checkpoint archive into its fingerprinted job."""

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        try:
            envelope = json.loads(archive.read("checkpoint_manifest.json"))
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                "checkpoint ZIP has no valid signed checkpoint manifest"
            ) from error
        signature = str(envelope.pop("hmac_sha256", ""))
        if envelope.get("schema_version") != CHECKPOINT_ARCHIVE_SCHEMA:
            raise ValueError("checkpoint ZIP uses an unsupported archive schema")
        expected_signature = _checkpoint_signature(
            envelope,
            _checkpoint_secret(job_dir),
        )
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("checkpoint ZIP signature is invalid")
        if envelope.get("job_id") != expected_fingerprint:
            raise ValueError("checkpoint ZIP belongs to different uploaded inputs")
        if (
            expected_pipeline_version is not None
            and envelope.get("pipeline_version") != expected_pipeline_version
        ):
            raise ValueError(
                "checkpoint ZIP was created by a different pipeline version"
            )
        if (
            expected_code_signature is not None
            and envelope.get("code_signature") != expected_code_signature
        ):
            raise ValueError("checkpoint ZIP was created by different alignment code")
        if (
            expected_owner_id is not None
            and str(envelope.get("owner_id", "")) != str(expected_owner_id)
        ):
            raise ValueError(
                "checkpoint ZIP belongs to a different authenticated user"
            )
        declared_members = envelope.get("members")
        if not isinstance(declared_members, list):
            raise ValueError("checkpoint ZIP member manifest is invalid")
        if len(declared_members) > max_members:
            raise ValueError("checkpoint ZIP contains too many files")
        declared_by_path = {}
        for item in declared_members:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("checkpoint ZIP member manifest is invalid")
            path = item["path"]
            if path in declared_by_path:
                raise ValueError(f"checkpoint ZIP contains duplicate path: {path}")
            declared_by_path[path] = item
        members = [member for member in archive.infolist() if not member.is_dir()]
        actual_paths = {member.filename for member in members}
        if len(actual_paths) != len(members):
            raise ValueError("checkpoint ZIP contains duplicate member names")
        if actual_paths != set(declared_by_path) | {"checkpoint_manifest.json"}:
            raise ValueError("checkpoint ZIP members do not match its signed manifest")
        payload_members = [
            member for member in members
            if member.filename != "checkpoint_manifest.json"
        ]
        total = sum(member.file_size for member in payload_members)
        if total > max_uncompressed_bytes:
            raise ValueError("checkpoint ZIP expands beyond the allowed size")
        for member in payload_members:
            if member.file_size > max_member_bytes:
                raise ValueError(
                    f"checkpoint ZIP member exceeds the allowed size: {member.filename}"
                )
            compressed = max(member.compress_size, 1)
            if member.file_size / compressed > max_compression_ratio:
                raise ValueError(
                    f"checkpoint ZIP member has an unsafe compression ratio: {member.filename}"
                )
        restored = 0
        _private_directory(job_dir)
        for member in payload_members:
            relative = Path(member.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.parts[0]
                not in {
                    "job_manifest.json",
                    "job_status.json",
                    "review_state.json",
                    "checkpoints",
                    "outputs",
                }
            ):
                raise ValueError(f"unsafe checkpoint ZIP path: {member.filename}")
            destination = job_dir / relative
            _private_directory(destination.parent)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.restore"
            )
            digest = hashlib.sha256()
            written = 0
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with archive.open(member) as source, os.fdopen(descriptor, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_member_bytes:
                            raise ValueError(
                                f"checkpoint ZIP member exceeds the allowed size: {member.filename}"
                            )
                        digest.update(chunk)
                        target.write(chunk)
                declared = declared_by_path[member.filename]
                if (
                    written != int(declared.get("size", -1))
                    or digest.hexdigest() != declared.get("sha256")
                ):
                    raise ValueError(
                        f"checkpoint ZIP member failed integrity validation: {member.filename}"
                    )
                if destination.name == "job_manifest.json" and destination.exists():
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                    incoming = json.loads(temporary.read_text(encoding="utf-8"))
                    identity_fields = (
                        "job_id",
                        "owner_id",
                        "pipeline_version",
                        "code_signature",
                    )
                    if any(
                        existing.get(field) != incoming.get(field)
                        for field in identity_fields
                    ):
                        raise ValueError(
                            "checkpoint ZIP job identity conflicts with current job"
                        )
                    temporary.unlink()
                else:
                    temporary.replace(destination)
                    os.chmod(destination, 0o600)
            finally:
                temporary.unlink(missing_ok=True)
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
