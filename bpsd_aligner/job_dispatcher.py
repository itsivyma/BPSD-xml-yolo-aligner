"""Single durable dispatcher for queued website alignment jobs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bpsd_aligner.job_store import job_store_root, prune_job_store, write_job_status
from bpsd_aligner.web_worker import run_background_job


def queued_requests(root: Path) -> list[Path]:
    """Return valid queued requests oldest first without loading private inputs."""

    queued = []
    if not root.is_dir():
        return queued
    for job_dir in root.iterdir():
        if not job_dir.is_dir() or len(job_dir.name) != 64:
            continue
        status_path = job_dir / "job_status.json"
        request_path = job_dir / "job_request.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") == "queued" and request_path.is_file():
            queued.append((str(status.get("created_at", "")), request_path))
    return [path for _created, path in sorted(queued, key=lambda item: item[0])]


def _prune_if_configured(root: Path) -> None:
    configured = os.environ.get("BPSD_ALIGNER_JOB_RETENTION_HOURS", "").strip()
    if not configured and os.environ.get(
        "BPSD_ALIGNER_DEPLOYMENT_MODE", "local"
    ).lower() == "production":
        configured = "168"
    if configured:
        prune_job_store(root=root, retention_hours=float(configured))


def _mark_dispatch_failure(request: Path, error: Exception) -> None:
    """Stop a preflight failure from remaining queued and spinning forever."""

    job_dir = request.parent
    status_path = job_dir / "job_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        status = {}
    # Failures raised after the worker starts are already persisted by the
    # worker. Only repair errors raised during request/code-signature preflight.
    if status.get("state") != "queued":
        return
    write_job_status(
        job_dir,
        state="failed",
        stage="dispatcher_preflight",
        completed_pages=int(status.get("completed_pages", 0) or 0),
        total_pages=int(status.get("total_pages", 0) or 0),
        message="The background worker could not start this alignment job.",
        error=f"{type(error).__name__}: {error}",
    )


def acquire_dispatcher_lock(root: Path) -> Path | None:
    """Atomically elect one local dispatcher process for this job store."""

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = root / ".dispatcher.lock"
    payload = json.dumps({"pid": os.getpid()})
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            current = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(current.get("pid", -1))
            if pid == os.getpid():
                return lock
            os.kill(pid, 0)
            return None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            lock.unlink(missing_ok=True)
            try:
                descriptor = os.open(
                    lock,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(payload)
    return lock


def release_dispatcher_lock(lock: Path | None) -> None:
    if lock is None:
        return
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) == os.getpid():
            lock.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass


def run_dispatcher(
    root: Path,
    *,
    idle_grace_seconds: float = 2.0,
    max_idle_checks: int = 2,
) -> int:
    """Drain queued jobs; a short idle grace closes queue-arrival races."""

    completed = 0
    idle_checks = 0
    attempted_failures: set[Path] = set()
    while idle_checks < max_idle_checks:
        requests = [
            request
            for request in queued_requests(root)
            if request not in attempted_failures
        ]
        if not requests:
            idle_checks += 1
            _prune_if_configured(root)
            if idle_checks < max_idle_checks:
                time.sleep(max(0.0, idle_grace_seconds))
            continue
        idle_checks = 0
        for request in requests:
            try:
                run_background_job(request)
            except Exception as error:
                attempted_failures.add(request)
                # The worker persists failures raised after its preflight. If
                # validation fails before that try/finally block, repair the
                # still-queued status here so the dispatcher cannot busy-loop.
                _mark_dispatch_failure(request, error)
            completed += 1
            _prune_if_configured(root)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=job_store_root())
    parser.add_argument("--idle-grace-seconds", type=float, default=2.0)
    args = parser.parse_args()
    root = args.job_dir.expanduser().resolve()
    lock = acquire_dispatcher_lock(root)
    if lock is None:
        return
    try:
        run_dispatcher(
            root,
            idle_grace_seconds=args.idle_grace_seconds,
        )
    finally:
        release_dispatcher_lock(lock)


if __name__ == "__main__":
    main()
