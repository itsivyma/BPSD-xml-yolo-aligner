"""Single durable dispatcher for queued website alignment jobs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bpsd_aligner.job_store import job_store_root, prune_job_store
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
    if configured:
        prune_job_store(root=root, retention_hours=float(configured))


def run_dispatcher(
    root: Path,
    *,
    idle_grace_seconds: float = 2.0,
    max_idle_checks: int = 2,
) -> int:
    """Drain queued jobs; a short idle grace closes queue-arrival races."""

    completed = 0
    idle_checks = 0
    while idle_checks < max_idle_checks:
        requests = queued_requests(root)
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
            except Exception:
                # The worker has already persisted a failed status and traceback.
                # Continue draining unrelated jobs instead of blocking the queue.
                pass
            completed += 1
            _prune_if_configured(root)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=job_store_root())
    parser.add_argument("--idle-grace-seconds", type=float, default=2.0)
    args = parser.parse_args()
    run_dispatcher(
        args.job_dir.expanduser().resolve(),
        idle_grace_seconds=args.idle_grace_seconds,
    )


if __name__ == "__main__":
    main()
