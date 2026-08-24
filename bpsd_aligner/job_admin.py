"""Inspect or prune the persistent website job store."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bpsd_aligner.job_store import (
    job_store_root,
    job_store_usage_bytes,
    prune_job_store,
)


def collect_job_statuses(root: Path | None = None) -> dict:
    """Return privacy-preserving operational status for fingerprinted jobs."""

    store = (root or job_store_root()).expanduser()
    jobs = []
    if store.is_dir():
        for directory in sorted(store.iterdir()):
            name = directory.name.lower()
            if (
                not directory.is_dir()
                or len(name) != 64
                or any(character not in "0123456789abcdef" for character in name)
            ):
                continue
            status_path = directory / "job_status.json"
            manifest_path = directory / "job_manifest.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            jobs.append(
                {
                    "job_id": name[:12],
                    "owner_id": str(manifest.get("owner_id", ""))[:12],
                    "state": status.get("state", "unknown"),
                    "stage": status.get("stage", ""),
                    "completed_pages": status.get("completed_pages", 0),
                    "total_pages": status.get("total_pages", 0),
                    "updated_at": status.get("updated_at", ""),
                    "error": status.get("error", ""),
                }
            )
    counts = Counter(job["state"] for job in jobs)
    return {
        "schema_version": "1.0",
        "job_store": str(store),
        "job_count": len(jobs),
        "storage_bytes": job_store_usage_bytes(store),
        "state_counts": dict(sorted(counts.items())),
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status", help="Print job states as JSON")
    prune = subparsers.add_parser("prune", help="Remove old inactive jobs")
    prune.add_argument("--retention-hours", type=float, required=True)
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(collect_job_statuses(args.job_dir), ensure_ascii=False, indent=2))
        return
    removed = prune_job_store(
        root=args.job_dir,
        retention_hours=args.retention_hours,
    )
    print(
        json.dumps(
            {"removed_count": len(removed), "removed_job_ids": removed},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
