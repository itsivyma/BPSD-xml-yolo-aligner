"""Validate immutable, pre-website human review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESOLVED_STATUS = "resolved_human_review"
BATCH_STATUS = "spot_check_batch_passed"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}


def validate_review_evidence(directory: Path) -> dict:
    """Validate archived evidence and summarize what is safe to normalize.

    A withdrawal invalidates both its own row and the earlier decision with the
    same sample/bounding-box identity.  Batch spot checks are counted
    separately and never become row-level approvals.
    """

    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError(f"no review evidence JSON files found in {directory}")

    errors: list[str] = []
    resolved_entries: list[tuple[str, dict]] = []
    batch_acceptances = 0
    for path in files:
        label = path.name
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: invalid JSON: {exc}")
            continue
        if not isinstance(document, dict):
            errors.append(f"{label}: top-level value must be an object")
            continue
        if str(document.get("schema_version", "")) not in SUPPORTED_SCHEMA_VERSIONS:
            errors.append(f"{label}: unsupported schema_version")
        status = document.get("status")
        if status == RESOLVED_STATUS:
            entries = document.get("entries")
            if not isinstance(entries, list) or not entries:
                errors.append(f"{label}: resolved evidence requires non-empty entries")
                continue
            for index, entry in enumerate(entries, start=1):
                entry_label = f"{label}: entry {index}"
                if not isinstance(entry, dict):
                    errors.append(f"{entry_label} must be an object")
                    continue
                for field in ("sample_id", "bbox_id", "audit_decision", "resolution_status"):
                    if not str(entry.get(field, "")).strip():
                        errors.append(f"{entry_label}: missing {field}")
                withdrawn = entry.get("audit_decision") == "withdrawn_typo"
                if withdrawn:
                    if entry.get("apply_to_audit") is not False:
                        errors.append(
                            f"{entry_label}: withdrawn_typo requires apply_to_audit=false"
                        )
                    if not str(entry.get("resolution_status", "")).startswith("withdrawn"):
                        errors.append(
                            f"{entry_label}: withdrawn_typo requires withdrawn resolution_status"
                        )
                elif entry.get("apply_to_audit") is False:
                    errors.append(
                        f"{entry_label}: apply_to_audit=false requires a withdrawal decision"
                    )
                resolved_entries.append((entry_label, entry))
        elif status == BATCH_STATUS:
            batch_acceptances += 1
            if document.get("verification_level") != "sampled_not_individually_verified":
                errors.append(
                    f"{label}: batch acceptance must remain sampled_not_individually_verified"
                )
            if document.get("per_row_human_approved") is not False:
                errors.append(f"{label}: batch acceptance cannot approve every row")
            if document.get("preserve_automatic_alignment") is not True:
                errors.append(f"{label}: batch acceptance must preserve automatic alignment")
            if document.get("entries"):
                errors.append(f"{label}: batch acceptance cannot contain row decisions")
        else:
            errors.append(f"{label}: unsupported status {status!r}")

    withdrawn_keys = {
        (str(entry.get("sample_id", "")), str(entry.get("bbox_id", "")))
        for _label, entry in resolved_entries
        if entry.get("audit_decision") == "withdrawn_typo"
    }
    active_entries = [
        entry
        for _label, entry in resolved_entries
        if (
            str(entry.get("sample_id", "")),
            str(entry.get("bbox_id", "")),
        )
        not in withdrawn_keys
        and entry.get("apply_to_audit") is not False
    ]

    if errors:
        raise ValueError("review evidence validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "valid",
        "file_count": len(files),
        "resolved_entry_count": len(resolved_entries),
        "withdrawn_identity_count": len(withdrawn_keys),
        "active_row_decision_count": len(active_entries),
        "batch_spot_check_count": batch_acceptances,
        "batch_rows_promoted_to_human_approved": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate archived human review evidence without applying it"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("regression/review_evidence"),
    )
    args = parser.parse_args()
    print(json.dumps(validate_review_evidence(args.directory), indent=2))


if __name__ == "__main__":
    main()
