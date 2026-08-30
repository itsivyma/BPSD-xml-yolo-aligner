"""Unified command-line entry point for the alignment toolkit."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from bpsd_aligner import __version__


COMMANDS = {
    "align": ("bpsd_aligner.bps_xml_alignment", "Align one score page"),
    "batch-align": ("bpsd_aligner.batch_align", "Align a score page range"),
    "inventory": ("bpsd_aligner.dataset_inventory", "Build dataset manifests"),
    "dry-run": ("bpsd_aligner.dataset_dry_run", "Run resumable dataset alignment"),
    "regression-smoke": (
        "bpsd_aligner.regression_smoke",
        "Run a resumable real-page regression suite",
    ),
    "xml-export": ("bpsd_aligner.xml_export", "Export every MusicXML event to CSV"),
    "combine": ("bpsd_aligner.combine_yolo_xml", "Build a lossless XML + YOLO CSV"),
    "review-queue": ("bpsd_aligner.build_review_queue", "Build a human-review queue"),
    "reduce-review": ("bpsd_aligner.reduce_review_queue", "Reduce a review queue"),
    "review-sheets": ("bpsd_aligner.alignment_review_sheets", "Render review sheets"),
    "render-overlays": ("bpsd_aligner.render_yolo_overlays", "Render YOLO page overlays"),
    "apply-review": (
        "bpsd_aligner.apply_human_audit_feedback",
        "Apply reviewed decisions",
    ),
    "review-eval": (
        "bpsd_aligner.review_corrections",
        "Normalize human evaluation ground truth",
    ),
    "validate-evidence": (
        "bpsd_aligner.review_evidence",
        "Validate archived human review evidence",
    ),
    "review-dataset": (
        "bpsd_aligner.review_dataset",
        "Build calibration/training rows from website reviews",
    ),
    "calibrate-thresholds": (
        "bpsd_aligner.calibrate_thresholds",
        "Recommend thresholds from reviewed alignment rows",
    ),
    "job-admin": (
        "bpsd_aligner.job_admin",
        "Inspect or prune persisted website jobs",
    ),
    "worker": (
        "bpsd_aligner.web_worker",
        "Run one persisted background job request",
    ),
    "dispatcher": (
        "bpsd_aligner.job_dispatcher",
        "Drain persisted background job requests",
    ),
}


def _help() -> str:
    lines = [
        "BPSD MusicXML–YOLO Aligner",
        "",
        "Usage:",
        "  bpsd-aligner <command> [options]",
        "  bpsd-aligner web [streamlit options]",
        "",
        "Commands:",
    ]
    width = max(len(command) for command in [*COMMANDS, "web"])
    for command, (_module, description) in COMMANDS.items():
        lines.append(f"  {command:<{width}}  {description}")
    lines.append(f"  {'web':<{width}}  Start the Streamlit website")
    lines.extend(
        [
            "",
            "Use 'bpsd-aligner <command> --help' for stage-specific options.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_help())
        return
    if args[0] == "--version":
        print(__version__)
        return
    command = args.pop(0)
    if command == "web":
        app_path = Path(__file__).with_name("web.py")
        completed = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), *args],
            check=False,
        )
        raise SystemExit(completed.returncode)
    selected = COMMANDS.get(command)
    if selected is None:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(_help(), file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(selected[0])
    original_argv = sys.argv
    try:
        sys.argv = [f"bpsd-aligner {command}", *args]
        module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
