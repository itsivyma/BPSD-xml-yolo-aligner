"""Run a small, resumable real-score regression suite."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from bpsd_aligner import __version__
from bpsd_aligner.provenance import alignment_runtime_identity
from bpsd_aligner.review_corrections import evaluate_ground_truth_rows
from bpsd_aligner.web_pipeline import prepare_score_sources, run_uploaded_alignment
from bpsd_aligner.pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


SUMMARY_FIELDS = [
    "page_id",
    "score_id",
    "page_number",
    "tags",
    "passed",
    "resumed",
    "alignment_rows",
    "matched_rows",
    "inferred_rows",
    "review_rows",
    "blank_time_rows",
    "cross_page_candidates",
    "validation_error_count",
    "missing_required_classes",
    "required_class_status_counts",
    "class_status_counts",
    "semantic_digest",
    "semantic_class_digests",
    "error",
    "detailed_csv",
    "validation_json",
]
BASELINE_FIELDS = [
    "alignment_rows",
    "matched_rows",
    "inferred_rows",
    "review_rows",
    "blank_time_rows",
    "cross_page_candidates",
    "required_class_status_counts",
    "semantic_digest",
    "semantic_class_digests",
]
REQUIRED_PATHS = ("notes_json", "image", "yolo", "xml_repetitions", "bps_notes")
OPTIONAL_PATHS = ("xml_unfolded",)
SEMANTIC_ROW_FIELDS = (
    "txt_line",
    "class_id",
    "class",
    "x",
    "y",
    "w",
    "h",
    "status",
    "match_source",
    "confidence",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "xml_staff",
    "start_xml_measure",
    "end_xml_measure",
    "start_xml_page",
    "end_xml_page",
    "cross_page_span_id",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _semantic_fingerprints(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Hash every alignment row's stable geometry and musical assignment."""

    canonical_rows = [
        {field: str(row.get(field, "")).strip() for field in SEMANTIC_ROW_FIELDS}
        for row in rows
    ]
    canonical_rows.sort(
        key=lambda row: (
            int(row["txt_line"]) if row["txt_line"].isdigit() else 10**12,
            row["txt_line"],
            row["class_id"],
            row["class"],
        )
    )
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical_rows:
        by_class[row["class"]].append(row)
    class_digests = {
        class_name: stable_digest(class_rows)
        for class_name, class_rows in sorted(by_class.items())
    }
    return (
        stable_digest(canonical_rows),
        json.dumps(class_digests, ensure_ascii=False, sort_keys=True),
    )


def _root_path(manifest_path: Path, payload: dict, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = Path(str(payload.get("dataset_root", "."))).expanduser()
    if not configured.is_absolute():
        configured = manifest_path.parent / configured
    return configured.resolve()


def _resolve_template(
    value: object,
    *,
    root: Path,
    context: dict[str, object],
) -> Path:
    rendered = str(value).format_map(context)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_regression_manifest(
    manifest_path: Path,
    *,
    dataset_root: Path | None = None,
    page_ids: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Resolve and validate one portable regression manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("regression manifest schema_version must be 1.0")
    root = _root_path(manifest_path, payload, dataset_root)
    templates = payload.get("paths")
    pages = payload.get("pages")
    if not isinstance(templates, dict) or not isinstance(pages, list):
        raise ValueError("regression manifest requires paths and pages")
    resolved = []
    seen = set()
    for raw in pages:
        if not isinstance(raw, dict):
            raise ValueError("every regression page must be an object")
        page_id = str(raw.get("page_id", "")).strip()
        score_id = str(raw.get("score_id", "")).strip()
        if not page_id or not score_id:
            raise ValueError("every regression page requires page_id and score_id")
        if page_id in seen:
            raise ValueError(f"duplicate regression page_id: {page_id}")
        seen.add(page_id)
        if page_ids and page_id not in page_ids:
            continue
        try:
            page_number = int(raw["page_number"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{page_id}: page_number must be an integer") from error
        context = {
            "page_id": page_id,
            "score_id": score_id,
            "page_number": page_number,
        }
        item = {
            **raw,
            **context,
            "tags": [str(value) for value in raw.get("tags", [])],
            "required_classes": [
                str(value) for value in raw.get("required_classes", [])
            ],
        }
        for name in (*REQUIRED_PATHS, *OPTIONAL_PATHS):
            template = raw.get(name, templates.get(name))
            if template in {None, ""}:
                if name in REQUIRED_PATHS:
                    raise ValueError(f"{page_id}: missing path template {name}")
                item[name] = None
                continue
            item[name] = _resolve_template(template, root=root, context=context)
        missing = [
            name
            for name in REQUIRED_PATHS
            if not Path(item[name]).is_file()
        ]
        if item.get("xml_unfolded") is not None and not Path(
            item["xml_unfolded"]
        ).is_file():
            missing.append("xml_unfolded")
        if missing:
            raise FileNotFoundError(f"{page_id}: missing inputs: {', '.join(missing)}")
        resolved.append(item)
    if page_ids:
        missing_ids = sorted(page_ids - {item["page_id"] for item in resolved})
        if missing_ids:
            raise ValueError("unknown regression page_id: " + ", ".join(missing_ids))
    if not resolved:
        raise ValueError("regression manifest selected no pages")
    expected_class_count = payload.get("expected_class_count")
    if expected_class_count is not None:
        try:
            expected_class_count = int(expected_class_count)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_class_count must be an integer") from error
        notes_paths = {Path(item["notes_json"]) for item in resolved}
        for notes_path in notes_paths:
            notes_payload = json.loads(notes_path.read_text(encoding="utf-8"))
            categories = notes_payload.get("categories", [])
            if not isinstance(categories, list):
                raise ValueError(f"{notes_path}: categories must be a list")
            if len(categories) != expected_class_count:
                raise ValueError(
                    f"{notes_path}: expected {expected_class_count} classes for "
                    f"profile {payload.get('dataset_profile', '(unnamed)')}, "
                    f"found {len(categories)}"
                )
    return payload, resolved


def _page_fingerprint(page: dict) -> str:
    inputs = [Path(page[name]) for name in REQUIRED_PATHS]
    if page.get("xml_unfolded") is not None:
        inputs.append(Path(page["xml_unfolded"]))
    return stable_digest(
        {
            "pipeline_version": __version__,
            **alignment_runtime_identity(),
            "page": {
                key: value
                for key, value in page.items()
                if key not in (*REQUIRED_PATHS, *OPTIONAL_PATHS)
            },
            "inputs": [path_signature(path) for path in inputs],
        }
    )


def _load_checkpoint(path: Path, fingerprint: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = checkpoint.get("summary")
    if (
        checkpoint.get("fingerprint") != fingerprint
        or checkpoint.get("status") != "completed"
        or not isinstance(summary, dict)
        or not Path(str(summary.get("detailed_csv", ""))).is_file()
        or not Path(str(summary.get("validation_json", ""))).is_file()
    ):
        return None
    return summary


def _summarize_page(page: dict, report: dict, *, resumed: bool) -> dict:
    detailed_path = Path(report["outputs"]["detailed_csv"]).resolve()
    rows = _read_csv(detailed_path)
    statuses = Counter(row.get("status", "") for row in rows)
    class_statuses: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        class_statuses[str(row.get("class", ""))][str(row.get("status", ""))] += 1
    missing_classes = sorted(
        set(page.get("required_classes", [])) - set(class_statuses)
    )
    errors = list(report.get("validation_errors", []))
    if missing_classes:
        errors.append("missing required classes: " + ", ".join(missing_classes))
    required_class_statuses = {
        name: dict(class_statuses[name])
        for name in sorted(page.get("required_classes", []))
        if name in class_statuses
    }
    semantic_digest, semantic_class_digests = _semantic_fingerprints(rows)
    return {
        "page_id": page["page_id"],
        "score_id": page["score_id"],
        "page_number": page["page_number"],
        "tags": json.dumps(page.get("tags", []), ensure_ascii=False),
        "passed": bool(report.get("passed")) and not missing_classes,
        "resumed": resumed,
        "alignment_rows": len(rows),
        "matched_rows": statuses.get("matched", 0),
        "inferred_rows": statuses.get("inferred", 0),
        "review_rows": statuses.get("review", 0),
        "blank_time_rows": int(report.get("final_rows_with_blank_time", 0)),
        "cross_page_candidates": sum(
            row.get("match_source") == "whole_score_cross_page_span_candidate"
            for row in rows
        ),
        "validation_error_count": len(errors),
        "missing_required_classes": json.dumps(missing_classes, ensure_ascii=False),
        "required_class_status_counts": json.dumps(
            required_class_statuses,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "class_status_counts": json.dumps(
            {
                name: dict(counts)
                for name, counts in sorted(class_statuses.items())
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "semantic_digest": semantic_digest,
        "semantic_class_digests": semantic_class_digests,
        "error": "; ".join(errors),
        "detailed_csv": str(detailed_path),
        "validation_json": str(Path(report["outputs"]["validation_json"]).resolve()),
    }


def _failed_summary(page: dict, error: Exception) -> dict:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "page_id": page["page_id"],
            "score_id": page["score_id"],
            "page_number": page["page_number"],
            "tags": json.dumps(page.get("tags", []), ensure_ascii=False),
            "passed": False,
            "resumed": False,
            "validation_error_count": 1,
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def _baseline_snapshot(rows: list[dict]) -> dict:
    return {
        row["page_id"]: {field: row.get(field, "") for field in BASELINE_FIELDS}
        for row in rows
    }


def _compare_baseline(rows: list[dict], baseline: dict) -> list[str]:
    differences = []
    expected_pages = baseline.get("pages", {}) if isinstance(baseline, dict) else {}
    for page_id, current in _baseline_snapshot(rows).items():
        expected = expected_pages.get(page_id)
        if expected is None:
            differences.append(f"{page_id}: missing from baseline")
            continue
        for field in BASELINE_FIELDS:
            if str(current.get(field, "")) != str(expected.get(field, "")):
                if field == "semantic_class_digests":
                    try:
                        current_classes = json.loads(str(current.get(field, "{}")))
                        expected_classes = json.loads(str(expected.get(field, "{}")))
                    except json.JSONDecodeError:
                        current_classes = {}
                        expected_classes = {}
                    changed = sorted(
                        name
                        for name in set(current_classes) | set(expected_classes)
                        if current_classes.get(name) != expected_classes.get(name)
                    )
                    differences.append(
                        f"{page_id}: row-level musical assignments changed for "
                        f"classes {', '.join(changed) or '(unknown)'}"
                    )
                    continue
                differences.append(
                    f"{page_id}: {field} changed from "
                    f"{expected.get(field, '')!r} to {current.get(field, '')!r}"
                )
    return differences


def run_regression_smoke(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset_root: Path | None = None,
    resume: bool = False,
    page_ids: set[str] | None = None,
    baseline_path: Path | None = None,
    update_baseline: bool = False,
    ground_truth_path: Path | None = None,
) -> dict:
    """Run selected real pages, checkpointing after every completed page."""

    payload, pages = load_regression_manifest(
        manifest_path, dataset_root=dataset_root, page_ids=page_ids
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = (
        baseline_path.expanduser().resolve()
        if baseline_path is not None
        else output_dir / "baseline.json"
    )
    prepared_by_score: dict[str, dict] = {}
    summaries = []
    resumed_pages = 0
    emit_progress("regression-pages", 0, len(pages), f"starting (resume={resume})")
    for index, page in enumerate(pages, start=1):
        page_id = page["page_id"]
        fingerprint = _page_fingerprint(page)
        checkpoint_path = output_dir / "checkpoints" / f"{page_id}.json"
        if resume:
            resumed = _load_checkpoint(checkpoint_path, fingerprint)
            if resumed is not None:
                resumed = {**resumed, "resumed": True}
                summaries.append(resumed)
                resumed_pages += 1
                emit_progress("regression-pages", index, len(pages), f"{page_id} resumed")
                continue
        emit_progress("regression-pages", index, len(pages), f"{page_id} running")
        try:
            score_id = page["score_id"]
            prepared = prepared_by_score.get(score_id)
            if prepared is None:
                prepared = prepare_score_sources(
                    xml_path=Path(page["xml_repetitions"]),
                    unfolded_xml_path=(
                        Path(page["xml_unfolded"])
                        if page.get("xml_unfolded") is not None
                        else None
                    ),
                    bps_notes_path=Path(page["bps_notes"]),
                    output_dir=output_dir / "shared" / score_id,
                    score_id=score_id,
                    resume=resume,
                    include_xml_nodes=False,
                    progress_callback=lambda step, total, message, score=score_id: (
                        emit_progress(f"shared:{score}", step, total, message)
                    ),
                )
                prepared_by_score[score_id] = prepared
            report = run_uploaded_alignment(
                image_path=Path(page["image"]),
                yolo_path=Path(page["yolo"]),
                xml_path=Path(page["xml_repetitions"]),
                unfolded_xml_path=(
                    Path(page["xml_unfolded"])
                    if page.get("xml_unfolded") is not None
                    else None
                ),
                bps_notes_path=Path(page["bps_notes"]),
                notes_json_path=Path(page["notes_json"]),
                output_dir=output_dir / "pages" / page_id,
                page_number=int(page["page_number"]),
                score_id=score_id,
                infer_fingerings=True,
                prepared_score=prepared,
                build_complete_exports=False,
                render_qa_images=False,
                progress_callback=lambda step, total, message, item=page_id: (
                    emit_progress(f"page:{item}", step, total, message)
                ),
            )
            summary = _summarize_page(page, report, resumed=False)
        except Exception as error:
            summary = _failed_summary(page, error)
        summaries.append(summary)
        atomic_write_json(
            checkpoint_path,
            {
                "pipeline_version": __version__,
                **alignment_runtime_identity(),
                "fingerprint": fingerprint,
                "status": "completed" if summary["passed"] else "failed",
                "summary": summary,
            },
        )
        emit_progress(
            "regression-pages",
            index,
            len(pages),
            f"{page_id} {'passed' if summary['passed'] else 'failed'}",
        )

    summary_path = output_dir / "regression_summary.csv"
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summaries)
    page_failures = [row["page_id"] for row in summaries if not row["passed"]]
    baseline_differences = []
    baseline_warning = ""
    if update_baseline:
        if page_failures:
            baseline_warning = "baseline not updated because one or more pages failed"
        else:
            existing = {}
            if baseline_path.is_file():
                try:
                    existing = json.loads(baseline_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
            pages_snapshot = dict(existing.get("pages", {}))
            pages_snapshot.update(_baseline_snapshot(summaries))
            atomic_write_json(
                baseline_path,
                {
                    "schema_version": "1.0",
                    "pipeline_version": __version__,
                    **alignment_runtime_identity(),
                    "pages": pages_snapshot,
                },
            )
    elif baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_differences = _compare_baseline(summaries, baseline)
    else:
        baseline_warning = "no baseline found; run once with --update-baseline"

    ground_truth_report = None
    ground_truth_errors: list[str] = []
    if ground_truth_path is not None:
        ground_truth_path = ground_truth_path.expanduser().resolve()
        ground_truth_rows = _read_csv(ground_truth_path)
        prediction_rows = []
        for summary in summaries:
            detailed_path = Path(str(summary.get("detailed_csv", "")))
            if not detailed_path.is_file():
                continue
            for row in _read_csv(detailed_path):
                row["page_id"] = row.get("page_id") or summary["page_id"]
                prediction_rows.append(row)
        ground_truth_report, ground_truth_errors = evaluate_ground_truth_rows(
            ground_truth_rows, prediction_rows
        )
        ground_truth_report["ground_truth_file"] = str(ground_truth_path)
        ground_truth_report_path = output_dir / "ground_truth_accuracy.json"
        atomic_write_json(ground_truth_report_path, ground_truth_report)

    report = {
        "schema_version": "1.0",
        "pipeline_version": __version__,
        **alignment_runtime_identity(),
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "manifest_name": payload.get("name", ""),
        "dataset_profile": payload.get("dataset_profile", "unspecified"),
        "expected_class_count": payload.get("expected_class_count"),
        "selected_pages": [page["page_id"] for page in pages],
        "page_count": len(summaries),
        "resumed_pages": resumed_pages,
        "passed_pages": len(summaries) - len(page_failures),
        "failed_pages": page_failures,
        "baseline": str(baseline_path),
        "baseline_warning": baseline_warning,
        "baseline_differences": baseline_differences,
        "ground_truth": ground_truth_report,
        "ground_truth_errors": ground_truth_errors,
        "passed": (
            not page_failures
            and not baseline_differences
            and not ground_truth_errors
        ),
        "outputs": {"summary_csv": str(summary_path)},
    }
    report_path = output_dir / "regression_report.json"
    atomic_write_json(report_path, report)
    report["outputs"]["report_json"] = str(report_path)
    emit_progress(
        "regression-validation",
        1,
        1,
        f"passed={report['passed']} failed_pages={len(page_failures)} "
        f"baseline_differences={len(baseline_differences)}",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse passing page checkpoints whose inputs have not changed.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional normalized human ground-truth CSV to score against.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record current passing metrics as the expected regression baseline.",
    )
    args = parser.parse_args()
    report = run_regression_smoke(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        baseline_path=args.baseline,
        resume=args.resume,
        page_ids=set(args.page_ids) if args.page_ids else None,
        update_baseline=args.update_baseline,
        ground_truth_path=args.ground_truth,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
