"""Recommend conservative per-class-family thresholds from human reviews."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from bpsd_aligner.thresholds import DEFAULT_THRESHOLDS
from pipeline_checkpoint import atomic_write_csv, atomic_write_json, emit_progress


CALIBRATION_FIELDS = (
    "scope",
    "name",
    "reviewed_rows",
    "correct_rows",
    "incorrect_rows",
    "current_threshold",
    "current_accepted",
    "current_precision",
    "current_precision_wilson_low",
    "current_recall",
    "recommended_threshold",
    "recommended_accepted",
    "recommended_precision",
    "recommended_precision_wilson_low",
    "recommended_recall",
    "ready",
    "reason",
)


def _as_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def _wilson_low(successes: int, total: int, z: float = 1.96) -> float | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def _metrics(samples: list[tuple[float, bool]], threshold: float) -> dict:
    accepted = [correct for score, correct in samples if score >= threshold]
    accepted_correct = sum(accepted)
    correct_total = sum(correct for _score, correct in samples)
    return {
        "accepted": len(accepted),
        "precision": accepted_correct / len(accepted) if accepted else None,
        "precision_wilson_low": _wilson_low(accepted_correct, len(accepted)),
        "recall": accepted_correct / correct_total if correct_total else None,
    }


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def calibrate_groups(
    rows: list[dict],
    *,
    target_precision: float = 0.98,
    min_reviewed_per_group: int = 20,
    min_accepted: int = 10,
    min_reviewed_overall: int = 200,
) -> tuple[list[dict], dict[str, float], list[str]]:
    """Evaluate current thresholds and suggest only adequately reviewed changes."""

    errors: list[str] = []
    parsed: list[tuple[dict, float, bool]] = []
    for index, row in enumerate(rows, start=2):
        if str(row.get("calibration_eligible", "")).strip() not in {"1", "true", "True"}:
            continue
        score = _as_float(row.get("confidence"))
        label = str(row.get("alignment_correct", "")).strip()
        if score is None or label not in {"0", "1"}:
            errors.append(f"row {index}: invalid confidence/alignment_correct")
            continue
        parsed.append((row, score, label == "1"))

    grouped: dict[tuple[str, str], list[tuple[float, bool]]] = defaultdict(list)
    current_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, score, correct in parsed:
        family = str(row.get("threshold_family", "")).strip() or "unknown"
        class_name = str(row.get("class", "")).strip() or "unknown"
        for key in (("family", family), ("class", class_name)):
            grouped[key].append((score, correct))
            current = _as_float(row.get("current_threshold"))
            if current is not None:
                current_values[key].append(current)

    overall_ready = len(parsed) >= min_reviewed_overall
    results: list[dict] = []
    recommendations: dict[str, float] = {}
    for (scope, name), samples in sorted(grouped.items()):
        current_candidates = current_values[(scope, name)]
        current_threshold = (
            max(current_candidates)
            if current_candidates
            else DEFAULT_THRESHOLDS.get(name, 1.0)
        )
        current_metrics = _metrics(samples, current_threshold)
        candidate = None
        candidate_metrics = None
        for threshold in sorted({score for score, _correct in samples}):
            metrics = _metrics(samples, threshold)
            if (
                metrics["accepted"] >= min_accepted
                and metrics["precision_wilson_low"] is not None
                and metrics["precision_wilson_low"] >= target_precision
            ):
                candidate = threshold
                candidate_metrics = metrics
                break
        reasons = []
        if not overall_ready:
            reasons.append(
                f"need {min_reviewed_overall} reviewed rows overall; have {len(parsed)}"
            )
        if len(samples) < min_reviewed_per_group:
            reasons.append(
                f"need {min_reviewed_per_group} reviewed rows in group; have {len(samples)}"
            )
        if candidate is None:
            reasons.append(
                f"no threshold reaches Wilson precision lower bound "
                f"{target_precision:.3f} with {min_accepted} accepted rows"
            )
        ready = not reasons
        if ready and scope == "family" and candidate is not None:
            recommendations[name] = round(candidate, 6)
        candidate_metrics = candidate_metrics or {
            "accepted": 0,
            "precision": None,
            "precision_wilson_low": None,
            "recall": None,
        }
        results.append(
            {
                "scope": scope,
                "name": name,
                "reviewed_rows": len(samples),
                "correct_rows": sum(correct for _score, correct in samples),
                "incorrect_rows": sum(not correct for _score, correct in samples),
                "current_threshold": f"{current_threshold:.6f}",
                "current_accepted": current_metrics["accepted"],
                "current_precision": _format_metric(current_metrics["precision"]),
                "current_precision_wilson_low": _format_metric(
                    current_metrics["precision_wilson_low"]
                ),
                "current_recall": _format_metric(current_metrics["recall"]),
                "recommended_threshold": (
                    "" if candidate is None else f"{candidate:.6f}"
                ),
                "recommended_accepted": candidate_metrics["accepted"],
                "recommended_precision": _format_metric(
                    candidate_metrics["precision"]
                ),
                "recommended_precision_wilson_low": _format_metric(
                    candidate_metrics["precision_wilson_low"]
                ),
                "recommended_recall": _format_metric(candidate_metrics["recall"]),
                "ready": "1" if ready else "0",
                "reason": "; ".join(reasons),
            }
        )
    return results, recommendations, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dataset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--min-reviewed-per-group", type=int, default=20)
    parser.add_argument("--min-accepted", type=int, default=10)
    parser.add_argument("--min-reviewed-overall", type=int, default=200)
    args = parser.parse_args()
    if not 0 < args.target_precision <= 1:
        parser.error("--target-precision must be greater than 0 and at most 1")
    if min(
        args.min_reviewed_per_group,
        args.min_accepted,
        args.min_reviewed_overall,
    ) < 1:
        parser.error("minimum sample options must be at least 1")
    rows = []
    emit_progress("threshold-calibration", 0, len(args.review_dataset), "loading")
    for index, path in enumerate(args.review_dataset, start=1):
        with path.open(newline="", encoding="utf-8-sig") as file:
            rows.extend(csv.DictReader(file))
        emit_progress("threshold-calibration", index, len(args.review_dataset), path.name)
    results, recommendations, errors = calibrate_groups(
        rows,
        target_precision=args.target_precision,
        min_reviewed_per_group=args.min_reviewed_per_group,
        min_accepted=args.min_accepted,
        min_reviewed_overall=args.min_reviewed_overall,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "threshold_calibration.csv"
    override_path = args.output_dir / "thresholds.recommended.json"
    report_path = args.output_dir / "threshold_calibration_report.json"
    atomic_write_csv(csv_path, CALIBRATION_FIELDS, results)
    atomic_write_json(override_path, recommendations)
    eligible = sum(
        str(row.get("calibration_eligible", "")).strip() in {"1", "true", "True"}
        for row in rows
    )
    report = {
        "schema_version": "1.0",
        "reviewed_rows": eligible,
        "target_precision": args.target_precision,
        "min_reviewed_overall": args.min_reviewed_overall,
        "min_reviewed_per_group": args.min_reviewed_per_group,
        "min_accepted": args.min_accepted,
        "ready_for_global_update": bool(recommendations) and not errors,
        "recommended_families": recommendations,
        "validation_errors": errors,
        "passed": not errors,
        "warning": (
            "Recommendations require the Wilson precision lower bound to meet the "
            "target; still inspect the fixed real-page regression before deployment."
        ),
        "outputs": {
            "calibration_csv": str(csv_path),
            "threshold_override_json": str(override_path),
        },
    }
    atomic_write_json(report_path, report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
