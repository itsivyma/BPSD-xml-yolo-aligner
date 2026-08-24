"""Build a normalized learning/calibration dataset from website reviews."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bpsd_aligner.review_corrections import field_values_equivalent
from bpsd_aligner.thresholds import auto_accept_threshold, threshold_family
from pipeline_checkpoint import atomic_write_csv, atomic_write_json, emit_progress


CALIBRATION_ACTIONS = {"confirm", "correct", "reject", "scan_only"}
NEGATIVE_ACTIONS = {"correct", "reject", "scan_only"}
MODEL_ACTIONS = {"wrong_class", "bad_bbox", "not_a_symbol"}
SEMANTIC_FIELDS = (
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
    "staff",
)
REVIEW_SAMPLE_FIELDS = (
    "alignment_fingerprint",
    "score_id",
    "pipeline_version",
    "source_predictions",
    "source_corrections",
    "review_full_image",
    "review_crop_image",
    "page_id",
    "yolo_line",
    "class_id",
    "class",
    "threshold_family",
    "action",
    "calibration_eligible",
    "alignment_correct",
    "requires_human_or_model",
    "machine_status",
    "match_source",
    "confidence",
    "geometry_score",
    "candidate_margin",
    "count_agreement",
    "xml_time_confirmed",
    "current_threshold",
    "corrected_class_id",
    "corrected_class",
    *tuple(f"machine_{field}" for field in SEMANTIC_FIELDS),
    *tuple(f"expected_{field}" for field in SEMANTIC_FIELDS),
    "comment",
)


def _text(value: object) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value.upper() == "NA" else value


def _score(row: dict) -> str:
    return _text(row.get("match_score")) or _text(row.get("confidence"))


def _prediction_key(row: dict) -> tuple[str, str]:
    page_id = _text(row.get("page_id"))
    yolo_line = _text(row.get("txt_line", row.get("yolo_line")))
    candidate_set_id = _text(row.get("review_candidate_set_id"))
    if not page_id and ":Y" in candidate_set_id:
        inferred_page, inferred_line = candidate_set_id.rsplit(":Y", 1)
        page_id = inferred_page
        if not yolo_line:
            yolo_line = inferred_line
    return page_id, yolo_line


def build_review_samples(
    detailed_rows: list[dict],
    corrections_payload: dict,
    *,
    source_predictions: str = "",
    source_corrections: str = "",
) -> tuple[list[dict], list[str]]:
    """Join machine evidence with final human decisions without image data."""

    predictions = {_prediction_key(row): row for row in detailed_rows}
    samples: list[dict] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(corrections_payload.get("entries", []), start=1):
        key = (_text(entry.get("page_id")), _text(entry.get("yolo_line")))
        if not all(key):
            errors.append(f"correction entry {index} is missing page_id/yolo_line")
            continue
        if key in seen:
            errors.append(f"duplicate correction key {key[0]}:Y{key[1]}")
            continue
        seen.add(key)
        machine = predictions.get(key)
        if machine is None:
            errors.append(f"missing prediction for {key[0]}:Y{key[1]}")
            continue
        action = _text(entry.get("action"))
        class_name = _text(machine.get("class")) or _text(entry.get("class"))
        family = threshold_family(class_name)
        score = _score(machine)
        calibration_eligible = action in CALIBRATION_ACTIONS and bool(score)
        alignment_correct = (
            "1" if action == "confirm" else "0" if action in NEGATIVE_ACTIONS else ""
        )
        corrected = entry.get("corrected", {})
        if not isinstance(corrected, dict):
            corrected = {}
        machine_original = entry.get("original", {})
        if not isinstance(machine_original, dict):
            machine_original = {}
        requires_model = (
            action in MODEL_ACTIONS
            or class_name == "fingeringSubstitution"
            or _text(machine.get("xml_time_confirmed")).lower() not in {"true", "1"}
        )
        sample = {
            "alignment_fingerprint": _text(
                corrections_payload.get("alignment_fingerprint")
            ),
            "score_id": _text(corrections_payload.get("score_id")),
            "pipeline_version": _text(corrections_payload.get("pipeline_version")),
            "source_predictions": source_predictions,
            "source_corrections": source_corrections,
            "review_full_image": "",
            "review_crop_image": "",
            "page_id": key[0],
            "yolo_line": key[1],
            "class_id": _text(machine.get("class_id")),
            "class": class_name,
            "threshold_family": family,
            "action": action,
            "calibration_eligible": "1" if calibration_eligible else "0",
            "alignment_correct": alignment_correct,
            "requires_human_or_model": "1" if requires_model else "0",
            "machine_status": _text(
                machine.get("status", machine.get("alignment_status"))
            ),
            "match_source": _text(machine.get("match_source")),
            "confidence": score,
            "geometry_score": _text(machine.get("geometry_score")),
            "candidate_margin": _text(machine.get("candidate_margin")),
            "count_agreement": _text(machine.get("count_agreement")),
            "xml_time_confirmed": _text(machine.get("xml_time_confirmed")),
            "current_threshold": str(auto_accept_threshold(class_name, 1.0)),
            "corrected_class_id": _text(entry.get("corrected_class_id")),
            "corrected_class": _text(entry.get("corrected_class")),
            "comment": _text(entry.get("comment")),
        }
        for field in SEMANTIC_FIELDS:
            source_field = "xml_staff" if field == "staff" else field
            sample[f"machine_{field}"] = _text(
                machine_original.get(field, machine.get(source_field))
            )
            sample[f"expected_{field}"] = _text(corrected.get(field))
        samples.append(sample)
    return samples, errors


def build_ground_truth_samples(
    detailed_rows: list[dict],
    ground_truth_rows: list[dict],
    *,
    source_predictions: str = "",
    source_ground_truth: str = "",
) -> tuple[list[dict], list[str]]:
    """Join predictions to normalized human ground truth for calibration.

    Only rows explicitly marked ``confirmed`` are calibration eligible. Empty
    expected fields are unknown rather than negative labels and are ignored.
    """

    predictions = {_prediction_key(row): row for row in detailed_rows}
    samples: list[dict] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, truth in enumerate(ground_truth_rows, start=2):
        key = (_text(truth.get("page_id")), _text(truth.get("yolo_line")))
        if not all(key):
            errors.append(f"ground-truth row {index} is missing page_id/yolo_line")
            continue
        if key in seen:
            errors.append(f"duplicate ground-truth key {key[0]}:Y{key[1]}")
            continue
        seen.add(key)
        machine = predictions.get(key)
        if machine is None:
            errors.append(f"missing prediction for {key[0]}:Y{key[1]}")
            continue

        class_name = _text(machine.get("class"))
        expected_class = _text(truth.get("class"))
        score = _score(machine)
        reviewed = _text(truth.get("review_status")).lower() == "confirmed"
        evaluated_fields = [
            field
            for field in SEMANTIC_FIELDS
            if _text(truth.get(f"expected_{field}"))
        ]
        correct = (not expected_class or class_name == expected_class) and all(
            field_values_equivalent(
                field,
                machine.get("xml_staff" if field == "staff" else field),
                truth.get(f"expected_{field}"),
            )
            for field in evaluated_fields
        )
        eligible = reviewed and bool(score) and bool(evaluated_fields)
        sample = {
            "alignment_fingerprint": "",
            "score_id": "",
            "pipeline_version": "",
            "source_predictions": source_predictions,
            "source_corrections": source_ground_truth,
            "review_full_image": "",
            "review_crop_image": "",
            "page_id": key[0],
            "yolo_line": key[1],
            "class_id": _text(machine.get("class_id")),
            "class": class_name,
            "threshold_family": threshold_family(class_name),
            "action": "ground_truth_confirm" if reviewed else "ground_truth_pending",
            "calibration_eligible": "1" if eligible else "0",
            "alignment_correct": "1" if correct else "0",
            "requires_human_or_model": "0" if reviewed else "1",
            "machine_status": _text(machine.get("status")),
            "match_source": _text(machine.get("match_source")),
            "confidence": score,
            "geometry_score": _text(machine.get("geometry_score")),
            "candidate_margin": _text(machine.get("candidate_margin")),
            "count_agreement": _text(machine.get("count_agreement")),
            "xml_time_confirmed": _text(machine.get("xml_time_confirmed")),
            "current_threshold": str(auto_accept_threshold(class_name, 1.0)),
            "corrected_class_id": _text(truth.get("class_id")),
            "corrected_class": expected_class,
            "comment": _text(truth.get("comment")),
        }
        for field in SEMANTIC_FIELDS:
            source_field = "xml_staff" if field == "staff" else field
            sample[f"machine_{field}"] = _text(machine.get(source_field))
            sample[f"expected_{field}"] = _text(truth.get(f"expected_{field}"))
        samples.append(sample)
    return samples, errors


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--corrections", type=Path, action="append")
    source_group.add_argument("--ground-truth", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_paths = args.corrections or args.ground_truth or []
    if len(args.predictions) != len(source_paths):
        parser.error("provide the same number of --predictions and review files")
    all_samples: list[dict] = []
    errors: list[str] = []
    total = len(args.predictions)
    emit_progress("review-dataset", 0, total, "joining reviewed jobs")
    for index, (prediction_path, source_path) in enumerate(
        zip(args.predictions, source_paths), start=1
    ):
        if args.corrections:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            samples, pair_errors = build_review_samples(
                _read_csv(prediction_path),
                payload,
                source_predictions=prediction_path.name,
                source_corrections=source_path.name,
            )
        else:
            samples, pair_errors = build_ground_truth_samples(
                _read_csv(prediction_path),
                _read_csv(source_path),
                source_predictions=prediction_path.name,
                source_ground_truth=source_path.name,
            )
        all_samples.extend(samples)
        errors.extend(f"{source_path.name}: {error}" for error in pair_errors)
        emit_progress("review-dataset", index, total, source_path.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "review_training_rows.csv"
    report_path = args.output_dir / "review_dataset_report.json"
    atomic_write_csv(dataset_path, REVIEW_SAMPLE_FIELDS, all_samples)
    report = {
        "schema_version": "1.0",
        "jobs": total,
        "rows": len(all_samples),
        "calibration_eligible_rows": sum(
            row["calibration_eligible"] == "1" for row in all_samples
        ),
        "model_candidate_rows": sum(
            row["requires_human_or_model"] == "1" for row in all_samples
        ),
        "action_counts": dict(Counter(row["action"] for row in all_samples)),
        "class_counts": dict(Counter(row["class"] for row in all_samples)),
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {"training_csv": str(dataset_path)},
    }
    atomic_write_json(report_path, report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
