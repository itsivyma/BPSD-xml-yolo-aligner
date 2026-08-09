"""Evaluate fingering matcher profiles on resolved human audit feedback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import bps_xml_alignment as alignment
from pipeline_checkpoint import atomic_write_csv, atomic_write_json, emit_progress, path_signature, stable_digest


PIPELINE_VERSION = "0.5.0-fingering-profile-eval-safe-threshold"
PROFILES = {
    "baseline": {"dy_weight": 0.18, "staff_penalty_ratio": 0.02, "strict_staff": False},
    "strict_staff": {"dy_weight": 0.18, "staff_penalty_ratio": 0.02, "strict_staff": True},
    "strict_staff_y045": {"dy_weight": 0.45, "staff_penalty_ratio": 0.02, "strict_staff": True},
    "strict_staff_y080": {"dy_weight": 0.80, "staff_penalty_ratio": 0.02, "strict_staff": True},
    "soft_staff_y045": {"dy_weight": 0.45, "staff_penalty_ratio": 0.06, "strict_staff": False},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def run(manifest: Path, audit_csv: Path, feedback_json: Path, repeat_dir: Path, output_dir: Path, *, resume: bool) -> dict:
    pages = {row["page_id"]: row for row in _read_csv(manifest)}
    audit = {row["sample_id"]: row for row in _read_csv(audit_csv)}
    feedback = json.loads(feedback_json.read_text(encoding="utf-8"))
    truths = {}
    for entry in feedback["entries"]:
        sample_id = entry["sample_id"]
        if entry.get("apply_to_audit") is False or audit[sample_id]["class"] == "fingeringSubstitution":
            continue
        truths[audit[sample_id]["bbox_id"]] = {
            "sample_id": sample_id,
            "expected_note_ids": entry["corrected_note_ids"],
            "expected_staff": (entry.get("user_correction") or {}).get("staff"),
            "expected_measure": (entry.get("user_correction") or {}).get("measure"),
        }
    page_ids = sorted({bbox_id.split(":Y", 1)[0] for bbox_id in truths})
    predictions = []
    summaries = []
    total_jobs = len(PROFILES) * len(page_ids)
    job = 0
    for profile_name, profile in PROFILES.items():
        alignment.FINGERING_DY_WEIGHT = profile["dy_weight"]
        alignment.FINGERING_WRONG_STAFF_PENALTY_RATIO = profile["staff_penalty_ratio"]
        alignment.FINGERING_STRICT_STAFF = profile["strict_staff"]
        by_bbox = {}
        for page_id in page_ids:
            job += 1
            page = pages[page_id]
            page_dir = output_dir / "runs" / profile_name / page_id
            detailed_path = page_dir / f"{page_id}_alignment_detailed.csv"
            checkpoint_path = output_dir / "checkpoints" / profile_name / f"{page_id}.json"
            fingerprint = stable_digest({
                "pipeline_version": PIPELINE_VERSION,
                "profile": profile,
                "image": path_signature(Path(page["image_path"])),
                "yolo": path_signature(Path(page["yolo_path"])),
                "xml": path_signature(Path(page["xml_path"])),
            })
            reusable = False
            if resume and detailed_path.is_file() and checkpoint_path.is_file():
                try:
                    reusable = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
                except (OSError, json.JSONDecodeError):
                    reusable = False
            emit_progress("fingering-profile-eval", job, total_jobs, f"{profile_name} {page_id} {'resumed' if reusable else 'running'}")
            if not reusable:
                report = alignment.run_alignment(
                    image_path=Path(page["image_path"]),
                    yolo_path=Path(page["yolo_path"]),
                    xml_path=Path(page["xml_path"]),
                    bps_note_path=Path(page["bps_notes_path"]),
                    output_dir=page_dir,
                    page_number=int(page["page"]),
                    infer_fingerings=True,
                    notes_json_path=Path(page["notes_json_path"]),
                    include_all_symbols=True,
                    repeat_mapping_path=repeat_dir / f"{page['score_id']}_repeat_mapping.csv",
                )
                detailed_path = Path(report["outputs"]["detailed_csv"])
                atomic_write_json(checkpoint_path, {"fingerprint": fingerprint, "profile": profile, "page_id": page_id})
            for row in _read_csv(detailed_path):
                by_bbox[f"{page_id}:Y{row['txt_line']}"] = row
        counts = {
            "exact": 0,
            "wrong": 0,
            "unresolved": 0,
            "autoaccept_exact": 0,
            "autoaccept_wrong": 0,
        }
        for bbox_id, truth in sorted(truths.items(), key=lambda item: item[1]["sample_id"]):
            row = by_bbox.get(bbox_id, {})
            try:
                predicted_ids = json.loads(row.get("note_ids") or "[]")
            except json.JSONDecodeError:
                predicted_ids = []
            if not predicted_ids:
                result = "unresolved"
            elif set(predicted_ids) & set(truth["expected_note_ids"]):
                result = "exact"
            else:
                result = "wrong"
            counts[result] += 1
            is_autoaccept = (
                row.get("status") == "inferred"
                and float(row.get("confidence") or 0) >= 0.70
            )
            if is_autoaccept and result == "exact":
                counts["autoaccept_exact"] += 1
            elif is_autoaccept and result == "wrong":
                counts["autoaccept_wrong"] += 1
            predictions.append({
                "profile": profile_name,
                "sample_id": truth["sample_id"],
                "bbox_id": bbox_id,
                "result": result,
                "expected_note_ids": json.dumps(truth["expected_note_ids"]),
                "predicted_note_ids": json.dumps(predicted_ids),
                "predicted_staff": row.get("xml_staff", ""),
                "predicted_measure": row.get("xml_measure", ""),
                "predicted_pitch": row.get("xml_symbol", ""),
                "confidence": row.get("confidence", ""),
                "status": row.get("status", ""),
            })
        summaries.append({"profile": profile_name, **profile, **counts, "total": len(truths)})
        emit_progress("fingering-profile-summary", len(summaries), len(PROFILES), f"{profile_name} exact={counts['exact']} wrong={counts['wrong']} unresolved={counts['unresolved']} autoaccept_wrong={counts['autoaccept_wrong']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "profile_predictions.csv", list(predictions[0]), predictions)
    atomic_write_csv(output_dir / "profile_summary.csv", list(summaries[0]), summaries)
    report = {"pipeline_version": PIPELINE_VERSION, "evaluated_regular_fingerings": len(truths), "profiles": summaries}
    atomic_write_json(output_dir / "evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args.manifest, args.audit_csv, args.feedback_json, args.repeat_dir, args.output_dir, resume=args.resume)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
