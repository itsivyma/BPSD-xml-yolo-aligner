"""Propose conservative auto-accepts and group the remaining review queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_review_queue import QUEUE_FIELDS
from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.3.0-review-reduction"
DEFAULT_THRESHOLD = 0.95
AUTO_ACCEPT_BLOCKED_CLASSES = {f"fingering{digit}" for digit in range(1, 6)}
PROPOSAL_EXTRA_FIELDS = [
    "proposal_id", "proposal_status", "proposed_combined_status",
    "eligibility_threshold", "unique_xml_event_id", "eligibility_evidence_json",
    "audit_required", "audit_sampled", "audit_decision", "audited_by",
    "audited_at", "audit_comment",
]
PROPOSAL_FIELDS = PROPOSAL_EXTRA_FIELDS + QUEUE_FIELDS
AUDIT_FIELDS = ["sample_id", "stratum_id", "sample_position"] + PROPOSAL_FIELDS
MANUAL_FIELDS = ["group_id", "group_rank", "group_member_index"] + QUEUE_FIELDS
GROUP_FIELDS = [
    "group_rank", "group_id", "priority_tier", "score_id",
    "primary_xml_event_id", "group_size", "bbox_ids_json", "classes_json",
    "statuses_json", "min_confidence", "max_confidence", "batch_ids_json",
    "review_pngs_json", "review_csvs_json", "group_review_status",
    "group_decision", "reviewer", "reviewed_at", "comment",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_event_ids(row: dict) -> list[str]:
    try:
        values = json.loads(row.get("xml_event_ids_json") or "[]")
    except json.JSONDecodeError:
        return []
    return list(dict.fromkeys(str(value) for value in values if value))


def _eligible(row: dict, master: dict, threshold: float) -> bool:
    try:
        confidence = float(row.get("confidence") or 0)
    except ValueError:
        return False
    return (
        row.get("source_status") == "candidate"
        and row.get("class") not in AUTO_ACCEPT_BLOCKED_CLASSES
        and confidence >= threshold
        and len(_unique_event_ids(row)) == 1
        and row.get("asset_status") == "ready"
        and master.get("page_mapping_status") == "direct"
        and master.get("movement_scope_status") == "in_bpsd_scope"
        and not master.get("error_code")
    )


def split_score_rows(
    queue_rows: list[dict], master_by_bbox: dict[str, dict], threshold: float
) -> tuple[list[dict], list[dict]]:
    proposals, manual = [], []
    for row in queue_rows:
        master = master_by_bbox[row["bbox_id"]]
        if _eligible(row, master, threshold):
            event_id = _unique_event_ids(row)[0]
            proposal = {
                **row,
                "proposal_id": "",
                "proposal_status": "proposed_pending_audit",
                "proposed_combined_status": "auto_accepted_pending_audit",
                "eligibility_threshold": f"{threshold:.3f}",
                "unique_xml_event_id": event_id,
                "eligibility_evidence_json": _json(
                    {
                        "confidence": float(row["confidence"]),
                        "unique_xml_event_id": event_id,
                        "page_mapping_status": master["page_mapping_status"],
                        "movement_scope_status": master["movement_scope_status"],
                        "error_code": master.get("error_code", ""),
                        "asset_status": row["asset_status"],
                    }
                ),
                "audit_required": "true",
                "audit_sampled": "false",
                "audit_decision": "",
                "audited_by": "",
                "audited_at": "",
                "audit_comment": "",
            }
            proposals.append(proposal)
        else:
            manual.append(dict(row))
    return proposals, manual


def select_audit_sample(proposals: list[dict], per_stratum: int = 2) -> list[dict]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in proposals:
        strata[(row["score_id"], row["class"])].append(row)
    sample = []
    for (score_id, class_name), rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: (float(row["confidence"]), row["bbox_id"]))
        selected = []
        if ordered:
            selected.append(("lowest_confidence", ordered[0]))
        if per_stratum > 1 and len(ordered) > 1:
            selected.append(("highest_confidence", ordered[-1]))
        stratum_id = f"{score_id}:{class_name}"
        for position, (reason, row) in enumerate(selected[:per_stratum], start=1):
            row["audit_sampled"] = "true"
            sample.append(
                {
                    **row,
                    "sample_id": "",
                    "stratum_id": stratum_id,
                    "sample_position": f"{position}:{reason}",
                }
            )
    return sample


def build_manual_groups(
    manual_rows: list[dict], primary_event_by_bbox: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in manual_rows:
        event_id = primary_event_by_bbox.get(row["bbox_id"], row["bbox_id"])
        grouped[(row["score_id"], event_id)].append(row)
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            0 if any(row["source_status"] == "ambiguous" for row in item[1]) else 1,
            min(float(row["confidence"] or 0) for row in item[1]),
            item[0],
        ),
    )
    groups, members = [], []
    for rank, ((score_id, event_id), rows) in enumerate(ordered, start=1):
        rows = sorted(
            rows,
            key=lambda row: (
                0 if row["source_status"] == "ambiguous" else 1,
                float(row["confidence"] or 0),
                row["bbox_id"],
            ),
        )
        group_id = f"G{rank:05d}"
        confidences = [float(row["confidence"] or 0) for row in rows]
        groups.append(
            {
                "group_rank": rank,
                "group_id": group_id,
                "priority_tier": (
                    "P1_ambiguous" if any(row["source_status"] == "ambiguous" for row in rows)
                    else "P2_candidate"
                ),
                "score_id": score_id,
                "primary_xml_event_id": event_id,
                "group_size": len(rows),
                "bbox_ids_json": _json([row["bbox_id"] for row in rows]),
                "classes_json": _json(sorted({row["class"] for row in rows})),
                "statuses_json": _json(sorted({row["source_status"] for row in rows})),
                "min_confidence": f"{min(confidences):.3f}",
                "max_confidence": f"{max(confidences):.3f}",
                "batch_ids_json": _json(sorted({row["batch_id"] for row in rows})),
                "review_pngs_json": _json(sorted({row["review_png"] for row in rows})),
                "review_csvs_json": _json(sorted({row["review_csv"] for row in rows})),
                "group_review_status": "not_started",
                "group_decision": "",
                "reviewer": "",
                "reviewed_at": "",
                "comment": "",
            }
        )
        for member_index, row in enumerate(rows, start=1):
            members.append(
                {
                    "group_id": group_id,
                    "group_rank": rank,
                    "group_member_index": member_index,
                    **row,
                }
            )
    return groups, members


def reduce_review_queue(
    review_queue_path: Path,
    combined_master_path: Path,
    alignment_links_path: Path,
    output_dir: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    resume: bool = False,
) -> dict:
    _queue_fields, queue = _read_csv(review_queue_path)
    _master_fields, combined = _read_csv(combined_master_path)
    _link_fields, links = _read_csv(alignment_links_path)
    master_by_bbox = {
        row["bbox_id"]: row for row in combined if row.get("row_origin") == "yolo"
    }
    primary_event_by_bbox = {
        row["bbox_id"]: row["xml_event_id"]
        for row in links if row.get("is_primary") == "true"
    }
    queue_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in queue:
        queue_by_score[row["score_id"]].append(row)
    score_ids = sorted(queue_by_score)
    proposals, manual_rows = [], []
    score_reports = []
    emit_progress("review-reduction", 0, len(score_ids), f"starting (resume={resume})")
    for index, score_id in enumerate(score_ids, start=1):
        score_dir = output_dir / "per_scores" / score_id
        proposal_path = score_dir / "auto_accept_proposal.csv"
        manual_path = score_dir / "manual_review_queue.csv"
        checkpoint_path = output_dir / "checkpoints" / f"{score_id}.json"
        fingerprint = stable_digest(
            {
                "version": PIPELINE_VERSION,
                "threshold": threshold,
                "review_queue": path_signature(review_queue_path),
                "combined_master": path_signature(combined_master_path),
                "alignment_links": path_signature(alignment_links_path),
                "score_id": score_id,
            }
        )
        reused = False
        if resume and proposal_path.is_file() and manual_path.is_file() and checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                _proposal_fields, score_proposals = _read_csv(proposal_path)
                _manual_fields, score_manual = _read_csv(manual_path)
                reused = (
                    checkpoint.get("fingerprint") == fingerprint
                    and checkpoint.get("proposal_rows") == len(score_proposals)
                    and checkpoint.get("manual_rows") == len(score_manual)
                )
            except (OSError, csv.Error, json.JSONDecodeError):
                reused = False
        if not reused:
            score_proposals, score_manual = split_score_rows(
                queue_by_score[score_id], master_by_bbox, threshold
            )
            atomic_write_csv(proposal_path, PROPOSAL_FIELDS, score_proposals)
            atomic_write_csv(manual_path, QUEUE_FIELDS, score_manual)
            atomic_write_json(
                checkpoint_path,
                {
                    "score_id": score_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "fingerprint": fingerprint,
                    "proposal_rows": len(score_proposals),
                    "manual_rows": len(score_manual),
                },
            )
        proposals.extend(score_proposals)
        manual_rows.extend(score_manual)
        score_reports.append(
            {
                "score_id": score_id,
                "proposal_rows": len(score_proposals),
                "manual_rows": len(score_manual),
                "resumed": reused,
            }
        )
        emit_progress(
            "review-reduction", index, len(score_ids),
            f"{score_id} {'resumed' if reused else 'built'} proposals={len(score_proposals)} manual={len(score_manual)}",
        )

    proposals.sort(key=lambda row: (row["score_id"], row["class"], float(row["confidence"]), row["bbox_id"]))
    for index, row in enumerate(proposals, start=1):
        row["proposal_id"] = f"A{index:04d}"
    audit_sample = select_audit_sample(proposals)
    audit_sample.sort(key=lambda row: (row["stratum_id"], row["sample_position"]))
    for index, row in enumerate(audit_sample, start=1):
        row["sample_id"] = f"S{index:03d}"
    groups, manual_members = build_manual_groups(manual_rows, primary_event_by_bbox)

    # Rewrite score outputs with stable proposal IDs assigned.
    proposals_by_score: dict[str, list[dict]] = defaultdict(list)
    manual_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in proposals:
        proposals_by_score[row["score_id"]].append(row)
    for row in manual_members:
        manual_by_score[row["score_id"]].append(row)
    for score_id in score_ids:
        atomic_write_csv(
            output_dir / "per_scores" / score_id / "auto_accept_proposal.csv",
            PROPOSAL_FIELDS,
            proposals_by_score[score_id],
        )
        atomic_write_csv(
            output_dir / "per_scores" / score_id / "manual_review_queue.csv",
            MANUAL_FIELDS,
            manual_by_score[score_id],
        )
    atomic_write_csv(output_dir / "auto_accept_proposal.csv", PROPOSAL_FIELDS, proposals)
    atomic_write_csv(output_dir / "auto_accept_audit_sample.csv", AUDIT_FIELDS, audit_sample)
    atomic_write_csv(output_dir / "manual_review_groups.csv", GROUP_FIELDS, groups)
    atomic_write_csv(output_dir / "manual_review_queue.csv", MANUAL_FIELDS, manual_members)

    source_ids = {row["bbox_id"] for row in queue}
    proposal_ids = {row["bbox_id"] for row in proposals}
    manual_ids = {row["bbox_id"] for row in manual_members}
    sample_ids = {row["bbox_id"] for row in audit_sample}
    errors = []
    if proposal_ids & manual_ids:
        errors.append("proposal and manual review rows overlap")
    if proposal_ids | manual_ids != source_ids:
        errors.append("proposal and manual review rows do not cover source queue")
    if any(row["source_status"] == "ambiguous" for row in proposals):
        errors.append("ambiguous row included in auto-accept proposal")
    if any(row["class"] in AUTO_ACCEPT_BLOCKED_CLASSES for row in proposals):
        errors.append("uncalibrated fingering row included in auto-accept proposal")
    if any(not _eligible(row, master_by_bbox[row["bbox_id"]], threshold) for row in proposals):
        errors.append("ineligible row included in auto-accept proposal")
    if not sample_ids <= proposal_ids:
        errors.append("audit sample is not a subset of proposals")
    if len({row["proposal_id"] for row in proposals}) != len(proposals):
        errors.append("duplicate proposal_id")
    proposal_strata = Counter((row["score_id"], row["class"]) for row in proposals)
    sample_strata = Counter((row["score_id"], row["class"]) for row in audit_sample)
    if any(
        sample_strata[stratum] != min(2, count)
        for stratum, count in proposal_strata.items()
    ):
        errors.append("audit sample does not cover every proposal stratum")
    sampled_flags = {
        row["bbox_id"] for row in proposals if row["audit_sampled"] == "true"
    }
    if sampled_flags != sample_ids:
        errors.append("proposal audit_sampled flags differ from audit sample")
    if len({row["group_id"] for row in groups}) != len(groups):
        errors.append("duplicate manual review group_id")
    if len(manual_members) != len(manual_ids):
        errors.append("duplicate bbox_id in manual review queue")
    group_member_ids = {
        bbox_id
        for group in groups
        for bbox_id in json.loads(group["bbox_ids_json"])
    }
    if group_member_ids != manual_ids:
        errors.append("manual review groups do not cover manual queue")
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "eligibility_threshold": threshold,
        "source_queue_rows": len(queue),
        "auto_accept_proposal_rows": len(proposals),
        "audit_sample_rows": len(audit_sample),
        "audit_strata": len({row["stratum_id"] for row in audit_sample}),
        "manual_review_rows": len(manual_members),
        "manual_review_groups": len(groups),
        "proposal_class_counts": dict(Counter(row["class"] for row in proposals)),
        "auto_accept_blocked_classes": sorted(AUTO_ACCEPT_BLOCKED_CLASSES),
        "blocked_class_source_rows": sum(
            row.get("class") in AUTO_ACCEPT_BLOCKED_CLASSES for row in queue
        ),
        "manual_status_counts": dict(Counter(row["source_status"] for row in manual_members)),
        "score_reports": score_reports,
        "master_modified": False,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "auto_accept_proposal": str(output_dir / "auto_accept_proposal.csv"),
            "auto_accept_audit_sample": str(output_dir / "auto_accept_audit_sample.csv"),
            "manual_review_groups": str(output_dir / "manual_review_groups.csv"),
            "manual_review_queue": str(output_dir / "manual_review_queue.csv"),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("review-reduction-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--combined-master", type=Path, required=True)
    parser.add_argument("--alignment-links", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = reduce_review_queue(
        args.review_queue,
        args.combined_master,
        args.alignment_links,
        args.output_dir,
        threshold=args.threshold,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
