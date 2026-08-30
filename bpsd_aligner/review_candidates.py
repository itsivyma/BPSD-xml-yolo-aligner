"""Normalize large per-row Review note choices into reusable side tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from bpsd_aligner.pipeline_checkpoint import atomic_write_csv


CANDIDATE_FIELDS = ("candidate_id", "candidate_json")
CANDIDATE_SET_FIELDS = ("candidate_set_id", "candidate_ids_json")


def _candidate_identity(page_id: str, candidate: dict) -> str:
    sequence = str(candidate.get("xml_note_sequence", "")).strip()
    if sequence:
        identity = ["sequence", page_id, sequence]
    else:
        identity = [
            "geometry",
            page_id,
            str(candidate.get("note_id", "")),
            str(candidate.get("xml_measure", "")),
            str(candidate.get("staff", "")),
            str(candidate.get("pitch", "")),
            str(candidate.get("x_px", "")),
            str(candidate.get("y_px", "")),
        ]
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"{page_id}:N{digest}"


def normalize_review_candidates(
    rows: list[dict], *, page_id: str, output_dir: Path
) -> tuple[Path, Path]:
    """Write unique candidates and small row-to-candidate-set associations."""

    candidates_by_id: dict[str, dict] = {}
    candidate_sets: list[dict] = []
    for row in rows:
        try:
            raw = json.loads(str(row.get("review_note_candidates_json", "") or "[]"))
        except json.JSONDecodeError:
            raw = []
        candidate_ids = []
        if isinstance(raw, list):
            for candidate in raw:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = _candidate_identity(page_id, candidate)
                candidate_ids.append(candidate_id)
                if candidate_id not in candidates_by_id:
                    normalized = {
                        key: value
                        for key, value in candidate.items()
                        if key != "distance_px"
                    }
                    normalized.setdefault("page_id", page_id)
                    candidates_by_id[candidate_id] = normalized
        set_id = f"{page_id}:Y{row.get('txt_line', len(candidate_sets) + 1)}"
        row["review_candidate_set_id"] = set_id
        row["review_note_candidates_json"] = ""
        candidate_sets.append(
            {
                "candidate_set_id": set_id,
                "candidate_ids_json": json.dumps(
                    candidate_ids, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / f"{page_id}_review_note_candidates.csv"
    sets_path = output_dir / f"{page_id}_review_candidate_sets.csv"
    atomic_write_csv(
        candidates_path,
        CANDIDATE_FIELDS,
        [
            {
                "candidate_id": candidate_id,
                "candidate_json": json.dumps(
                    candidate, ensure_ascii=False, separators=(",", ":")
                ),
            }
            for candidate_id, candidate in sorted(candidates_by_id.items())
        ],
    )
    atomic_write_csv(sets_path, CANDIDATE_SET_FIELDS, candidate_sets)
    return candidates_path, sets_path


def hydrate_review_candidates(
    rows: list[dict], candidates_path: Path, sets_path: Path
) -> list[dict]:
    """Restore legacy in-memory JSON for Review UI compatibility."""

    with candidates_path.open(newline="", encoding="utf-8-sig") as file:
        candidates = {
            row["candidate_id"]: json.loads(row["candidate_json"])
            for row in csv.DictReader(file)
        }
    with sets_path.open(newline="", encoding="utf-8-sig") as file:
        candidate_sets = {
            row["candidate_set_id"]: json.loads(row["candidate_ids_json"])
            for row in csv.DictReader(file)
        }
    for row in rows:
        candidate_ids = candidate_sets.get(str(row.get("review_candidate_set_id", "")), [])
        row["review_note_candidates_json"] = json.dumps(
            [candidates[item] for item in candidate_ids if item in candidates],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return rows
