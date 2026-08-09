"""Run BPSD/MusicXML/YOLO alignment across every available score page."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from bps_xml_alignment import run_alignment


PAGE_PATTERN = re.compile(r"^(?P<score>.+)-(?P<page>\d+)$")
SUMMARY_FIELDS = [
    "score",
    "page",
    "status",
    "image",
    "yolo",
    "detected_systems",
    "input_boxes",
    "output_rows",
    "matched",
    "inferred",
    "review",
    "unresolved",
    "other_statuses",
    "csv",
    "report",
    "error",
]


def discover_pages(
    dataset_dir: Path,
    score: str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Pair score images and YOLO labels by the numeric page suffix."""

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    images: dict[int, Path] = {}
    for path in sorted(image_dir.glob(f"{score}-*")):
        if path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
            continue
        match = PAGE_PATTERN.match(path.stem)
        if match and match.group("score") == score:
            images[int(match.group("page"))] = path
    labels: dict[int, Path] = {}
    for path in sorted(label_dir.glob(f"{score}-*.txt")):
        match = PAGE_PATTERN.match(path.stem)
        if match and match.group("score") == score:
            labels[int(match.group("page"))] = path

    selected = sorted(set(images) | set(labels))
    if start_page is not None:
        selected = [page for page in selected if page >= start_page]
    if end_page is not None:
        selected = [page for page in selected if page <= end_page]

    ready = []
    missing = []
    for page in selected:
        image = images.get(page)
        label = labels.get(page)
        if image is None or label is None:
            missing.append(
                {
                    "page": page,
                    "missing_image": image is None,
                    "missing_yolo": label is None,
                    "image": str(image or ""),
                    "yolo": str(label or ""),
                }
            )
        else:
            ready.append({"page": page, "image": image, "yolo": label})
    return ready, missing


def _status_counts(report: dict) -> dict[str, int]:
    output: dict[str, int] = {}
    for statuses in report.get("counts", {}).values():
        for status, count in statuses.items():
            output[status] = output.get(status, 0) + int(count)
    return output


def run_batch(
    *,
    dataset_dir: Path,
    score: str,
    xml_path: Path,
    bps_notes_path: Path,
    notes_json_path: Path,
    output_dir: Path,
    start_page: int | None = None,
    end_page: int | None = None,
    infer_fingerings: bool = False,
    fail_fast: bool = False,
) -> dict:
    pages, missing = discover_pages(
        dataset_dir,
        score,
        start_page=start_page,
        end_page=end_page,
    )
    if not pages and not missing:
        raise ValueError(f"No image or label files found for score {score}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in pages:
        page = item["page"]
        page_dir = output_dir / "pages" / f"{page:02d}"
        report_path = page_dir / "qa" / f"{item['image'].stem}_report.json"
        try:
            report = run_alignment(
                image_path=item["image"],
                yolo_path=item["yolo"],
                xml_path=xml_path,
                bps_note_path=bps_notes_path,
                output_dir=page_dir,
                page_number=page,
                infer_fingerings=infer_fingerings,
                notes_json_path=notes_json_path,
                include_all_symbols=True,
            )
            counts = _status_counts(report)
            known = {"matched", "inferred", "review", "unresolved"}
            other = {key: value for key, value in counts.items() if key not in known}
            rows.append(
                {
                    "score": score,
                    "page": page,
                    "status": "completed",
                    "image": str(item["image"]),
                    "yolo": str(item["yolo"]),
                    "detected_systems": report["detected_systems"],
                    "input_boxes": report["target_yolo_boxes"],
                    "output_rows": report["output_rows"],
                    "matched": counts.get("matched", 0),
                    "inferred": counts.get("inferred", 0),
                    "review": counts.get("review", 0),
                    "unresolved": counts.get("unresolved", 0),
                    "other_statuses": json.dumps(other, ensure_ascii=False),
                    "csv": report["outputs"]["csv"],
                    "report": str(report_path),
                    "error": "",
                }
            )
        except Exception as error:  # Batch execution must preserve later pages.
            rows.append(
                {
                    "score": score,
                    "page": page,
                    "status": "failed",
                    "image": str(item["image"]),
                    "yolo": str(item["yolo"]),
                    "detected_systems": "",
                    "input_boxes": "",
                    "output_rows": "",
                    "matched": "",
                    "inferred": "",
                    "review": "",
                    "unresolved": "",
                    "other_statuses": "",
                    "csv": "",
                    "report": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if fail_fast:
                break

    for item in missing:
        rows.append(
            {
                "score": score,
                "page": item["page"],
                "status": "missing_input",
                "image": item["image"],
                "yolo": item["yolo"],
                "detected_systems": "",
                "input_boxes": "",
                "output_rows": "",
                "matched": "",
                "inferred": "",
                "review": "",
                "unresolved": "",
                "other_statuses": "",
                "csv": "",
                "report": "",
                "error": "missing image" if item["missing_image"] else "missing YOLO label",
            }
        )
    rows.sort(key=lambda row: int(row["page"]))

    summary_csv = output_dir / "batch_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "score": score,
        "dataset_dir": str(dataset_dir),
        "xml": str(xml_path),
        "bps_notes": str(bps_notes_path),
        "notes_json": str(notes_json_path),
        "requested_range": {"start_page": start_page, "end_page": end_page},
        "infer_fingerings": infer_fingerings,
        "completed_pages": sum(row["status"] == "completed" for row in rows),
        "failed_pages": sum(row["status"] == "failed" for row in rows),
        "missing_input_pages": sum(row["status"] == "missing_input" for row in rows),
        "pages": rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_json": str(output_dir / "batch_summary.json"),
        },
    }
    summary_json = output_dir / "batch_summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run alignment for every image/TXT page pair in one score."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--score", required=True, help="Filename prefix without page number")
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--bps-notes", type=Path, required=True)
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--infer-fingerings", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    summary = run_batch(
        dataset_dir=args.dataset_dir,
        score=args.score,
        xml_path=args.xml,
        bps_notes_path=args.bps_notes,
        notes_json_path=args.notes_json,
        output_dir=args.output_dir,
        start_page=args.start_page,
        end_page=args.end_page,
        infer_fingerings=args.infer_fingerings,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
