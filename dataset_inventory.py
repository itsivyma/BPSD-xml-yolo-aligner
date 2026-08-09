"""Validate a YOLO/BPSD corpus and write a reproducible alignment manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


PAGE_RE = re.compile(r"^(?P<score>.+)-(?P<page>\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xml_stats(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    measures = sum(
        element.tag.split("}")[-1] == "measure" for element in root.iter()
    )
    pages = sum(
        element.tag.split("}")[-1] == "print"
        and element.attrib.get("new-page") == "yes"
        for element in root.iter()
    )
    return max(1, pages), measures


def _repeat_stats(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    repeat_marks = sum(
        element.tag.split("}")[-1] == "repeat" for element in root.iter()
    )
    ending_marks = sum(
        element.tag.split("}")[-1] == "ending" for element in root.iter()
    )
    return repeat_marks, ending_marks


def _bps_note_stats(path: Path) -> tuple[list[str], int]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)
        return header, sum(1 for _ in reader)


def build_inventory(xia_dir: Path, bpsd_dir: Path, output_dir: Path) -> dict:
    images_dir = xia_dir / "images"
    labels_dir = xia_dir / "labels"
    notes_json = xia_dir / "notes.json"
    categories = {
        int(row["id"]): row["name"]
        for row in json.loads(notes_json.read_text(encoding="utf-8"))["categories"]
    }
    classes = (xia_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    category_mismatches = [
        {"class_id": index, "classes_txt": name, "notes_json": categories.get(index)}
        for index, name in enumerate(classes)
        if categories.get(index) != name
    ]

    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    labels = {path.stem: path for path in labels_dir.glob("*.txt")}
    all_stems = sorted(set(images) | set(labels))
    score_pages: dict[str, list[int]] = defaultdict(list)
    parsed_stems = {}
    for stem in all_stems:
        match = PAGE_RE.match(stem)
        if not match:
            continue
        score = match.group("score")
        page = int(match.group("page"))
        parsed_stems[stem] = (score, page)
        score_pages[score].append(page)

    xml_dir = bpsd_dir / "0_RawData" / "score_xml_repetitions"
    unfolded_xml_dir = bpsd_dir / "0_RawData" / "score_xml_unfolded"
    sibelius_dir = bpsd_dir / "0_RawData" / "score_sibelius_repetitions"
    note_dir = bpsd_dir / "2_Annotations" / "ann_score_note"
    scan_pdf_dir = bpsd_dir / "0_RawData" / "score_pdf_scan"
    repetition_pdf_dir = bpsd_dir / "0_RawData" / "score_pdf_repetitions"
    score_info = {}
    for score, pages in sorted(score_pages.items()):
        xml_path = xml_dir / f"{score}.xml"
        unfolded_xml_path = unfolded_xml_dir / f"{score}.xml"
        sibelius_path = sibelius_dir / f"{score}.sib"
        bps_notes_path = note_dir / f"{score}.csv"
        scan_pdf_path = scan_pdf_dir / f"{score}.pdf"
        repetition_pdf_path = repetition_pdf_dir / f"{score}.pdf"
        xml_pages, xml_measures = _xml_stats(xml_path)
        _, unfolded_xml_measures = _xml_stats(unfolded_xml_path)
        repeat_marks, ending_marks = _repeat_stats(xml_path)
        bps_header, bps_note_rows = _bps_note_stats(bps_notes_path)
        scan_pdf_pages = len(PdfReader(scan_pdf_path).pages)
        repetition_pdf_pages = len(PdfReader(repetition_pdf_path).pages)
        image_pages = len(set(pages))
        direct = image_pages == xml_pages == repetition_pdf_pages
        score_info[score] = {
            "image_pages": image_pages,
            "xml_pages": xml_pages,
            "scan_pdf_pages": scan_pdf_pages,
            "repetition_pdf_pages": repetition_pdf_pages,
            "xml_measures": xml_measures,
            "unfolded_xml_measures": unfolded_xml_measures,
            "repeat_marks": repeat_marks,
            "ending_marks": ending_marks,
            "bps_note_rows": bps_note_rows,
            "bps_note_header": bps_header,
            "page_mapping_status": (
                "direct_page_candidate" if direct else "requires_system_alignment"
            ),
            "xml_path": str(xml_path),
            "unfolded_xml_path": str(unfolded_xml_path),
            "sibelius_path": str(sibelius_path),
            "bps_notes_path": str(bps_notes_path),
            "scan_pdf_path": str(scan_pdf_path),
            "repetition_pdf_path": str(repetition_pdf_path),
        }

    manifest = []
    invalid_yolo_rows = []
    class_counts: Counter[tuple[int, str]] = Counter()
    score_class_counts: dict[str, Counter[tuple[int, str]]] = defaultdict(Counter)
    for stem in all_stems:
        parsed = parsed_stems.get(stem)
        if parsed is None:
            continue
        score, page = parsed
        image_path = images.get(stem)
        label_path = labels.get(stem)
        yolo_count = 0
        page_counts: Counter[str] = Counter()
        if label_path is not None:
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                parts = line.split()
                error = ""
                if len(parts) != 5:
                    error = f"expected 5 fields; got {len(parts)}"
                else:
                    try:
                        class_id = int(parts[0])
                        coordinates = [float(value) for value in parts[1:]]
                        if class_id not in categories:
                            error = f"unknown class_id {class_id}"
                        elif any(value < 0 or value > 1 for value in coordinates):
                            error = "coordinate outside [0,1]"
                        else:
                            class_name = categories[class_id]
                            class_counts[(class_id, class_name)] += 1
                            score_class_counts[score][(class_id, class_name)] += 1
                            page_counts[class_name] += 1
                    except ValueError as exception:
                        error = str(exception)
                if error:
                    invalid_yolo_rows.append(
                        {
                            "page_id": stem,
                            "yolo_line": line_number,
                            "error": error,
                            "raw": line,
                        }
                    )
                yolo_count += 1

        image_width = image_height = ""
        if image_path is not None:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        info = score_info[score]
        manifest.append(
            {
                "score_id": score,
                "page": page,
                "page_id": stem,
                "image_path": str(image_path or ""),
                "yolo_path": str(label_path or ""),
                "xml_path": info["xml_path"],
                "unfolded_xml_path": info["unfolded_xml_path"],
                "sibelius_path": info["sibelius_path"],
                "bps_notes_path": info["bps_notes_path"],
                "scan_pdf_path": info["scan_pdf_path"],
                "repetition_pdf_path": info["repetition_pdf_path"],
                "notes_json_path": str(notes_json),
                "image_width": image_width,
                "image_height": image_height,
                "yolo_rows": yolo_count,
                "class_counts_json": json.dumps(
                    dict(sorted(page_counts.items())), ensure_ascii=False
                ),
                "image_sha256": _sha256(image_path) if image_path else "",
                "yolo_sha256": _sha256(label_path) if label_path else "",
                "page_mapping_status": info["page_mapping_status"],
                "input_status": (
                    "ready"
                    if image_path is not None and label_path is not None
                    else "missing_input"
                ),
            }
        )
    manifest.sort(key=lambda row: (row["score_id"], int(row["page"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "page_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    class_path = output_dir / "class_counts.csv"
    score_names = sorted(score_info)
    with class_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["class_id", "class", "total", *score_names]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (class_id, class_name), total in sorted(class_counts.items()):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class": class_name,
                    "total": total,
                    **{
                        score: score_class_counts[score][(class_id, class_name)]
                        for score in score_names
                    },
                }
            )

    summary = {
        "xia_dir": str(xia_dir),
        "bpsd_dir": str(bpsd_dir),
        "images": len(images),
        "labels": len(labels),
        "matched_image_label_pages": len(set(images) & set(labels)),
        "total_yolo_boxes": sum(class_counts.values()),
        "defined_classes": len(categories),
        "used_classes": len(class_counts),
        "image_without_label": sorted(set(images) - set(labels)),
        "label_without_image": sorted(set(labels) - set(images)),
        "invalid_yolo_rows": invalid_yolo_rows,
        "category_mismatches": category_mismatches,
        "scores": score_info,
        "outputs": {
            "page_manifest": str(manifest_path),
            "class_counts": str(class_path),
            "summary": str(output_dir / "dataset_summary.json"),
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xia-dir", type=Path, required=True)
    parser.add_argument("--bpsd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_inventory(args.xia_dir, args.bpsd_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
