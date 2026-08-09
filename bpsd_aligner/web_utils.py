"""Dependency-light helpers used by the Streamlit app and tests."""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import PurePath


def read_csv_bytes(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "scores": len({row.get("score_id", "") for row in rows if row.get("score_id")}),
        "origins": dict(Counter(row.get("row_origin", "unknown") or "unknown" for row in rows)),
        "statuses": dict(Counter(row.get("combined_status", "unknown") or "unknown" for row in rows)),
    }


def sanitize_path_columns(data: bytes) -> bytes:
    fields, rows = read_csv_bytes(data)
    path_fields = [field for field in fields if field == "source_xml_path" or field.endswith("_path")]
    for row in rows:
        for field in path_fields:
            value = row.get(field, "")
            row[field] = PurePath(value).name if value else ""
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")
