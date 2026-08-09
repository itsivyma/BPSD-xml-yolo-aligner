"""Small, dependency-free helpers for resumable pipeline stages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image


def stable_digest(payload: object) -> str:
    """Return a deterministic digest for JSON-compatible checkpoint inputs."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_signature(path: Path) -> dict[str, object]:
    """Capture inexpensive file identity data without hashing large inputs."""

    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding=encoding)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_png(image: Image.Image, path: Path, *, optimize: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        image.save(temporary, format="PNG", optimize=optimize)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def emit_progress(
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    print(f"[{stage}] {current}/{total} {message}", flush=True)
