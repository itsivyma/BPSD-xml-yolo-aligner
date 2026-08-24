"""Shared secure, namespace-independent MusicXML primitives."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from defusedxml import ElementTree as SafeET


def parse_musicxml(path: Path) -> ET.Element:
    """Parse MusicXML while allowing its DOCTYPE but blocking external data."""

    return SafeET.parse(
        path,
        forbid_dtd=False,
        forbid_entities=True,
        forbid_external=True,
    ).getroot()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((item for item in element if local_name(item.tag) == name), None)


def children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [item for item in element if local_name(item.tag) == name]


def child_text(element: ET.Element | None, name: str, default: str = "") -> str:
    item = child(element, name)
    return (item.text or default) if item is not None else default
