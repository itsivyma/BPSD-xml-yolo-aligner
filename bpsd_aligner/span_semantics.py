"""Shared note-endpoint semantics for MusicXML spans.

Slurs attach to a marked notehead but semantically connect complete endpoint
chords.  Ties remain pitch-specific.  Keeping this rule here prevents the
single-page matcher and whole-score/cross-page pipeline from drifting apart.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable


def index_chord_members(
    events: Iterable[dict],
    key_for_event: Callable[[dict], Hashable | None],
) -> dict[Hashable, list[dict]]:
    """Index chord members without changing their source/XML order."""

    indexed: dict[Hashable, list[dict]] = {}
    for event in events:
        key = key_for_event(event)
        if key is not None:
            indexed.setdefault(key, []).append(event)
    return indexed


def endpoint_note_ids(
    start: dict,
    end: dict,
    *,
    start_members: Iterable[dict] | None = None,
    end_members: Iterable[dict] | None = None,
    expand_chords: bool = False,
) -> list[object]:
    """Return unique endpoint IDs using slur or tie semantics.

    ``expand_chords=True`` includes every note in both endpoint chords.  The
    exact marked noteheads still belong in the separate ``start_note`` and
    ``end_note`` fields.  With ``expand_chords=False`` only those two marked
    endpoints are returned, which is the required behavior for ties.
    """

    def note_id(event: dict, endpoint: str) -> object | None:
        keys = (
            ("note_id", "start_note", "end_note")
            if endpoint == "start"
            else ("note_id", "end_note", "start_note")
        )
        for key in keys:
            value = event.get(key)
            if value is not None and str(value).strip() not in {"", "NA"}:
                return value
        return None

    groups = (
        (
            list(start_members or [start]),
            list(end_members or [end]),
        )
        if expand_chords
        else ([start], [end])
    )
    result: list[object] = []
    for endpoint, group in zip(("start", "end"), groups, strict=True):
        for event in group:
            value = note_id(event, endpoint)
            if value is not None and value not in result:
                result.append(value)
    return result
