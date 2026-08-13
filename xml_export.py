"""Export lossless MusicXML nodes and BPS-OMR-oriented XML events to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

from defusedxml.ElementTree import DefusedXMLParser

from pipeline_checkpoint import (
    atomic_write_csv,
    atomic_write_json,
    emit_progress,
    path_signature,
    stable_digest,
)


PIPELINE_VERSION = "0.3.0"
BPS_FIELDS = [
    "class_id", "x", "y", "w", "h", "class", "musical_time",
    "start_meas", "end_meas", "start_note", "end_note",
    "connected_note", "stem_dir",
]
NODE_FIELDS = [
    "score_id", "xml_node_id", "parent_node_id", "depth", "sibling_index",
    "xml_xpath", "tag", "namespace", "attributes_json", "text", "tail",
    "child_count", "child_node_ids_json", "part_id", "measure_number",
    "measure_index", "page", "system", "source_xml_path", "source_sha256",
]
EVENT_FIELDS = BPS_FIELDS + [
    "record_id", "row_origin", "score_id", "xml_event_id", "xml_node_id",
    "event_type", "event_subtype", "part_id", "page", "system",
    "xml_measure", "xml_measure_index", "staff", "voice",
    "onset_divisions", "duration_divisions", "divisions", "beat_position",
    "duration_quarterLength", "written_start_meas", "written_end_meas",
    "pitch", "pitchName", "pitch_step", "pitch_alter", "pitch_octave",
    "midi_pitch", "is_rest", "is_chord", "is_grace", "chord_id",
    "note_type", "dot_count", "beam_json", "accidental", "dynamic",
    "articulation_json", "fermata_json", "slur_json", "tie_json",
    "tuplet_json", "ornament_json", "time_modification_json",
    "direction_json", "clef_json", "key_signature_json", "time_signature",
    "repeat_json", "event_payload_json", "anchor_xml_event_ids_json",
    "xml_attributes_json", "xml_text",
    "xml_xpath", "source_xml_path", "pipeline_version", "validation_status",
]
ISSUE_FIELDS = [
    "issue_type", "score_id", "xml_event_id", "xml_node_id", "note_id",
    "xml_measure", "staff", "voice", "pitch", "pitchName", "start_meas",
    "end_meas", "repeat_json", "source_path",
]


def _local(tag: object) -> str:
    if tag is ET.Comment:
        return "#comment"
    if tag is ET.ProcessingInstruction:
        return "#processing-instruction"
    return str(tag).rsplit("}", 1)[-1]


def _namespace(tag: object) -> str:
    value = str(tag)
    return value[1:].split("}", 1)[0] if value.startswith("{") else ""


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local(child.tag) == name), None)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _text(element: ET.Element | None, name: str | None = None, default: str = "") -> str:
    target = _child(element, name) if element is not None and name else element
    return ((target.text or default).strip() if target is not None else default)


def _float(element: ET.Element | None, name: str, default: float = 0.0) -> float:
    try:
        return float(_text(element, name, str(default)))
    except ValueError:
        return default


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_xml(path: Path) -> ET.Element:
    parser = DefusedXMLParser(
        target=ET.TreeBuilder(insert_comments=True, insert_pis=True),
        # MusicXML 3.x files commonly contain the official Recordare DOCTYPE.
        # Accept the declaration but never resolve entities or external files.
        forbid_dtd=False,
        forbid_entities=True,
        forbid_external=True,
    )
    return ET.parse(path, parser=parser).getroot()


def flatten_nodes(root: ET.Element, score_id: str, xml_path: Path) -> tuple[list[dict], dict[int, dict]]:
    """Flatten every parsed XML node while preserving hierarchy and context."""

    source_hash = _sha256(xml_path)
    rows: list[dict] = []
    by_object: dict[int, dict] = {}
    layout_by_object: dict[int, dict] = {}

    # Precompute layout context. Recursive sibling traversal cannot safely
    # maintain a measure counter when each child receives copied state.
    for part in (element for element in root.iter() if _local(element.tag) == "part"):
        part_id = part.attrib.get("id", "")
        layout_by_object[id(part)] = {"part_id": part_id}
        page, system = 0, 0
        for measure_index, measure in enumerate(_children(part, "measure"), start=1):
            print_element = _child(measure, "print")
            if print_element is not None and print_element.attrib.get("new-page") == "yes":
                page += 1
                system = 1
            elif print_element is not None and print_element.attrib.get("new-system") == "yes":
                system += 1
            elif page == 0:
                page, system = 1, 1
            measure_context = {
                "part_id": part_id,
                "measure_number": measure.attrib.get("number", str(measure_index)),
                "measure_index": measure_index,
                "page": page,
                "system": system,
            }
            for descendant in measure.iter():
                layout_by_object[id(descendant)] = measure_context

    def visit(
        element: ET.Element,
        parent_id: str,
        depth: int,
        xpath: str,
        sibling_index: int,
        context: dict,
    ) -> None:
        tag = _local(element.tag)
        current = dict(context)
        current.update(layout_by_object.get(id(element), {}))

        node_id = f"{score_id}:N{len(rows) + 1:07d}"
        row = {
            "score_id": score_id,
            "xml_node_id": node_id,
            "parent_node_id": parent_id,
            "depth": depth,
            "sibling_index": sibling_index,
            "xml_xpath": xpath,
            "tag": tag,
            "namespace": _namespace(element.tag),
            "attributes_json": _json(dict(element.attrib)),
            "text": (element.text or "").strip(),
            "tail": (element.tail or "").strip(),
            "child_count": len(element),
            "child_node_ids_json": "",
            "part_id": current.get("part_id", ""),
            "measure_number": current.get("measure_number", ""),
            "measure_index": current.get("measure_index", ""),
            "page": current.get("page", ""),
            "system": current.get("system", ""),
            "source_xml_path": str(xml_path),
            "source_sha256": source_hash,
        }
        rows.append(row)
        by_object[id(element)] = row
        tag_counts: dict[str, int] = defaultdict(int)
        child_ids = []
        for child in element:
            child_tag = _local(child.tag)
            tag_counts[child_tag] += 1
            child_xpath = f"{xpath}/{child_tag}[{tag_counts[child_tag]}]"
            visit(
                child,
                node_id,
                depth + 1,
                child_xpath,
                tag_counts[child_tag],
                current,
            )
            child_ids.append(by_object[id(child)]["xml_node_id"])
        row["child_node_ids_json"] = _json(child_ids)

    root_tag = _local(root.tag)
    visit(root, "", 0, f"/{root_tag}[1]", 1, {})
    return rows, by_object


def _load_bps_notes(path: Path) -> list[dict]:
    notes = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for note_id, row in enumerate(csv.DictReader(file, delimiter=";")):
            notes.append(
                {
                    "note_id": note_id,
                    "start_meas": float(row["start_meas"]),
                    "end_meas": float(row["end_meas"]),
                    "midi": int(row["pitch"]),
                    "pitchName": row["pitchName"],
                    "duration_quarterLength": row.get("duration_quarterLength", ""),
                    "timeSig": row.get("timeSig", ""),
                    "articulation": row.get("articulation", ""),
                    "grace": row.get("grace", ""),
                }
            )
    return notes


def _load_repeat_rows(path: Path) -> dict[int, list[dict]]:
    by_written: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("mapping_status") == "unresolved" or not row.get("written_measure_index"):
                continue
            by_written[int(row["written_measure_index"])].append(row)
    for rows in by_written.values():
        rows.sort(key=lambda row: int(row["unfolded_measure_index"]))
    return by_written


def _pitch(note: ET.Element) -> dict:
    pitch = _child(note, "pitch")
    if pitch is None:
        return {}
    step = _text(pitch, "step")
    alter = int(float(_text(pitch, "alter", "0")))
    octave = int(_text(pitch, "octave", "0"))
    semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
    midi = (octave + 1) * 12 + semitone + alter
    suffix = "#" * alter if alter > 0 else "b" * (-alter)
    return {
        "pitch_step": step,
        "pitch_alter": alter,
        "pitch_octave": octave,
        "pitch": midi,
        "pitchName": f"{step}{suffix}{octave}",
        "midi_pitch": midi,
    }


def _empty_event() -> dict:
    row = {field: "" for field in EVENT_FIELDS}
    row.update(
        {
            "row_origin": "xml_only",
            "musical_time": 0,
            "connected_note": "NA",
            "stem_dir": "NA",
            "pipeline_version": PIPELINE_VERSION,
            "validation_status": "ok",
        }
    )
    return row


def _camel_words(value: str) -> str:
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    return "".join(word[:1].upper() + word[1:] for word in words)


def _xml_event_class(
    element: ET.Element,
    event_type: str,
    subtype: str,
    values: dict,
) -> str:
    """Return a readable YOLO-compatible class where MusicXML permits it."""

    placement = element.attrib.get("placement", "").lower()
    placement_suffix = "Above" if placement == "above" else "Below" if placement == "below" else ""
    if event_type == "note":
        return "note"
    if event_type == "rest":
        return "rest"
    if event_type == "attribute":
        return {
            "clef": "xmlClef",
            "key": "xmlKeySignature",
            "time_signature": "xmlTimeSignature",
        }.get(subtype, f"xml{_camel_words(subtype)}")
    if event_type == "barline":
        return "barline"
    if event_type == "notation":
        if subtype == "staccato":
            return f"articStaccato{placement_suffix}"
        if subtype == "staccatissimo":
            return f"articStaccatissimo{placement_suffix}"
        if subtype == "accent":
            return f"articAccent{placement_suffix}"
        if subtype == "strong-accent":
            return f"articMarcato{placement_suffix}"
        if subtype == "fermata":
            return f"fermata{placement_suffix or 'Above'}"
        return {
            "slur": "slur",
            "tie": "tie",
            "tied": "tie",
            "tuplet": "tuplet",
            "trill-mark": "ornamentTrill",
            "wavy-line": "ornamentWiggleTrill",
            "turn": "ornamentTurn",
            "inverted-turn": "ornamentTurnInverted",
        }.get(subtype, f"xmlNotation{_camel_words(subtype)}")
    if event_type == "direction":
        if subtype == "dynamic":
            symbol = str(values.get("dynamic", ""))
            return {
                "f": "dynamicF",
                "p": "dynamicP",
                "s": "dynamicS",
                "mf": "dynamicMF",
                "mp": "dynamicMP",
                "sf": "dynamicSF",
                "sfp": "dynamicSFP",
                "ff": "dynamicFF",
                "pp": "dynamicPP",
                "fp": "dynamicFP",
            }.get(symbol, f"dynamic{symbol.upper()}" if symbol else "dynamic")
        if subtype == "wedge":
            wedge_type = element.attrib.get("type", "")
            if wedge_type == "crescendo":
                return "dynamicCrescendoHairpin"
            if wedge_type == "diminuendo":
                return "dynamicDiminuendoHairpin"
            return "dynamicHairpinStop"
        if subtype == "pedal":
            return (
                "keyboardPedalUp"
                if element.attrib.get("type") in {"stop", "change"}
                else "keyboardPedalPed"
            )
        if subtype == "octave-shift":
            return "ottavaBracket"
        if subtype == "words":
            text = (element.text or "").strip()
            normalized = text.casefold().rstrip(". ")
            tempo_words = {
                "a tempo": "tempoATempo",
                "adagio": "tempoAdagio",
                "allegretto": "tempoAllegretto",
                "allegro": "tempoAllegro",
                "andante": "tempoAndante",
                "grave": "tempoGrave",
                "in tempo": "tempoInTempo",
                "largo": "tempoLargo",
                "moderato": "tempoModerato",
                "prestissimo": "tempoPrestissimo",
                "presto": "tempoPresto",
                "vivace": "tempoVivace",
            }
            if normalized in tempo_words:
                return tempo_words[normalized]
            if normalized.startswith(("rit", "rall")):
                return "tempoRitardando"
            if normalized.startswith("cresc"):
                return "dynamicCrescendo"
            if normalized.startswith(("dim", "decresc")):
                return "dynamicDiminuendo"
            return f"term{_camel_words(text)}" if text else "xmlWords"
        return f"xmlDirection{_camel_words(subtype)}"
    return f"xml{_camel_words(event_type)}{_camel_words(subtype)}"


def extract_events(
    root: ET.Element,
    score_id: str,
    xml_path: Path,
    node_map: dict[int, dict],
    repeat_by_written: dict[int, list[dict]],
    bps_notes: list[dict],
) -> list[dict]:
    """Create semantic event rows with BPS-OMR fields leading the schema."""

    events: list[dict] = []
    event_sequence = 0
    note_sequence = 0
    chord_sequence = 0

    def add_event(element: ET.Element, event_type: str, subtype: str, context: dict, **values) -> dict:
        nonlocal event_sequence
        event_sequence += 1
        row = _empty_event()
        node = node_map[id(element)]
        row.update(
            {
                "record_id": f"{score_id}:R{event_sequence:07d}",
                "score_id": score_id,
                "xml_event_id": f"{score_id}:E{event_sequence:07d}",
                "xml_node_id": node["xml_node_id"],
                "event_type": event_type,
                "event_subtype": subtype,
                "part_id": context.get("part_id", ""),
                "page": context.get("page", ""),
                "system": context.get("system", ""),
                "xml_measure": context.get("measure_number", ""),
                "xml_measure_index": context.get("measure_index", ""),
                "staff": context.get("staff", ""),
                "voice": context.get("voice", ""),
                "onset_divisions": context.get("onset", ""),
                "duration_divisions": context.get("duration", ""),
                "divisions": context.get("divisions", ""),
                "beat_position": context.get("beat_position", ""),
                "duration_quarterLength": context.get("duration_quarterLength", ""),
                "written_start_meas": context.get("written_start", ""),
                "written_end_meas": context.get("written_end", ""),
                "time_signature": context.get("time_signature", ""),
                "xml_attributes_json": _json(dict(element.attrib)),
                "xml_text": (element.text or "").strip(),
                "xml_xpath": node["xml_xpath"],
                "source_xml_path": str(xml_path),
            }
        )
        row.update(values)
        row["class"] = _xml_event_class(
            element, event_type, subtype, values
        )
        mappings = repeat_by_written.get(int(context.get("measure_index", 0)), [])
        within = float(context.get("within", 0.0))
        duration_measures = float(context.get("duration_measures", 0.0))
        occurrences = [
            {
                "unfolded_measure_index": int(mapping["unfolded_measure_index"]),
                "unfolded_measure": mapping.get("unfolded_measure", ""),
                "repeat_occurrence": int(mapping["repeat_occurrence"]),
                "repeat_occurrence_count": int(mapping["repeat_occurrence_count"]),
                "repeat_group_id": mapping.get("repeat_group_id", ""),
                "mapping_status": mapping.get("mapping_status", ""),
                "start_meas": int(mapping["unfolded_measure_index"]) - 1 + float(context.get("timeline_offset", 0)) + within,
                "end_meas": int(mapping["unfolded_measure_index"]) - 1 + float(context.get("timeline_offset", 0)) + within + duration_measures,
            }
            for mapping in mappings
        ]
        if not occurrences and context.get("measure_index"):
            occurrences = [
                {
                    "unfolded_measure_index": context["measure_index"],
                    "unfolded_measure": context["measure_number"],
                    "repeat_occurrence": 1,
                    "repeat_occurrence_count": 1,
                    "repeat_group_id": "",
                    "mapping_status": "written_fallback",
                    "start_meas": context["written_start"],
                    "end_meas": context["written_end"],
                }
            ]
        row["repeat_json"] = _json(occurrences)
        if occurrences:
            row["start_meas"] = f"{occurrences[0]['start_meas']:.6f}"
            row["end_meas"] = f"{occurrences[0]['end_meas']:.6f}"
        events.append(row)
        return row

    parts = [element for element in root.iter() if _local(element.tag) == "part"]
    for part in parts:
        part_id = part.attrib.get("id", "")
        divisions, beats, beat_type = 1, 4, 4
        page, system = 0, 0
        timeline_offset = 0
        measures = _children(part, "measure")
        for measure_index, measure in enumerate(measures, start=1):
            measure_number = measure.attrib.get("number", str(measure_index))
            print_element = _child(measure, "print")
            if print_element is not None and print_element.attrib.get("new-page") == "yes":
                page += 1
                system = 1
            elif print_element is not None and print_element.attrib.get("new-system") == "yes":
                system += 1
            elif page == 0:
                page, system = 1, 1

            attributes = _child(measure, "attributes")
            if attributes is not None:
                if _child(attributes, "divisions") is not None:
                    divisions = int(float(_text(attributes, "divisions", str(divisions))))
                time = _child(attributes, "time")
                if time is not None:
                    beats = int(_text(time, "beats", str(beats)))
                    beat_type = int(_text(time, "beat-type", str(beat_type)))
            nominal = divisions * beats * 4 / beat_type

            cursor = 0.0
            last_onset = 0.0
            raw: list[tuple[ET.Element, str, str, float, float, dict]] = []
            current_chord = -1
            max_cursor = 0.0
            for child in measure:
                name = _local(child.tag)
                if name == "backup":
                    cursor -= _float(child, "duration")
                    continue
                if name == "forward":
                    cursor += _float(child, "duration")
                    max_cursor = max(max_cursor, cursor)
                    continue
                if name == "note":
                    is_chord = _child(child, "chord") is not None
                    if not is_chord:
                        current_chord = chord_sequence
                        chord_sequence += 1
                    onset = last_onset if is_chord else cursor
                    if not is_chord:
                        last_onset = onset
                    duration = _float(child, "duration")
                    raw.append((child, "note", "rest" if _child(child, "rest") is not None else "note", onset, duration, {"is_chord": is_chord, "chord_id": current_chord}))
                    if not is_chord:
                        cursor += duration
                        max_cursor = max(max_cursor, cursor)
                elif name == "direction":
                    onset = cursor + _float(child, "offset")
                    direction_type = _child(child, "direction-type")
                    if direction_type is not None:
                        for item in direction_type:
                            item_name = _local(item.tag)
                            if item_name == "dynamics":
                                for dynamic in item:
                                    raw.append((dynamic, "direction", "dynamic", onset, 0.0, {"dynamic": _local(dynamic.tag), "direction": dict(child.attrib), "staff": _text(child, "staff"), "voice": _text(child, "voice")}))
                            else:
                                raw.append((item, "direction", item_name, onset, 0.0, {"direction": dict(child.attrib), "staff": _text(child, "staff"), "voice": _text(child, "voice")}))
                elif name == "barline":
                    raw.append((child, "barline", "barline", nominal, 0.0, {}))

            pickup = nominal - max_cursor if measure_index == 1 and max_cursor < nominal else 0.0
            if measure_index == 1:
                # BPSD uses measure 0 for an anacrusis, but starts a complete
                # first measure at 1.000. Carry that convention through every
                # repeat-expanded occurrence in this part.
                timeline_offset = 0 if pickup > 0 else 1
            base = {
                "part_id": part_id,
                "page": page,
                "system": system,
                "measure_number": measure_number,
                "measure_index": measure_index,
                "divisions": divisions,
                "time_signature": f"{beats}/{beat_type}",
                "timeline_offset": timeline_offset,
            }
            # Attribute changes are events at the beginning of the measure.
            if attributes is not None:
                for item in attributes:
                    name = _local(item.tag)
                    if name not in {"clef", "key", "time"}:
                        continue
                    context = dict(base, onset=0.0, duration=0.0, within=pickup / nominal, duration_measures=0.0, written_start=measure_index - 1 + timeline_offset + pickup / nominal, written_end=measure_index - 1 + timeline_offset + pickup / nominal)
                    payload = {"staff": item.attrib.get("number", "")}
                    if name == "clef": payload["clef_json"] = _json({"sign": _text(item, "sign"), "line": _text(item, "line"), **item.attrib})
                    elif name == "key": payload["key_signature_json"] = _json({"fifths": _text(item, "fifths"), "mode": _text(item, "mode"), **item.attrib})
                    add_event(item, "attribute", "time_signature" if name == "time" else name, context, **payload)

            for element, event_type, subtype, onset, duration, extra in raw:
                within = (pickup + onset) / nominal if nominal else 0.0
                duration_measures = duration / nominal if nominal else 0.0
                context = dict(
                    base,
                    onset=onset,
                    duration=duration,
                    within=within,
                    duration_measures=duration_measures,
                    written_start=measure_index - 1 + timeline_offset + within,
                    written_end=measure_index - 1 + timeline_offset + within + duration_measures,
                    staff=_text(element, "staff", "1") if event_type == "note" else extra.get("staff", ""),
                    voice=_text(element, "voice", "") if event_type == "note" else extra.get("voice", ""),
                    beat_position=1 + (pickup + onset) / divisions if divisions else "",
                    duration_quarterLength=duration / divisions if divisions else "",
                )
                if event_type == "note":
                    note_sequence += 1
                    pitch = _pitch(element)
                    is_rest = subtype == "rest"
                    stem = _text(element, "stem")
                    row = add_event(
                        element,
                        "rest" if is_rest else "note",
                        subtype,
                        context,
                        **pitch,
                        is_rest=str(is_rest).lower(),
                        is_chord=str(extra["is_chord"]).lower(),
                        is_grace=str(_child(element, "grace") is not None).lower(),
                        chord_id=f"{score_id}:C{extra['chord_id']:07d}",
                        note_type=_text(element, "type"),
                        dot_count=len(_children(element, "dot")),
                        beam_json=_json([{"value": (beam.text or "").strip(), **beam.attrib} for beam in _children(element, "beam")]),
                        accidental=_text(element, "accidental"),
                        stem_dir=0 if stem == "down" else 1 if stem == "up" else "NA",
                    )
                    notations = _child(element, "notations")
                    if notations is not None:
                        for notation in notations:
                            group = _local(notation.tag)
                            marks = list(notation) if group in {"articulations", "ornaments", "technical"} else [notation]
                            for mark in marks:
                                mark_name = _local(mark.tag)
                                notation_row = add_event(mark, "notation", mark_name, context)
                                notation_row["anchor_xml_event_ids_json"] = _json([row["xml_event_id"]])
                                payload = _json([{"tag": mark_name, "text": (mark.text or "").strip(), **mark.attrib}])
                                if group == "articulations": notation_row["articulation_json"] = payload
                                elif group == "ornaments": notation_row["ornament_json"] = payload
                                elif mark_name == "fermata": notation_row["fermata_json"] = payload
                                elif mark_name == "slur": notation_row["slur_json"] = payload
                                elif mark_name == "tied": notation_row["tie_json"] = payload
                                elif mark_name == "tuplet": notation_row["tuplet_json"] = payload
                    for tie in _children(element, "tie"):
                        tie_row = add_event(tie, "notation", "tie", context)
                        tie_row["anchor_xml_event_ids_json"] = _json([row["xml_event_id"]])
                        tie_row["tie_json"] = _json([{"tag": "tie", **tie.attrib}])
                    time_mod = _child(element, "time-modification")
                    if time_mod is not None:
                        row["time_modification_json"] = _json({child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in time_mod})
                elif event_type == "direction":
                    values = {"direction_json": _json(extra.get("direction", {}))}
                    if subtype == "dynamic": values["dynamic"] = extra["dynamic"]
                    elif subtype == "words": values["xml_text"] = (element.text or "").strip()
                    add_event(element, event_type, subtype, context, **values)
                else:
                    row = add_event(element, event_type, subtype, context)
                    row["event_payload_json"] = _json([{"tag": _local(child.tag), "text": (child.text or "").strip(), **child.attrib} for child in element])

    # Assign BPSD note IDs to every unfolded occurrence of pitched note rows.
    queues: dict[tuple[float, int], deque[int]] = defaultdict(deque)
    by_id = {note["note_id"]: note for note in bps_notes}
    for note in bps_notes:
        queues[(round(note["start_meas"], 3), note["midi"])].append(note["note_id"])
    pitched_rows = [
        row for row in events
        if row["event_type"] == "note" and row["midi_pitch"] != ""
    ]
    row_occurrences: dict[str, list[dict]] = {
        row["xml_event_id"]: json.loads(row["repeat_json"] or "[]")
        for row in pitched_rows
    }
    occurrence_tasks = []
    for row in pitched_rows:
        for occurrence_index, occurrence in enumerate(row_occurrences[row["xml_event_id"]]):
            occurrence_tasks.append(
                (
                    float(occurrence["start_meas"]),
                    int(row["midi_pitch"]),
                    row["xml_event_id"],
                    occurrence_index,
                    row,
                    occurrence,
                )
            )
    consumed_note_ids: set[int] = set()
    for occurrence_start, midi, _, _, _row, occurrence in sorted(occurrence_tasks):
        key = (round(occurrence_start, 3), midi)
        while queues[key] and queues[key][0] in consumed_note_ids:
            queues[key].popleft()
        note_id = queues[key].popleft() if queues[key] else None
        match_status = "exact" if note_id is not None else "unmatched"
        if note_id is None:
            rounded_candidates = [
                note
                for note in bps_notes
                if note["note_id"] not in consumed_note_ids
                and note["midi"] == midi
                and abs(note["start_meas"] - occurrence_start) <= 0.0015
            ]
            if rounded_candidates:
                note_id = min(
                    rounded_candidates,
                    key=lambda note: (
                        abs(note["start_meas"] - occurrence_start),
                        note["note_id"],
                    ),
                )["note_id"]
                match_status = "rounded_tolerance"
        if note_id is None:
            candidates = [
                note
                for note in bps_notes
                if note["midi"] == midi
                and note["start_meas"] <= occurrence_start <= note["end_meas"]
            ]
            if candidates:
                note_id = min(
                    candidates,
                    key=lambda note: (
                        abs(note["start_meas"] - occurrence_start),
                        note["note_id"],
                    ),
                )["note_id"]
                match_status = "within_tied_span"
        occurrence["note_id"] = note_id
        occurrence["note_match_status"] = match_status
        if note_id is not None:
            consumed_note_ids.add(note_id)

    for row in pitched_rows:
        occurrences = row_occurrences[row["xml_event_id"]]
        note_ids = [
            occurrence["note_id"]
            for occurrence in occurrences
            if occurrence.get("note_id") is not None
        ]
        row["repeat_json"] = _json(occurrences)
        unique_ids = list(dict.fromkeys(note_ids))
        if unique_ids:
            first = by_id[unique_ids[0]]
            row["start_note"] = unique_ids[0]
            row["end_note"] = unique_ids[0]
            row["connected_note"] = _json(unique_ids)
            row["start_meas"] = f"{first['start_meas']:.6f}"
            row["end_meas"] = f"{first['end_meas']:.6f}"
            row["validation_status"] = "bps_note_matched"
        else:
            row["start_note"] = ""
            row["end_note"] = ""
            row["connected_note"] = ""
            row["validation_status"] = "bps_note_unmatched"
    return events


def export_score(score: dict, repeat_mapping_dir: Path) -> tuple[list[dict], list[dict]]:
    xml_path = Path(score["xml_path"])
    root = _parse_xml(xml_path)
    nodes, node_map = flatten_nodes(root, score["score_id"], xml_path)
    repeats = _load_repeat_rows(
        repeat_mapping_dir / f"{score['score_id']}_repeat_mapping.csv"
    )
    events = extract_events(
        root,
        score["score_id"],
        xml_path,
        node_map,
        repeats,
        _load_bps_notes(Path(score["bps_notes_path"])),
    )
    return nodes, events


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def export_dataset(
    manifest_path: Path,
    repeat_mapping_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict:
    manifest = _read_csv(manifest_path)
    scores = {}
    for row in manifest:
        scores.setdefault(
            row["score_id"],
            {key: row[key] for key in ["score_id", "xml_path", "bps_notes_path"]},
        )
    all_nodes, all_events = [], []
    score_reports = []
    emit_progress("xml-export", 0, len(scores), f"starting (resume={resume})")
    for index, score in enumerate(scores.values(), start=1):
        score_id = score["score_id"]
        score_dir = output_dir / "per_scores" / score_id
        nodes_path = score_dir / "xml_nodes.csv"
        events_path = score_dir / "xml_events.csv"
        checkpoint_path = output_dir / "checkpoints" / f"{score_id}.json"
        repeat_path = repeat_mapping_dir / f"{score_id}_repeat_mapping.csv"
        fingerprint = stable_digest(
            {
                "version": PIPELINE_VERSION,
                "xml": path_signature(Path(score["xml_path"])),
                "bps": path_signature(Path(score["bps_notes_path"])),
                "repeat": path_signature(repeat_path),
            }
        )
        reused = False
        if resume and checkpoint_path.is_file() and nodes_path.is_file() and events_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                nodes = _read_csv(nodes_path)
                events = _read_csv(events_path)
                reused = (
                    checkpoint.get("fingerprint") == fingerprint
                    and checkpoint.get("node_rows") == len(nodes)
                    and checkpoint.get("event_rows") == len(events)
                )
            except (OSError, json.JSONDecodeError, csv.Error):
                reused = False
        if not reused:
            nodes, events = export_score(score, repeat_mapping_dir)
            atomic_write_csv(nodes_path, NODE_FIELDS, nodes)
            atomic_write_csv(events_path, EVENT_FIELDS, events)
            atomic_write_json(
                checkpoint_path,
                {
                    "score_id": score_id,
                    "pipeline_version": PIPELINE_VERSION,
                    "fingerprint": fingerprint,
                    "node_rows": len(nodes),
                    "event_rows": len(events),
                },
            )
        all_nodes.extend(nodes)
        all_events.extend(events)
        score_reports.append(
            {"score_id": score_id, "node_rows": len(nodes), "event_rows": len(events), "resumed": reused}
        )
        emit_progress("xml-export", index, len(scores), f"{score_id} {'resumed' if reused else 'exported'} nodes={len(nodes)} events={len(events)}")

    atomic_write_csv(output_dir / "xml_nodes.csv", NODE_FIELDS, all_nodes)
    atomic_write_csv(output_dir / "xml_events.csv", EVENT_FIELDS, all_events)
    node_ids = [row["xml_node_id"] for row in all_nodes]
    event_ids = [row["xml_event_id"] for row in all_events]
    node_id_set = set(node_ids)
    errors = []
    if len(node_ids) != len(node_id_set): errors.append("duplicate xml_node_id")
    if len(event_ids) != len(set(event_ids)): errors.append("duplicate xml_event_id")
    missing_nodes = sorted({row["xml_node_id"] for row in all_events} - node_id_set)
    if missing_nodes: errors.append(f"event references {len(missing_nodes)} missing nodes")
    for row in all_nodes:
        for field in ["attributes_json", "child_node_ids_json"]:
            try:
                value = json.loads(row[field])
            except json.JSONDecodeError: errors.append(f"invalid {field}: {row['xml_node_id']}")
            else:
                if field == "child_node_ids_json":
                    missing_children = [child for child in value if child not in node_id_set]
                    if missing_children:
                        errors.append(
                            f"node references {len(missing_children)} missing children: {row['xml_node_id']}"
                        )
    for row in all_events:
        for field in [
            "repeat_json", "event_payload_json", "anchor_xml_event_ids_json",
            "xml_attributes_json", "beam_json", "articulation_json",
            "fermata_json", "slur_json", "tie_json", "tuplet_json",
            "ornament_json", "time_modification_json", "direction_json",
            "clef_json", "key_signature_json",
        ]:
            try: json.loads(row[field] or "[]")
            except json.JSONDecodeError: errors.append(f"invalid {field}: {row['xml_event_id']}")
    bps_score_reports = []
    bps_issue_rows = []
    total_bps_notes = 0
    total_referenced_bps_notes = 0
    occurrence_match_statuses: Counter = Counter()
    for score_id, score in scores.items():
        expected_bps_notes = _load_bps_notes(Path(score["bps_notes_path"]))
        expected_ids = {note["note_id"] for note in expected_bps_notes}
        score_note_rows = [
            row for row in all_events
            if row["score_id"] == score_id and row["event_type"] == "note"
        ]
        referenced_ids = set()
        score_occurrence_statuses: Counter = Counter()
        for row in score_note_rows:
            for occurrence in json.loads(row["repeat_json"] or "[]"):
                score_occurrence_statuses[occurrence.get("note_match_status", "unknown")] += 1
                if occurrence.get("note_id") is not None:
                    referenced_ids.add(int(occurrence["note_id"]))
        for row in score_note_rows:
            if row["validation_status"] != "bps_note_unmatched":
                continue
            bps_issue_rows.append(
                {
                    "issue_type": "xml_note_unmatched",
                    "score_id": score_id,
                    "xml_event_id": row["xml_event_id"],
                    "xml_node_id": row["xml_node_id"],
                    "note_id": "",
                    "xml_measure": row["xml_measure"],
                    "staff": row["staff"],
                    "voice": row["voice"],
                    "pitch": row["pitch"],
                    "pitchName": row["pitchName"],
                    "start_meas": row["start_meas"],
                    "end_meas": row["end_meas"],
                    "repeat_json": row["repeat_json"],
                    "source_path": row["source_xml_path"],
                }
            )
        for note in expected_bps_notes:
            if note["note_id"] in referenced_ids:
                continue
            bps_issue_rows.append(
                {
                    "issue_type": "bps_note_unreferenced",
                    "score_id": score_id,
                    "xml_event_id": "",
                    "xml_node_id": "",
                    "note_id": note["note_id"],
                    "xml_measure": "",
                    "staff": "",
                    "voice": "",
                    "pitch": note["midi"],
                    "pitchName": note["pitchName"],
                    "start_meas": note["start_meas"],
                    "end_meas": note["end_meas"],
                    "repeat_json": "",
                    "source_path": score["bps_notes_path"],
                }
            )
        total_bps_notes += len(expected_ids)
        total_referenced_bps_notes += len(referenced_ids)
        occurrence_match_statuses.update(score_occurrence_statuses)
        bps_score_reports.append(
            {
                "score_id": score_id,
                "bps_note_rows": len(expected_ids),
                "bps_note_ids_referenced": len(referenced_ids),
                "bps_note_ids_unreferenced": len(expected_ids - referenced_ids),
                "bps_note_coverage": (
                    len(referenced_ids) / len(expected_ids) if expected_ids else 1.0
                ),
                "xml_note_rows_unmatched": sum(
                    row["validation_status"] == "bps_note_unmatched"
                    for row in score_note_rows
                ),
                "occurrence_match_status_counts": dict(score_occurrence_statuses),
            }
        )
    atomic_write_csv(output_dir / "bps_match_issues.csv", ISSUE_FIELDS, bps_issue_rows)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "scores": len(scores),
        "node_rows": len(all_nodes),
        "event_rows": len(all_events),
        "unique_node_ids": len(node_id_set),
        "unique_event_ids": len(set(event_ids)),
        "event_type_counts": dict(Counter(row["event_type"] for row in all_events)),
        "bps_note_status_counts": dict(Counter(row["validation_status"] for row in all_events if row["event_type"] == "note")),
        "bps_note_rows": total_bps_notes,
        "bps_note_ids_referenced": total_referenced_bps_notes,
        "bps_note_ids_unreferenced": total_bps_notes - total_referenced_bps_notes,
        "bps_note_coverage": (
            total_referenced_bps_notes / total_bps_notes if total_bps_notes else 1.0
        ),
        "bps_match_issue_rows": len(bps_issue_rows),
        "occurrence_match_status_counts": dict(occurrence_match_statuses),
        "bps_score_reports": bps_score_reports,
        "score_reports": score_reports,
        "validation_errors": errors,
        "passed": not errors,
        "outputs": {
            "xml_nodes": str(output_dir / "xml_nodes.csv"),
            "xml_events": str(output_dir / "xml_events.csv"),
            "bps_match_issues": str(output_dir / "bps_match_issues.csv"),
        },
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    emit_progress("xml-validation", 1, 1, f"passed={report['passed']} errors={len(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat-mapping-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = export_dataset(
        args.manifest,
        args.repeat_mapping_dir,
        args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
