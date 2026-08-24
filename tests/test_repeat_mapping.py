from pathlib import Path

from repeat_mapping import align_fingerprints, build_repeat_mapping


def test_align_fingerprints_maps_inserted_repeat_occurrence() -> None:
    written = ["m1", "m2", "m3", "m4", "m5"]
    unfolded = ["m1", "m2", "m3", "m2", "m3", "m4", "m5"]

    mapping, _evidence = align_fingerprints(written, unfolded)

    assert mapping == [0, 1, 2, 1, 2, 3, 4]


def test_align_fingerprints_fills_gap_between_exact_contiguous_anchors() -> None:
    written = ["m1", "m2", "m3", "m4"]
    unfolded = ["m1", "exported-m2", "exported-m3", "m4"]

    mapping, evidence = align_fingerprints(written, unfolded)

    assert mapping == [0, 1, 2, 3]
    assert evidence[-1]["method"] == "bounded_contiguous_interpolation"


def _write_score(tmp_path: Path, measures: str) -> Path:
    path = tmp_path / "score.xml"
    path.write_text(
        """<?xml version="1.0"?><score-partwise><part-list>
        <score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
        <part id="P1">"""
        + measures
        + "</part></score-partwise>",
        encoding="utf-8",
    )
    return path


def test_repeat_mapping_uses_musicxml_repeat_structure_without_unfolded_xml(
    tmp_path: Path,
) -> None:
    path = _write_score(
        tmp_path,
        """
        <measure number="1"><barline location="left"><repeat direction="forward"/></barline></measure>
        <measure number="2"/>
        <measure number="3"><barline location="right"><repeat direction="backward"/></barline></measure>
        <measure number="4"/>
        """,
    )

    report = build_repeat_mapping(path)

    assert [row["written_measure_index"] for row in report["rows"]] == [
        1, 2, 3, 1, 2, 3, 4
    ]
    measure_two = [
        row for row in report["rows"] if row["written_measure_index"] == 2
    ]
    assert [row["repeat_occurrence"] for row in measure_two] == [1, 2]
    assert all(row["is_repeated_measure"] for row in measure_two)
    assert all(row["repeat_group_id"] == "R01" for row in measure_two)
    assert all(row["repeat_status"] == "repeat_body" for row in measure_two)
    assert report["repeat_source"] == "repetition_musicxml"


def test_repeat_mapping_skips_first_ending_on_second_pass(tmp_path: Path) -> None:
    path = _write_score(
        tmp_path,
        """
        <measure number="1"><barline location="left"><repeat direction="forward"/></barline></measure>
        <measure number="2"/>
        <measure number="3"><barline location="left"><ending number="1" type="start"/></barline></measure>
        <measure number="4"><barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline></measure>
        <measure number="5"><barline location="left"><ending number="2" type="start"/></barline></measure>
        <measure number="6"><barline location="right"><ending number="2" type="stop"/></barline></measure>
        """,
    )

    report = build_repeat_mapping(path)

    assert [row["written_measure_index"] for row in report["rows"]] == [
        1, 2, 3, 4, 1, 2, 5, 6
    ]
    ending = next(row for row in report["rows"] if row["written_measure_index"] == 5)
    assert ending["repeat_status"] == "ending_2"
    assert ending["volta_numbers"] == "[2]"
    assert ending["repeat_occurrence"] == 1
    assert ending["repeat_group_id"] == "R01"


def test_backward_repeat_without_forward_leaves_pickup_outside_repeat(
    tmp_path: Path,
) -> None:
    path = _write_score(
        tmp_path,
        """
        <measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>4</duration></note></measure>
        <measure number="2"><note><rest/><duration>16</duration></note></measure>
        <measure number="3"><note><rest/><duration>16</duration></note><barline location="right"><repeat direction="backward"/></barline></measure>
        <measure number="4"><note><rest/><duration>16</duration></note></measure>
        """,
    )

    report = build_repeat_mapping(path)

    assert [row["written_measure_index"] for row in report["rows"]] == [
        1, 2, 3, 2, 3, 4
    ]
    pickup = next(
        row for row in report["rows"] if row["written_measure_index"] == 1
    )
    assert pickup["is_repeated_measure"] is False


def test_sibelius_combined_first_and_second_ending_stays_local(
    tmp_path: Path,
) -> None:
    path = _write_score(
        tmp_path,
        """
        <measure number="1"/>
        <measure number="2"><barline location="left"><repeat direction="forward"/></barline></measure>
        <measure number="3"/>
        <measure number="4"><barline location="left"><ending number="1" type="start"/></barline><barline location="left"><ending number="2" type="start"/></barline><barline location="right"><repeat direction="backward"/></barline></measure>
        <measure number="5"><barline location="right"><ending number="2" type="discontinue"/></barline></measure>
        <measure number="6"/>
        """,
    )

    report = build_repeat_mapping(path)

    assert [row["written_measure_index"] for row in report["rows"]] == [
        1, 2, 3, 4, 2, 3, 5, 6
    ]
    measure_four = next(
        row for row in report["rows"] if row["written_measure_index"] == 4
    )
    measure_five = next(
        row for row in report["rows"] if row["written_measure_index"] == 5
    )
    measure_six = next(
        row for row in report["rows"] if row["written_measure_index"] == 6
    )
    assert measure_four["volta_numbers"] == "[1]"
    assert measure_five["volta_numbers"] == "[1, 2]"
    assert measure_six["volta_numbers"] == "[]"
