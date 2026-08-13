from types import SimpleNamespace

import pytest

from bpsd_aligner.web_utils import (
    apply_page_mapping_edits,
    apply_page_number_edits,
    group_review_overlays,
    pair_page_uploads,
    read_csv_bytes,
    sanitize_path_columns,
    summarize_rows,
)


def test_group_review_overlays_builds_page_first_review_and_class_model():
    overlays = {
        "page-01__review_overlay": b"review-1",
        "page-01__all_symbols_overlay": b"all-1",
        "class_page-01__slur_overlay": b"slur-1",
        "class_page-01__tie_overlay": b"tie-1",
        "page-02__review_overlay": b"review-2",
        "class_page-02__slur_overlay": b"slur-2",
    }
    rows = [
        {"page_id": "page-01", "status": "matched"},
        {"page_id": "page-01", "status": "review"},
        {"page_id": "page-01", "status": "unmatched"},
        {"page_id": "page-02", "status": "matched"},
    ]

    pages = group_review_overlays(overlays, rows)

    assert list(pages) == ["page-01", "page-02"]
    assert pages["page-01"] == {
        "review_overlay": b"review-1",
        "classes": {"slur": b"slur-1", "tie": b"tie-1"},
        "needs_review": 2,
    }
    assert pages["page-02"] == {
        "review_overlay": b"review-2",
        "classes": {"slur": b"slur-2"},
        "needs_review": 0,
    }


def test_sanitize_path_columns_keeps_only_filenames():
    source = (
        "score_id,row_origin,combined_status,image_path,source_xml_path,value\n"
        "S1,yolo,aligned,/example/data/page.png,/example/data/score.xml,ok\n"
    ).encode()
    fields, rows = read_csv_bytes(sanitize_path_columns(source))
    assert fields[-1] == "value"
    assert rows[0]["image_path"] == "page.png"
    assert rows[0]["source_xml_path"] == "score.xml"
    assert rows[0]["value"] == "ok"


def test_summary_counts_rows_scores_origins_and_statuses():
    summary = summarize_rows(
        [
            {"score_id": "S1", "row_origin": "yolo", "combined_status": "aligned"},
            {"score_id": "S1", "row_origin": "xml", "combined_status": "xml_only"},
            {"score_id": "S2", "row_origin": "yolo", "combined_status": "ambiguous"},
        ]
    )
    assert summary["rows"] == 3
    assert summary["scores"] == 2
    assert summary["origins"] == {"yolo": 2, "xml": 1}


def test_read_csv_bytes_accepts_large_review_candidate_field():
    payload = "x" * 140_000
    fields, rows = read_csv_bytes(
        ("id,review_note_candidates_json\n1," + payload + "\n").encode()
    )

    assert fields == ["id", "review_note_candidates_json"]
    assert rows[0]["review_note_candidates_json"] == payload


def test_pair_page_uploads_matches_stems_and_infers_page_suffixes():
    images = [
        SimpleNamespace(name="score-02.png"),
        SimpleNamespace(name="score-01.jpeg"),
    ]
    yolos = [
        SimpleNamespace(name="score-01.txt"),
        SimpleNamespace(name="score-02.txt"),
    ]

    pairs = pair_page_uploads(
        images,
        yolos,
        first_page=10,
        infer_page_from_filename=True,
    )

    assert [(pair["stem"], pair["page_number"]) for pair in pairs] == [
        ("score-01", 1),
        ("score-02", 2),
    ]


def test_pair_page_uploads_can_assign_consecutive_pages_and_reject_missing_pairs():
    images = [SimpleNamespace(name="score-a.png"), SimpleNamespace(name="score-b.png")]
    yolos = [SimpleNamespace(name="score-a.txt"), SimpleNamespace(name="score-b.txt")]

    pairs = pair_page_uploads(
        images,
        yolos,
        first_page=4,
        infer_page_from_filename=False,
    )
    assert [pair["page_number"] for pair in pairs] == [4, 5]

    with pytest.raises(ValueError, match="missing YOLO TXT for: score-b"):
        pair_page_uploads(
            images,
            yolos[:1],
            first_page=4,
            infer_page_from_filename=False,
        )


def test_apply_page_number_edits_accepts_manual_mapping_and_rejects_duplicates():
    pairs = [
        {"stem": "page-a", "page_number": 1},
        {"stem": "page-b", "page_number": 2},
    ]
    edited = apply_page_number_edits(
        pairs,
        [
            {"page stem": "page-a", "MusicXML page": 5},
            {"page stem": "page-b", "MusicXML page": 7},
        ],
    )
    assert [pair["page_number"] for pair in edited] == [5, 7]
    with pytest.raises(ValueError, match="Duplicate MusicXML page"):
        apply_page_number_edits(
            pairs,
            [
                {"page stem": "page-a", "MusicXML page": 5},
                {"page stem": "page-b", "MusicXML page": 5},
            ],
        )


def test_page_mapping_edits_parse_scan_system_starts_and_derive_page_end():
    pairs = [
        {"stem": "page-05", "page_number": 5},
        {"stem": "page-06", "page_number": 6},
    ]

    mapped = apply_page_mapping_edits(
        pairs,
        [
            {
                "page stem": "page-05",
                "MusicXML page": 5,
                "Scan system starts": "170, 176, 181",
            },
            {
                "page stem": "page-06",
                "MusicXML page": 6,
                "Scan system starts": "198，202 206、211;223 235",
                "Scan page end": 246,
            },
        ],
    )

    assert mapped[0]["system_start_measures"] == [170, 176, 181]
    assert mapped[0]["page_end_measure"] == 197
    assert mapped[1]["system_start_measures"] == [198, 202, 206, 211, 223, 235]
    assert mapped[1]["page_end_measure"] == 246
