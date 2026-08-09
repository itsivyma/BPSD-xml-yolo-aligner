from bpsd_aligner.web_utils import read_csv_bytes, sanitize_path_columns, summarize_rows


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
