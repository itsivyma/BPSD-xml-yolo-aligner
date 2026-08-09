from pathlib import Path

from batch_align import discover_pages


def test_discover_pages_pairs_numeric_suffixes_and_reports_missing(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "Piece-01.jpeg").touch()
    (images / "Piece-02.jpeg").touch()
    (labels / "Piece-01.txt").touch()
    (labels / "Piece-03.txt").touch()

    ready, missing = discover_pages(tmp_path, "Piece", start_page=1)

    assert [item["page"] for item in ready] == [1]
    assert missing == [
        {
            "page": 2,
            "missing_image": False,
            "missing_yolo": True,
            "image": str(images / "Piece-02.jpeg"),
            "yolo": "",
        },
        {
            "page": 3,
            "missing_image": True,
            "missing_yolo": False,
            "image": "",
            "yolo": str(labels / "Piece-03.txt"),
        },
    ]
