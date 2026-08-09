from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_app_loads_without_exceptions():
    app = Path(__file__).parents[1] / "bpsd_aligner" / "web.py"
    result = AppTest.from_file(str(app)).run(timeout=15)
    assert not result.exception
    assert result.title[0].value == "BPSD XML–YOLO Aligner"
    assert [tab.label for tab in result.tabs] == [
        "Run alignment",
        "Inspect CSV",
        "Advanced combine",
        "CLI guide",
    ]
