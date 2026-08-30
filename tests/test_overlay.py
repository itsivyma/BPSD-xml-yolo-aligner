from PIL import Image

from bpsd_aligner.bps_xml_alignment import render_overlay
from bpsd_aligner.overlay import render_alignment_overlay


def _row() -> dict:
    return {
        "txt_line": "1",
        "class": "slur",
        "x": "0.5",
        "y": "0.5",
        "w": "0.2",
        "h": "0.1",
        "status": "review",
        "start_meas": "1",
        "end_meas": "2",
        "start_xml_measure": "1",
        "end_xml_measure": "2",
        "xml_measure": "1",
        "target_x_px": "",
        "target_y_px": "",
    }


def test_compatibility_overlay_matches_isolated_renderer_pixels():
    image = Image.new("RGB", (300, 200), "white")
    expected = render_alignment_overlay(
        image, [_row()], "class", frozenset({"dynamicF"})
    )
    actual = render_overlay(image, [_row()], "class")

    assert actual.tobytes() == expected.tobytes()
