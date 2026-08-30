from PIL import Image, ImageDraw

from bpsd_aligner.bps_xml_alignment import SystemGeometry as CompatibilitySystemGeometry
from bpsd_aligner.bps_xml_alignment import assign_system as compatibility_assign_system
from bpsd_aligner.geometry import (
    StaffGeometry,
    SystemGeometry,
    assign_system,
    detect_systems,
)


def _systems() -> list[SystemGeometry]:
    return [
        SystemGeometry(
            number=1,
            upper=StaffGeometry(center=100, line_spacing=10, lines=[]),
            lower=StaffGeometry(center=200, line_spacing=10, lines=[]),
            x_left=10,
            x_right=990,
        ),
        SystemGeometry(
            number=2,
            upper=StaffGeometry(center=500, line_spacing=10, lines=[]),
            lower=StaffGeometry(center=600, line_spacing=10, lines=[]),
            x_left=10,
            x_right=990,
        ),
    ]


def test_geometry_module_assigns_nearest_system():
    systems = _systems()
    assert assign_system({"y": 0.16}, systems, 1000) == 1
    assert assign_system({"y": 0.54}, systems, 1000) == 2


def test_alignment_module_preserves_geometry_compatibility_exports():
    assert CompatibilitySystemGeometry is SystemGeometry
    assert compatibility_assign_system is assign_system


def test_detect_systems_ignores_uniform_scanner_border_bands():
    image = Image.new("L", (1000, 1400), 255)
    draw = ImageDraw.Draw(image)
    top = 80
    for _system_index in range(6):
        for staff_top in (top, top + 70):
            for line_index in range(5):
                y = staff_top + line_index * 8
                draw.line((60, y, 940, y), fill=0, width=2)
        top += 200

    for band_index in range(5):
        y = 1320 + band_index * 15
        draw.rectangle((0, y, 999, y + 1), fill=100)

    systems = detect_systems(image)

    assert len(systems) == 6
