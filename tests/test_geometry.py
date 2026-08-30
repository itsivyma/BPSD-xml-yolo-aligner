from bpsd_aligner.bps_xml_alignment import SystemGeometry as CompatibilitySystemGeometry
from bpsd_aligner.bps_xml_alignment import assign_system as compatibility_assign_system
from bpsd_aligner.geometry import StaffGeometry, SystemGeometry, assign_system


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
