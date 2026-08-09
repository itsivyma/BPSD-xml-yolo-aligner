from alignment_scope import resolve_system_mapping


def test_resolve_system_mapping_defaults_to_identity_then_outside_scope() -> None:
    direct = resolve_system_mapping(
        page_id="score-01", scan_page=1, scan_system=2, xml_system_count=3, overrides={}
    )
    outside = resolve_system_mapping(
        page_id="score-01", scan_page=1, scan_system=4, xml_system_count=3, overrides={}
    )

    assert direct["xml_systems"] == [2]
    assert direct["movement_scope_status"] == "in_bpsd_scope"
    assert outside["movement_scope_status"] == "outside_bpsd_scope"


def test_resolve_system_mapping_supports_merged_xml_system_override() -> None:
    result = resolve_system_mapping(
        page_id="score-04",
        scan_page=4,
        scan_system=5,
        xml_system_count=6,
        overrides={
            "score-04": {
                "5": {
                    "xml_page": 4,
                    "xml_systems": [5, 6],
                    "mapping_status": "merged_xml_systems",
                }
            }
        },
    )

    assert result["xml_systems"] == [5, 6]
    assert result["review_status"] == "needs_review"
