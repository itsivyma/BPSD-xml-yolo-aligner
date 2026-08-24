from bps_xml_alignment import OUTPUT_FIELDS
from bpsd_aligner.schema import BPS_OMR_FIELDS, FINAL_BPS_FIELDS
from bpsd_aligner.web_pipeline import FINAL_BPS_FIELDS as WEB_FINAL_FIELDS
from dataset_dry_run import OFFICIAL_FIELDS
from xml_export import BPS_FIELDS


def test_every_pipeline_surface_uses_the_canonical_bps_omr_schema() -> None:
    assert OUTPUT_FIELDS is BPS_OMR_FIELDS
    assert OFFICIAL_FIELDS is BPS_OMR_FIELDS
    assert BPS_FIELDS is BPS_OMR_FIELDS
    assert WEB_FINAL_FIELDS is FINAL_BPS_FIELDS
