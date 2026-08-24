"""Canonical BPS-OMR CSV field definitions shared by every pipeline surface."""

BPS_OMR_FIELDS = [
    "class_id",
    "x",
    "y",
    "w",
    "h",
    "class",
    "musical_time",
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
]

FINAL_BPS_FIELDS = [
    *BPS_OMR_FIELDS,
    "human_corrected",
    "is_repeated_measure",
]

FINAL_UNCERTAIN_FIELDS = [
    "start_meas",
    "end_meas",
    "start_note",
    "end_note",
    "connected_note",
    "stem_dir",
]
