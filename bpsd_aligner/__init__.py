"""BPSD MusicXML–YOLO alignment toolkit."""

import csv
import sys


# Detailed alignment CSVs deliberately retain a JSON note index for direct
# image-based review.  A full score page can exceed Python's historical
# 128-KiB default CSV field limit, so configure one bounded process-wide limit
# before any web, worker, or CLI module opens those files.
CSV_FIELD_SIZE_LIMIT = 16 * 1024 * 1024
csv.field_size_limit(min(sys.maxsize, CSV_FIELD_SIZE_LIMIT))

__version__ = "0.3.2"
