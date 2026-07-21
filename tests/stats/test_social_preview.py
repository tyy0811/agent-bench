"""The committed social preview card (assets/social_preview.png) stays a
1280x640 PNG carrying the same source-hash pin as the README plots.

Byte-level checks only (PNG IHDR + tEXt), so this runs without matplotlib
or the [plots] extra, like the rest of the plot tests. The card is not in
EXPECTED_PLOTS on purpose: it is uploaded in GitHub repo settings rather
than embedded in the README, so the freshness check does not gate it.
"""

import re
import struct
from pathlib import Path

ASSET = Path(__file__).parents[2] / "assets" / "social_preview.png"


def test_social_preview_is_1280x640_png():
    data = ASSET.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1280, 640)


def test_social_preview_carries_source_hash_pin():
    data = ASSET.read_bytes()
    assert re.search(rb"source-hash:[0-9a-f]{16}", data) is not None
