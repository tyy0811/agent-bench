"""The committed social preview card (assets/social_preview.png) stays a
1280x640 PNG whose embedded source-hash matches the forest source computed
from the live stats report, i.e. the card cannot go stale with green CI.

Byte-level checks only (PNG IHDR + tEXt), so this runs without matplotlib
or the [plots] extra, like the rest of the plot tests. The card is not in
EXPECTED_PLOTS (it is uploaded in GitHub repo settings rather than embedded
in the README), so this test IS its freshness gate.
"""

import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import make_plots  # noqa: E402

ASSET = Path(__file__).parents[2] / "assets" / "social_preview.png"


def test_social_preview_is_1280x640_png():
    data = ASSET.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1280, 640)


def test_social_preview_hash_is_fresh_against_the_report():
    values = make_plots.read_values(make_plots.REPORT.read_text())
    expected = make_plots.source_hash(make_plots.forest_source(values))
    match = re.search(rb"source-hash:([0-9a-f]{16})", ASSET.read_bytes())
    assert match is not None, "no source-hash tEXt chunk in the card"
    embedded = match.group(1).decode()
    assert embedded == expected, (
        f"social card is stale: embedded {embedded}, report computes {expected}; "
        "regenerate with `make plots`"
    )
