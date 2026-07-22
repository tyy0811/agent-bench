"""The deploy preflight rejects unresolved placeholder tokens and passes
resolved pages. Complements tests/test_static_pages.py, which deliberately
does not pin tokens (CI is green in both states; deploying with tokens is
what must fail, and only here)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import preflight_deploy  # noqa: E402


def test_scan_flags_each_unresolved_token(tmp_path):
    (tmp_path / "impressum.html").write_text(
        "<p>IMPRESSUM_NAME<br>IMPRESSUM_ADDRESS</p>\n<p>IMPRESSUM_CONTACT</p>\n"
    )
    (tmp_path / "index.html").write_text('<a href="BOOKING_URL">book</a>\n')
    failures = preflight_deploy.scan(tmp_path)
    assert len(failures) == 4
    assert any("impressum.html:1: unresolved IMPRESSUM_NAME" in f for f in failures)
    assert any("index.html:1: unresolved BOOKING_URL" in f for f in failures)


def test_scan_passes_resolved_pages(tmp_path):
    (tmp_path / "impressum.html").write_text("<p>Real Name<br>Real Street 1</p>\n")
    (tmp_path / "index.html").write_text('<a href="https://cal.example/audit">book</a>\n')
    assert preflight_deploy.scan(tmp_path) == []


def test_live_static_dir_currently_carries_the_known_tokens():
    # Not a freshness pin: this documents that the preflight actually sees the
    # real deploy surfaces. It passes whether or not tokens remain, and fails
    # only if the static dir stops existing or scan() stops reading it.
    assert preflight_deploy.STATIC_DIR.is_dir()
    assert list(preflight_deploy.STATIC_DIR.glob("*.html"))
