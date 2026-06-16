"""Verify README statistics match docs/_generated/stats_report.md.

Protocol: every number the README quotes from the report is wrapped in an
HTML marker comment: ``<!-- stats:KEY -->value<!-- /stats -->``. The same KEY
must appear in the report as a line containing ``KEY = value`` (emitted by the
README-values block in stats/report.py). Drift in either direction fails
loudly: regenerate the report (``make evaluate-stats``) then run this check.

Run from the repo (``python scripts/check_readme_stats.py``); paths resolve
relative to this file, so the working directory does not matter.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"<!-- stats:([a-z0-9_]+) -->([^<]+)<!-- /stats -->")


def check(readme: str, report: str) -> list[str]:
    """Return one failure string per drifted or missing marker (empty == pass)."""
    failures = []
    pairs = MARKER.findall(readme)
    if not pairs:
        failures.append("README contains no stats markers; WP6 edits not applied")
    for key, value in pairs:
        needle = f"{key} = {value.strip()}"
        if needle not in report:
            failures.append(f"README claims {key} = {value.strip()} but report does not state it")
    return failures


def main() -> int:
    report_path = ROOT / "docs" / "_generated" / "stats_report.md"
    readme_path = ROOT / "README.md"
    if not report_path.exists():
        print(f"FAIL: {report_path} missing; run `make evaluate-stats` first")
        return 1
    readme = readme_path.read_text()
    failures = check(readme, report_path.read_text())
    for line in failures:
        print(f"FAIL: {line}")
    if not failures:
        print(f"OK: all {len(MARKER.findall(readme))} README stats markers match the report")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
