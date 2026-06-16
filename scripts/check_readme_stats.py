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
# Non-greedy value group so a value may itself contain "<" (e.g. "<0.05"); it
# still stops at the first closing tag.
MARKER = re.compile(r"<!-- stats:([a-z0-9_]+) -->(.+?)<!-- /stats -->")
# The report's README-values block: "- KEY = value", KEY in [a-z0-9_]. No other
# report line matches (table rows start with "|"; appendix keys carry ":"/caps).
REPORT_LINE = re.compile(r"^- ([a-z0-9_]+) = (.+)$", re.MULTILINE)


def check(readme: str, report: str) -> list[str]:
    """Return one failure string per drifted, missing, or unbacked marker (empty == pass).

    Each README marker is matched against the report's ``KEY = value`` lines by
    exact value equality. A substring test would pass a truncated value (README
    ``0.71`` against report ``0.718``), which is the drift the checker exists to catch.
    """
    failures = []
    report_values = dict(REPORT_LINE.findall(report))
    pairs = MARKER.findall(readme)
    if not pairs:
        failures.append("README contains no stats markers; WP6 edits not applied")
    if not report_values:
        failures.append("report states no 'KEY = value' lines; run `make evaluate-stats`")
    for key, value in pairs:
        value = value.strip()
        if key not in report_values:
            failures.append(f"README marks {key} but the report states no such key")
        elif report_values[key] != value:
            failures.append(
                f"README claims {key} = {value} but report says {key} = {report_values[key]}"
            )
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
