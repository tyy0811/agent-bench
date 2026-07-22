"""Fail if any README test-count claim drifts from the measured suite size.

The README states the deterministic test count in four places (the badge
chip, two `make test` comments, and the production-engineering bullet).
Those sit in code chips and bash fences where the stats marker protocol
cannot wrap them (an HTML comment inside a code span renders literally), so
this checker pattern-matches every "<N> tests" / "<N> deterministic tests"
claim and compares each against the count pytest actually collects. Runs in
CI right after the suite; stdlib-only.

    python scripts/check_readme_test_count.py             # collects via pytest
    python scripts/check_readme_test_count.py --expect N  # offline
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
# A number followed by lowercase "tests": matches the count surfaces, not the
# V1/V2/V3 milestone table (word-then-numbers) or "Tests" column headers.
COUNT_RE = re.compile(r"\b(\d+) (?:deterministic )?tests\b")
COLLECTED_RE = re.compile(r"(\d+) tests? collected")


def check(readme_text: str, actual: int) -> list[str]:
    """One failure line per README count claim that differs from actual."""
    claims = [int(m) for m in COUNT_RE.findall(readme_text)]
    if not claims:
        return ["no test-count claims found in README (expected at least one)"]
    return [
        f"README claims {claim} tests; the suite collects {actual}"
        for claim in claims
        if claim != actual
    ]


def collected_count() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    match = COLLECTED_RE.search(out.stdout)
    if match is None:
        raise SystemExit(
            f"could not parse pytest collect output:\n{out.stdout[-2000:]}{out.stderr[-500:]}"
        )
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="skip the pytest collect and compare against this count",
    )
    args = parser.parse_args()
    actual = args.expect if args.expect is not None else collected_count()
    failures = check(README.read_text(), actual)
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print(f"OK: all README test-count claims match the collected {actual}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
