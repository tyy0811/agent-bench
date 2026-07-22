"""Deploy preflight: refuse to ship unresolved placeholder tokens.

The repo deliberately carries BOOKING_URL and IMPRESSUM_* placeholder tokens
on public dashboard surfaces until Jane fills the real values by hand (legal
details and the booking target are not inventable). CI stays green with the
tokens in place; what must never happen is a deploy of the HF Space while a
token is unresolved. Run this before any push to the hf remote:

    make preflight-deploy

Exits 1 listing file:line for every unresolved token. Stdlib-only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "agent_bench" / "serving" / "static"
TOKENS = ("BOOKING_URL", "IMPRESSUM_NAME", "IMPRESSUM_ADDRESS", "IMPRESSUM_CONTACT")


def scan(static_dir: Path) -> list[str]:
    """Return one failure line per unresolved token occurrence in *.html."""
    failures = []
    for path in sorted(static_dir.glob("*.html")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for token in TOKENS:
                if token in line:
                    failures.append(f"{path.name}:{lineno}: unresolved {token}")
    return failures


def main() -> int:
    failures = scan(STATIC_DIR)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"{len(failures)} unresolved placeholder token(s); do not deploy.")
        return 1
    print("OK: no unresolved placeholder tokens on deploy surfaces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
