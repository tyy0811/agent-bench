"""Generate sklearn-parity fixtures for tests/evaluation/test_calibration_metrics.py.

Run from a venv with sklearn installed (NOT the project venv):

    python -m venv /tmp/sklearn-fixture-venv
    /tmp/sklearn-fixture-venv/bin/pip install scikit-learn==1.5.2
    /tmp/sklearn-fixture-venv/bin/python scripts/_dev/generate_kappa_fixtures.py

The script:
  1. Defines CASES (input arrays + weight option).
  2. Computes sklearn.metrics.cohen_kappa_score for each case.
  3. Prints copy-pasteable Python constants for the test file.
  4. Writes inputs to tests/evaluation/fixtures/sklearn_kappa_inputs.json
     for the cross-check CI test (forgot-to-regenerate detection).

DO NOT add scikit-learn to the project's runtime dependencies — these
constants are the contract; the project hand-rolls the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError as e:
    raise SystemExit(
        "scikit-learn not installed. Install in a venv outside this project:\n"
        "  python -m venv /tmp/sklearn-fixture-venv\n"
        "  /tmp/sklearn-fixture-venv/bin/pip install scikit-learn==1.5.2\n"
        "  /tmp/sklearn-fixture-venv/bin/python scripts/_dev/generate_kappa_fixtures.py"
    ) from e

CASES: list[dict] = [
    {
        "name": "imbalanced_binary",
        "y1": [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        "y2": [1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        "weights": None,
    },
    {
        "name": "three_point_one_diagonal_swap",
        "y1": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
        "y2": [0, 1, 1, 1, 2, 2, 0, 1, 2, 0],
        "weights": None,
    },
    {
        "name": "weighted_ordinal_drift_linear",
        "y1": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "y2": [0, 1, 2, 1, 1, 2, 0, 2, 2, 1],
        "weights": "linear",
    },
]

OUT_INPUTS = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "evaluation"
    / "fixtures"
    / "sklearn_kappa_inputs.json"
)

print("# --- Paste into test_calibration_metrics.py ---\n")
print("SKLEARN_KAPPA_FIXTURES: dict[str, float] = {")
for case in CASES:
    expected = cohen_kappa_score(case["y1"], case["y2"], weights=case["weights"])
    print(f'    "{case["name"]}": {expected:.10f},  # sklearn 1.5.2')
print("}")

print("\nSKLEARN_KAPPA_INPUTS: dict[str, dict] = {")
for case in CASES:
    print(f'    "{case["name"]}": {{')
    print(f'        "y1": {case["y1"]},')
    print(f'        "y2": {case["y2"]},')
    print(f'        "weights": {case["weights"]!r},')
    print("    },")
print("}")

OUT_INPUTS.parent.mkdir(parents=True, exist_ok=True)
OUT_INPUTS.write_text(
    json.dumps(
        {
            case["name"]: {
                "y1": case["y1"],
                "y2": case["y2"],
                "weights": case["weights"],
            }
            for case in CASES
        },
        indent=2,
    )
)
print(f"\n# Wrote {OUT_INPUTS}")
