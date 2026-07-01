"""Unit tests for the reveal placeholder substitution: display formatting and
geometry derivation, in isolation from the real index.html template."""
from __future__ import annotations

from agent_bench.serving.routes import _apply_reveal_placeholders, _get_reveal_anchor

_TEMPLATE = (
    "MODEL={{REVEAL_MODEL}} AL={{REVEAL_A_LABEL}} A={{REVEAL_A_P5}} "
    "ACI={{REVEAL_A_CI}} BL={{REVEAL_B_LABEL}} B={{REVEAL_B_P5}} "
    "BCI={{REVEAL_B_CI}} DIFF={{REVEAL_DIFF}} CI90={{REVEAL_DIFF_CI90}} "
    "TOST={{REVEAL_TOST}} SUP={{REVEAL_SUPPORT}} R={{REVEAL_RDIFF}} "
    "CA={{REVEAL_COST_A}} CB={{REVEAL_COST_B}} RATIO={{REVEAL_COST_RATIO}} "
    "FA={{REVEAL_FLOOR_API}} F7={{REVEAL_FLOOR_7B}} GEOM=[{{REVEAL_GEOM}}]"
)


def test_all_placeholders_filled_with_committed_values():
    out = _apply_reveal_placeholders(_TEMPLATE, _get_reveal_anchor())
    assert "{{REVEAL_" not in out
    assert "A=0.791" in out
    assert "B=0.760" in out
    assert "ACI=[0.664, 0.917]" in out
    assert "DIFF=+0.031" in out
    assert "CI90=[-0.013, +0.076]" in out
    assert "TOST=equivalent" in out
    assert "SUP=0.076" in out
    assert "R=+0.000" in out
    assert "CA=$0.0007" in out
    assert "CB=$0.0046" in out
    assert "RATIO=6.6x" in out
    assert "FA=1.00" in out
    assert "F7=0.14" in out


def test_geometry_is_derived_from_values():
    out = _apply_reveal_placeholders(_TEMPLATE, _get_reveal_anchor())
    # 0.791 -> 79.1%, 0.664 -> 66.4%, 1.00 -> 100.0%, 0.14 -> 14.0%
    assert "--a-p5:79.1%" in out
    assert "--a-lo:66.4%" in out
    assert "--b-hi:87.5%" in out
    assert "--f-api:100.0%" in out
    assert "--f-sh:14.0%" in out
