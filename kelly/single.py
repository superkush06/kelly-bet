"""Single-bet Kelly criterion."""

from __future__ import annotations

import math


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Optimal fraction of bankroll to stake on a single binary bet.

    f* = (b p - q) / b, with b = decimal_odds - 1 and q = 1 - p.
    Returns 0 when there is no edge (f* <= 0) — never stake a -EV bet.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be > 1")
    b = decimal_odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def fractional_kelly(p: float, decimal_odds: float, fraction: float = 0.5) -> float:
    """Scale the full-Kelly stake by `fraction` (e.g. 0.5 = half Kelly).

    Fractional Kelly sacrifices a little long-run growth for a large
    reduction in variance and drawdown — the standard practitioner choice.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return fraction * kelly_fraction(p, decimal_odds)


def expected_log_growth(f: float, p: float, decimal_odds: float) -> float:
    """Expected log-growth per bet at stake fraction f.

    g(f) = p ln(1 + f(d-1)) + (1-p) ln(1 - f). Maximised at f = kelly_fraction.
    """
    b = decimal_odds - 1.0
    if f < 0 or f >= 1:
        return -math.inf
    win = 1.0 + f * b
    lose = 1.0 - f
    if win <= 0 or lose <= 0:
        return -math.inf
    return p * math.log(win) + (1.0 - p) * math.log(lose)


def edge(p: float, decimal_odds: float) -> float:
    """Edge = p * d - 1. Positive means +EV."""
    return p * decimal_odds - 1.0
