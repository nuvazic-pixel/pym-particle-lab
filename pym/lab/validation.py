from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pym.core.state import PYMState


@dataclass(frozen=True)
class ValidationReport:
    baseline_position_divergence: float
    baseline_momentum_divergence: float
    finite: bool
    baseline_pass: bool


def validate_baseline(a: PYMState, b: PYMState, tolerance: float = 1e-12) -> ValidationReport:
    dx = float(np.linalg.norm(a.position - b.position))
    dp = float(np.linalg.norm(a.momentum - b.momentum))
    finite = bool(
        np.all(np.isfinite(a.position))
        and np.all(np.isfinite(b.position))
        and np.all(np.isfinite(a.momentum))
        and np.all(np.isfinite(b.momentum))
    )
    return ValidationReport(dx, dp, finite, finite and dx < tolerance and dp < tolerance)
