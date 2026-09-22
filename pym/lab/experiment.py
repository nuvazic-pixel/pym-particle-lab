from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pym.core.engine import PYMEngine
from pym.core.state import PYMState
from pym.lab.validation import ValidationReport, validate_baseline
from pym.physics.pym_forces import memory_force
from pym.physics.standard import harmonic_energy, harmonic_force


@dataclass(frozen=True)
class ExperimentResult:
    control: ValidationReport
    pym_position_divergence: float
    pym_momentum_divergence: float
    control_energy_drift_a: float
    pym_energy_drift_a: float


def _state(memory: float, lambda_pym: float) -> PYMState:
    return PYMState(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 0.0]),
        memory=memory,
        lambda_pym=lambda_pym,
        alpha=0.95,
    )


def _run_pair(a: PYMState, b: PYMState, steps: int, dt: float):
    engine = PYMEngine()
    e0a = harmonic_energy(a)
    for _ in range(steps):
        a = engine.step(a, dt, harmonic_force, memory_force)
        b = engine.step(b, dt, harmonic_force, memory_force)
    return a, b, harmonic_energy(a) - e0a


def run_double_history_experiment(steps: int = 1000, dt: float = 0.01, lambda_pym: float = 0.1) -> ExperimentResult:
    control_a, control_b, control_drift = _run_pair(_state(5.0, 0.0), _state(0.0, 0.0), steps, dt)
    report = validate_baseline(control_a, control_b)
    if not report.baseline_pass:
        raise RuntimeError("INVALID experiment: lambda=0 baseline diverged")

    pym_a, pym_b, pym_drift = _run_pair(_state(5.0, lambda_pym), _state(0.0, lambda_pym), steps, dt)

    return ExperimentResult(
        control=report,
        pym_position_divergence=float(np.linalg.norm(pym_a.position - pym_b.position)),
        pym_momentum_divergence=float(np.linalg.norm(pym_a.momentum - pym_b.momentum)),
        control_energy_drift_a=float(control_drift),
        pym_energy_drift_a=float(pym_drift),
    )
