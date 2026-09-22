from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pym.physics.hidden_bath import HiddenBathState, step_velocity_verlet, total_energy


@dataclass(frozen=True)
class BathComparison:
    max_divergence: float
    final_divergence: float
    relative_total_energy_drift: float


def run_hidden_bath_pair(steps: int = 2000, dt: float = 0.001, k_ext: float = 1.0, k_bath: float = 0.5) -> BathComparison:
    common = dict(
        q=np.array([1.0, 0.0, 0.0]),
        p=np.array([0.0, 1.0, 0.0]),
        py=np.zeros(3),
    )
    a = HiddenBathState(y=np.array([0.5, 0.0, 0.0]), **common)
    b = HiddenBathState(y=np.zeros(3), **common)
    e0 = total_energy(a, k_ext, k_bath)
    div = []
    for _ in range(steps):
        a = step_velocity_verlet(a, dt, k_ext, k_bath)
        b = step_velocity_verlet(b, dt, k_ext, k_bath)
        div.append(float(np.linalg.norm(a.q - b.q)))
    drift = abs(total_energy(a, k_ext, k_bath) - e0) / (abs(e0) + 1e-15)
    return BathComparison(max(div), div[-1], drift)
