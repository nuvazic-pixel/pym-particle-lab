from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pym.physics.multimode_bath import (
    BathParameters,
    MultiModeState,
    forces,
    total_energy,
    step_velocity_verlet,
)


@dataclass(frozen=True)
class DampedBathParameters:
    """Level-1 linear bath: Level-0 parameters plus gamma_i >= 0."""

    mass: np.ndarray
    omega: np.ndarray
    coupling: np.ndarray
    gamma: np.ndarray

    def __post_init__(self) -> None:
        mass = np.asarray(self.mass, dtype=float)
        omega = np.asarray(self.omega, dtype=float)
        coupling = np.asarray(self.coupling, dtype=float)
        gamma = np.asarray(self.gamma, dtype=float)
        if not (mass.shape == omega.shape == coupling.shape == gamma.shape):
            raise ValueError("mass, omega, coupling and gamma must have identical shapes")
        if mass.ndim != 1 or mass.size == 0:
            raise ValueError("Level-1 parameters must be non-empty 1D arrays")
        if not all(np.all(np.isfinite(x)) for x in (mass, omega, coupling, gamma)):
            raise ValueError("Level-1 parameters must be finite")
        if np.any(mass <= 0.0) or np.any(omega <= 0.0) or np.any(coupling <= 0.0):
            raise ValueError("mass, omega and coupling must be positive")
        if np.any(gamma < 0.0):
            raise ValueError("gamma must be non-negative")
        object.__setattr__(self, "mass", mass.copy())
        object.__setattr__(self, "omega", omega.copy())
        object.__setattr__(self, "coupling", coupling.copy())
        object.__setattr__(self, "gamma", gamma.copy())

    @property
    def n_modes(self) -> int:
        return len(self.mass)

    def conservative(self) -> BathParameters:
        return BathParameters(self.mass.copy(), self.omega.copy(), self.coupling.copy())


def _damping_half_step(py: np.ndarray, gamma: np.ndarray, dt: float) -> np.ndarray:
    """Exact half-step for p_y_dot = -2 gamma p_y."""
    return py * np.exp(-gamma[:, None] * dt)


def step_damped_bath(
    state: MultiModeState,
    params: DampedBathParameters,
    dt: float,
    k_ext: float = 1.0,
) -> MultiModeState:
    """Strang split: exact damping half-step around the Level-0 VV step.

    gamma == 0 deliberately delegates to the exact Level-0 implementation,
    making the preregistered reduction gate structural rather than approximate.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    conservative = params.conservative()
    if np.all(params.gamma == 0.0):
        return step_velocity_verlet(state, conservative, dt, k_ext)

    s = state.clone()
    s.py = _damping_half_step(s.py, params.gamma, 0.5 * dt)
    s = step_velocity_verlet(s, conservative, dt, k_ext)
    s.py = _damping_half_step(s.py, params.gamma, 0.5 * dt)
    return s


def instantaneous_dissipation_power(
    state: MultiModeState, params: DampedBathParameters
) -> float:
    """Reservoir power implied by p_y_dot=-2 gamma p_y.

    P_res = sum_i 2 gamma_i |p_y_i|^2 / m_i >= 0.
    """
    per_mode = 2.0 * params.gamma * np.sum(state.py * state.py, axis=1) / params.mass
    return float(np.sum(per_mode))


def step_with_reservoir_accounting(
    state: MultiModeState,
    params: DampedBathParameters,
    dt: float,
    reservoir_energy: float = 0.0,
    k_ext: float = 1.0,
):
    """Advance one Level-1 step and accumulate dissipated energy diagnostically.

    Reservoir increment uses trapezoidal power quadrature. This accounting is a
    numerical diagnostic and is convergence-tested; it is not asserted exact.
    """
    p0 = instantaneous_dissipation_power(state, params)
    next_state = step_damped_bath(state, params, dt, k_ext)
    p1 = instantaneous_dissipation_power(next_state, params)
    reservoir_next = float(reservoir_energy + 0.5 * dt * (p0 + p1))
    mechanical_bath_energy = total_energy(next_state, params.conservative(), k_ext)
    return next_state, reservoir_next, mechanical_bath_energy
