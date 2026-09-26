from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
import numpy as np


class SignalLaw(str, Enum):
    LINEAR_Q = "LINEAR_Q"
    LINEAR_P = "LINEAR_P"
    NONLINEAR_Q_DOT_P = "NONLINEAR_Q_DOT_P"
    Q_SQUARED = "Q_SQUARED"


@dataclass(frozen=True)
class CoreV03Config:
    tau_M: float = 1.0
    lambda_pym: float = 0.05
    memory_force_scale: float = 0.1
    signal_law: SignalLaw = SignalLaw.LINEAR_Q

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau_M) or self.tau_M <= 0.0:
            raise ValueError("tau_M must be finite and positive")
        if not np.isfinite(self.lambda_pym):
            raise ValueError("lambda_pym must be finite")
        if not np.isfinite(self.memory_force_scale):
            raise ValueError("memory_force_scale must be finite")


@dataclass
class CoreV03State:
    position: np.ndarray
    momentum: np.ndarray
    mass: float = 1.0
    memory: float = 0.0
    interaction_signal: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).copy()
        self.momentum = np.asarray(self.momentum, dtype=float).copy()
        if self.position.shape != self.momentum.shape:
            raise ValueError("position and momentum must have identical shapes")
        if not np.all(np.isfinite(self.position)) or not np.all(np.isfinite(self.momentum)):
            raise ValueError("state arrays must be finite")
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be finite and positive")
        if not np.isfinite(self.memory):
            raise ValueError("memory must be finite")

    def clone(self) -> "CoreV03State":
        return CoreV03State(self.position.copy(), self.momentum.copy(), self.mass,
                            self.memory, self.interaction_signal)


StandardForceFn = Callable[[CoreV03State], np.ndarray]


def signal(state: CoreV03State, law: SignalLaw) -> float:
    q, p = state.position, state.momentum
    if q.size == 0:
        return 0.0
    if law is SignalLaw.LINEAR_Q:
        return float(q[0])
    if law is SignalLaw.LINEAR_P:
        return float(p[0])
    if law is SignalLaw.NONLINEAR_Q_DOT_P:
        return float(np.dot(q, p))
    if law is SignalLaw.Q_SQUARED:
        return float(np.dot(q, q))
    raise ValueError(f"unsupported signal law: {law}")


def memory_exact_piecewise_constant(memory: float, J: float, dt: float, tau_M: float) -> float:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if tau_M <= 0.0:
        raise ValueError("tau_M must be positive")
    a = float(np.exp(-dt / tau_M))
    return float(a * memory + (1.0 - a) * J)


def memory_force(state: CoreV03State, scale: float) -> np.ndarray:
    out = np.zeros_like(state.position)
    if out.size:
        out[0] = scale * state.memory
    return out


class PYMEngineV03:
    """Symmetric split integrator with exact piecewise-constant memory substeps."""

    @staticmethod
    def _checked_force(force: StandardForceFn, state: CoreV03State) -> np.ndarray:
        value = np.asarray(force(state), dtype=float)
        if value.shape != state.position.shape:
            raise ValueError("force shape must match state shape")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("non-finite force")
        return value

    def step(self, state: CoreV03State, dt: float, standard_force: StandardForceFn,
             config: CoreV03Config) -> CoreV03State:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        J0 = signal(state, config.signal_law)
        Mhalf = memory_exact_piecewise_constant(state.memory, J0, 0.5 * dt, config.tau_M)

        old = state.clone()
        old.memory = Mhalf
        f0 = self._checked_force(standard_force, old)
        fp0 = memory_force(old, config.memory_force_scale)
        p_half = state.momentum + 0.5 * dt * (f0 + config.lambda_pym * fp0)
        q1 = state.position + dt * p_half / state.mass

        mid = CoreV03State(q1, p_half, state.mass, Mhalf, J0)
        J1 = signal(mid, config.signal_law)
        M1 = memory_exact_piecewise_constant(Mhalf, J1, 0.5 * dt, config.tau_M)

        new_for_force = CoreV03State(q1, p_half, state.mass, M1, J1)
        f1 = self._checked_force(standard_force, new_for_force)
        fp1 = memory_force(new_for_force, config.memory_force_scale)
        p1 = p_half + 0.5 * dt * (f1 + config.lambda_pym * fp1)

        return CoreV03State(q1, p1, state.mass, M1, J1)


def discrete_memory_work(state0: CoreV03State, state1: CoreV03State,
                         config: CoreV03Config) -> float:
    """Trapezoidal force-dot-displacement diagnostic for the memory force."""
    f0 = config.lambda_pym * memory_force(state0, config.memory_force_scale)
    f1 = config.lambda_pym * memory_force(state1, config.memory_force_scale)
    dq = state1.position - state0.position
    return float(np.dot(0.5 * (f0 + f1), dq))
