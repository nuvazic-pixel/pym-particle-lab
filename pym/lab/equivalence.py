from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from pym.physics.multimode_bath import BathParameters, simulate


@dataclass(frozen=True)
class FitResult:
    n_modes: int
    train_loss: float
    params: BathParameters
    success: bool


def unpack(log_params: np.ndarray, n_modes: int) -> BathParameters:
    # Positive parameters by construction; log-space also improves optimizer scaling.
    raw = np.exp(np.clip(log_params, -12.0, 12.0))
    return BathParameters(raw[:n_modes], raw[n_modes:2*n_modes], raw[2*n_modes:3*n_modes])


def trajectory_loss(params: BathParameters, target_q: np.ndarray, target_p: np.ndarray, q0: np.ndarray, p0: np.ndarray, dt: float, beta: float = 1.0) -> float:
    q, p, _ = simulate(params, q0, p0, len(target_q), dt)
    dq = q - target_q
    dp = p - target_p
    return float(np.mean(np.sum(dq*dq, axis=1) + beta*np.sum(dp*dp, axis=1)))


def fit_bath(target_q: np.ndarray, target_p: np.ndarray, q0: np.ndarray, p0: np.ndarray, dt: float, n_modes: int, beta: float = 1.0, maxiter: int = 300) -> FitResult:
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    # Deterministic initialization: masses=1, log-spaced frequencies, weak couplings.
    x0 = np.concatenate([
        np.zeros(n_modes),
        np.log(np.geomspace(0.25, 4.0, n_modes)),
        np.full(n_modes, np.log(0.1)),
    ])

    def objective(x):
        return trajectory_loss(unpack(x, n_modes), target_q, target_p, q0, p0, dt, beta)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    return FitResult(n_modes, float(result.fun), unpack(result.x, n_modes), bool(result.success))


def oos_loss(fit: FitResult, target_q: np.ndarray, target_p: np.ndarray, q0: np.ndarray, p0: np.ndarray, dt: float, beta: float = 1.0) -> float:
    # Critical rule: NO refitting on the test initial condition.
    return trajectory_loss(fit.params, target_q, target_p, q0, p0, dt, beta)
