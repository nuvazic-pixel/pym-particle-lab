from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pym.lab.experiment_003l_level0 import (
    BASE_SEED, CASES, COUPLING_BOUNDS, DT, FTOL, GTOL, K, MAXITER,
    OMEGA_BOUNDS, PENALTY, PREP_TIME, V03_CONFIG, prepare_v03, release_v03,
)
from pym.physics.damped_multimode_bath import DampedBathParameters, step_damped_bath
from pym.physics.multimode_bath import MultiModeState

HORIZONS = (0.3, 0.6, 1.0, 2.0, 3.0, 5.0)
CAPACITIES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
GAMMA_BOUNDS = (1e-3, 1e2)
SIGMA_V03 = {
    "IC_1": 0.196199716029399,
    "IC_2": 0.15779818649314112,
    "IC_3": 0.2509655101723989,
}
A0 = 0.004734304658737562
EPSILON_003L = 0.00497101989167444
OUT = Path("artifacts_003l_level1")


def horizon_token(T: float) -> str:
    """Canonical filename token: seconds -> integer milliseconds; no decimal paths."""
    ms = int(round(T * 1000.0))
    if not np.isclose(T, ms / 1000.0, rtol=0.0, atol=1e-12):
        raise ValueError("horizon must map exactly to integer milliseconds")
    return f"T{ms:04d}ms"


def artifact_path(T: float, n: int) -> Path:
    return OUT / f"cell_{horizon_token(T)}_N{n:02d}.json"


def targets_for(T: float):
    out = {}
    for name, q0, p0, protocol in CASES:
        s = prepare_v03(q0, p0, protocol)
        out[name] = release_v03(s, T=T)
    return out


def params_from_log(x, n: int) -> DampedBathParameters:
    x = np.asarray(x, float)
    if x.shape != (3 * n,):
        raise ValueError("Level-1 parameter vector must contain omega,coupling,gamma blocks")
    return DampedBathParameters(
        mass=np.ones(n),
        omega=np.exp(x[:n]),
        coupling=np.exp(x[n:2*n]),
        gamma=np.exp(x[2*n:]),
    )


def random_x0(n: int, seed: int):
    """Frozen RNG draw order: log-omega, log-coupling, log-gamma."""
    rng = np.random.default_rng(seed)
    ob = tuple(np.log(OMEGA_BOUNDS))
    cb = tuple(np.log(COUPLING_BOUNDS))
    gb = tuple(np.log(GAMMA_BOUNDS))
    return np.r_[
        rng.uniform(*ob, n),
        rng.uniform(*cb, n),
        rng.uniform(*gb, n),
    ]


def _finite_state(s: MultiModeState) -> bool:
    return all(np.all(np.isfinite(a)) for a in (s.q, s.p, s.y, s.py))


def prepare_level1(params: DampedBathParameters, q0, p0, protocol):
    state = MultiModeState(
        np.asarray(q0, float).copy(),
        np.asarray(p0, float).copy(),
        np.zeros((params.n_modes, len(q0))),
        np.zeros((params.n_modes, len(q0))),
        1.0,
    )
    prep_steps = int(round(PREP_TIME / DT))
    # External preparation force acts on q only. Reproduce the Level-1 Strang
    # damping split while adding u(t) consistently to the two q half-kicks.
    from pym.physics.damped_multimode_bath import _damping_half_step
    from pym.physics.multimode_bath import forces

    conservative = params.conservative()
    for k in range(prep_steps):
        t = (k - prep_steps) * DT
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                s = state.clone()
                s.py = _damping_half_step(s.py, params.gamma, 0.5 * DT)
                fq, fy = forces(s, conservative)
                fq = fq + protocol.force(t)
                p_half = s.p + 0.5 * DT * fq
                py_half = s.py + 0.5 * DT * fy
                q_new = s.q + DT * p_half / s.particle_mass
                y_new = s.y + DT * py_half / conservative.mass[:, None]
                trial = MultiModeState(q_new, p_half, y_new, py_half, s.particle_mass)
                fq2, fy2 = forces(trial, conservative)
                fq2 = fq2 + protocol.force(t + DT)
                state = MultiModeState(
                    q_new,
                    p_half + 0.5 * DT * fq2,
                    y_new,
                    py_half + 0.5 * DT * fy2,
                    s.particle_mass,
                )
                state.py = _damping_half_step(state.py, params.gamma, 0.5 * DT)
                if not _finite_state(state):
                    return None
        except (FloatingPointError, OverflowError, ValueError):
            return None
    return state


def release_level1(state, params: DampedBathParameters, T: float):
    if state is None:
        return None, None
    qs, ps = [], []
    for _ in range(int(round(T / DT))):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                state = step_damped_bath(state, params, DT)
        except (FloatingPointError, OverflowError, ValueError):
            return None, None
        if not _finite_state(state):
            return None, None
        qs.append(state.q.copy())
        ps.append(state.p.copy())
    return np.asarray(qs), np.asarray(ps)


def raw_loss(params, case, target, T: float) -> float:
    _, q0, p0, protocol = case
    state = prepare_level1(params, q0, p0, protocol)
    q, p = release_level1(state, params, T)
    if q is None:
        return PENALTY
    tq, tp = target
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = float(np.mean(np.sum((q - tq) ** 2 + (p - tp) ** 2, axis=1)))
    except FloatingPointError:
        return PENALTY
    return value if np.isfinite(value) else PENALTY


def metrics(params, T: float, tgts):
    ans = {}
    for case in CASES:
        name = case[0]
        r = raw_loss(params, case, tgts[name], T)
        ans[name] = {
            "raw_loss": r,
            "sigma_v03_Tref": SIGMA_V03[name],
            "nrmse_fixed": float(np.sqrt(r) / SIGMA_V03[name]),
        }
    return ans


def is_pass(m) -> bool:
    return all(m[name]["nrmse_fixed"] <= EPSILON_003L for name in SIGMA_V03)


def optimize_start(n: int, T: float, tgts, seed: int):
    ob = tuple(np.log(OMEGA_BOUNDS))
    cb = tuple(np.log(COUPLING_BOUNDS))
    gb = tuple(np.log(GAMMA_BOUNDS))
    bounds = [ob] * n + [cb] * n + [gb] * n
    train = CASES[0]

    def objective(x):
        return raw_loss(params_from_log(x, n), train, tgts["IC_1"], T)

    res = minimize(
        objective, random_x0(n, seed), method="L-BFGS-B", bounds=bounds,
        options={"maxiter": MAXITER, "gtol": GTOL, "ftol": FTOL},
    )
    params = params_from_log(res.x, n)
    m = metrics(params, T, tgts)
    return {
        "seed": seed,
        "optimizer_success": bool(res.success),
        "nit": int(res.nit),
        "fun_train": float(res.fun),
        "omega": params.omega.tolist(),
        "coupling": params.coupling.tolist(),
        "gamma": params.gamma.tolist(),
        "metrics": m,
        "pass": is_pass(m),
    }


def run_cell(T: float, n: int):
    if T not in HORIZONS:
        raise ValueError(f"T must be one of {HORIZONS}")
    if n not in CAPACITIES:
        raise ValueError(f"N must be one of {CAPACITIES}")

    tgts = targets_for(T)
    starts = [optimize_start(n, T, tgts, BASE_SEED + k) for k in range(K)]
    # Frozen selection: TRAIN raw loss only.
    best = min(starts, key=lambda r: r["metrics"]["IC_1"]["raw_loss"])
    result = {
        "experiment": "003L-Level1-per-cell-v1",
        "T": T,
        "horizon_token": horizon_token(T),
        "N": n,
        "target": "PYM-Core-v0.3",
        "target_config": {
            "tau_M": V03_CONFIG.tau_M,
            "lambda_pym": V03_CONFIG.lambda_pym,
            "memory_force_scale": V03_CONFIG.memory_force_scale,
            "signal_law": V03_CONFIG.signal_law.value,
        },
        "dt": DT,
        "prep_time": PREP_TIME,
        "K": K,
        "base_seed": BASE_SEED,
        "optimizer": "L-BFGS-B",
        "maxiter": MAXITER,
        "gtol": GTOL,
        "ftol": FTOL,
        "omega_bounds": OMEGA_BOUNDS,
        "coupling_bounds": COUPLING_BOUNDS,
        "gamma_bounds": GAMMA_BOUNDS,
        "sigma_v03_Tref": SIGMA_V03,
        "A0": A0,
        "epsilon_003L": EPSILON_003L,
        "selection_policy": "minimum IC_1 raw loss only; OOS never ranks starts",
        "best": best,
        "starts": starts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = artifact_path(T, n)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "T": T, "N": n, "best_seed": best["seed"], "pass": best["pass"],
        "train_raw_loss": best["metrics"]["IC_1"]["raw_loss"],
        "max_nrmse": max(best["metrics"][name]["nrmse_fixed"] for name in SIGMA_V03),
        "artifact_path": str(path),
    }, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    args = ap.parse_args()
    run_cell(args.T, args.N)
