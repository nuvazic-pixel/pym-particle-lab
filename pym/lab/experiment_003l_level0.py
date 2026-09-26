from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pym.core.v03 import CoreV03Config, CoreV03State, PYMEngineV03, SignalLaw
from pym.lab.prehistory import PrepProtocol, prepare_bath, release_bath
from pym.physics.multimode_bath import BathParameters

DT = 0.001
PREP_TIME = 1.0
T_REF = 0.6
K = 20
BASE_SEED = 47000
OMEGA_BOUNDS = (1e-2, 1e3)
COUPLING_BOUNDS = (1e-4, 1e2)
MAXITER = 300
GTOL = 1e-6
FTOL = 1e-9
PENALTY = 1e12
# Level-0 reference capacity is frozen before execution. N=3 is the
# constructively demonstrated 003G legacy reference capacity; 003L does not
# reuse its parameters, targets, sigma values, or epsilon.
REFERENCE_N = 3
OUT = Path("artifacts_003l_level0_calibration")

CASES = (
    ("IC_1", (1.0,), (0.15,), PrepProtocol(0.50, 2.0, 0.0)),
    ("IC_2", (0.8,), (-0.10,), PrepProtocol(0.35, 3.0, 0.4)),
    ("IC_3", (1.2,), (0.05,), PrepProtocol(0.65, 1.5, -0.3)),
)

V03_CONFIG = CoreV03Config(
    tau_M=1.0,
    lambda_pym=0.05,
    memory_force_scale=0.1,
    signal_law=SignalLaw.LINEAR_Q,
)


def harmonic_force(state: CoreV03State) -> np.ndarray:
    return -state.position


def prepare_v03(q0, p0, protocol: PrepProtocol) -> CoreV03State:
    state = CoreV03State(np.asarray(q0, float), np.asarray(p0, float))
    engine = PYMEngineV03()
    steps = int(round(PREP_TIME / DT))
    for k in range(steps):
        t = (k - steps) * DT
        u = protocol.force(t)

        def forced(s: CoreV03State, u=u) -> np.ndarray:
            return -s.position + np.full_like(s.position, u)

        state = engine.step(state, DT, forced, V03_CONFIG)
    return state


def release_v03(state: CoreV03State, T: float = T_REF):
    engine = PYMEngineV03()
    qs, ps = [], []
    for _ in range(int(round(T / DT))):
        state = engine.step(state, DT, harmonic_force, V03_CONFIG)
        qs.append(state.position.copy())
        ps.append(state.momentum.copy())
    return np.asarray(qs), np.asarray(ps)


def targets():
    out = {}
    for name, q0, p0, protocol in CASES:
        s = prepare_v03(q0, p0, protocol)
        q, p = release_v03(s)
        out[name] = (q, p)
    return out


def sigma_target(target) -> float:
    q, p = target
    qc = q - np.mean(q, axis=0)
    pc = p - np.mean(p, axis=0)
    return float(np.sqrt(np.mean(np.sum(qc * qc, axis=1) + np.sum(pc * pc, axis=1))))


def params_from_log(x, n=REFERENCE_N):
    return BathParameters(np.ones(n), np.exp(x[:n]), np.exp(x[n:]))


def random_x0(n, seed):
    rng = np.random.default_rng(seed)
    ob = (np.log(OMEGA_BOUNDS[0]), np.log(OMEGA_BOUNDS[1]))
    cb = (np.log(COUPLING_BOUNDS[0]), np.log(COUPLING_BOUNDS[1]))
    return np.r_[rng.uniform(*ob, n), rng.uniform(*cb, n)]


def raw_loss(params, case, target):
    name, q0, p0, protocol = case
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            b = prepare_bath(params, q0, p0, DT, int(round(PREP_TIME / DT)), protocol)
            if b is None:
                return PENALTY
            q, p = release_bath(b, params, int(round(T_REF / DT)), DT)
            if q is None or not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
                return PENALTY
            tq, tp = target
            value = float(np.mean(np.sum((q - tq) ** 2 + (p - tp) ** 2, axis=1)))
            return value if np.isfinite(value) else PENALTY
    except (FloatingPointError, OverflowError, ValueError):
        return PENALTY


def metrics(params, tgts, sigmas):
    ans = {}
    for case in CASES:
        name = case[0]
        r = raw_loss(params, case, tgts[name])
        ans[name] = {
            "raw_loss": r,
            "sigma_v03_Tref": sigmas[name],
            "nrmse_fixed": float(np.sqrt(r) / sigmas[name]),
        }
    return ans


def optimize_one(seed, tgts, sigmas):
    n = REFERENCE_N
    ob = (np.log(OMEGA_BOUNDS[0]), np.log(OMEGA_BOUNDS[1]))
    cb = (np.log(COUPLING_BOUNDS[0]), np.log(COUPLING_BOUNDS[1]))
    bounds = [ob] * n + [cb] * n
    x0 = random_x0(n, seed)
    train_case = CASES[0]

    def objective(x):
        return raw_loss(params_from_log(x), train_case, tgts["IC_1"])

    res = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": MAXITER, "gtol": GTOL, "ftol": FTOL},
    )
    params = params_from_log(res.x)
    return {
        "seed": seed,
        "success": bool(res.success),
        "nit": int(res.nit),
        "omega": params.omega.tolist(),
        "coupling": params.coupling.tolist(),
        "metrics": metrics(params, tgts, sigmas),
    }


def calibrate():
    tgts = targets()
    sigmas = {name: sigma_target(tgts[name]) for name, *_ in CASES}
    starts = [optimize_one(BASE_SEED + k, tgts, sigmas) for k in range(K)]
    # Selection is TRAIN-only. OOS is evaluated but never used to rank candidates.
    best = min(starts, key=lambda x: x["metrics"]["IC_1"]["raw_loss"])
    a0 = max(best["metrics"][name]["nrmse_fixed"] for name, *_ in CASES)
    epsilon = 1.05 * a0
    result = {
        "experiment": "003L-Level0-calibration",
        "manifest_rule": "epsilon_003L = 1.05 * A0",
        "target": "PYM-Core-v0.3",
        "target_config": {
            "tau_M": V03_CONFIG.tau_M,
            "lambda_pym": V03_CONFIG.lambda_pym,
            "memory_force_scale": V03_CONFIG.memory_force_scale,
            "signal_law": V03_CONFIG.signal_law.value,
        },
        "dt": DT,
        "prep_time": PREP_TIME,
        "T_ref": T_REF,
        "reference_N": REFERENCE_N,
        "K": K,
        "base_seed": BASE_SEED,
        "optimizer": "L-BFGS-B",
        "maxiter": MAXITER,
        "gtol": GTOL,
        "ftol": FTOL,
        "omega_bounds": OMEGA_BOUNDS,
        "coupling_bounds": COUPLING_BOUNDS,
        "sigma_v03_Tref": sigmas,
        "selection_policy": "minimum IC_1 raw loss only; IC_2/IC_3 never rank candidates",
        "A0": a0,
        "epsilon_003L": epsilon,
        "best": best,
        "starts": starts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "level0_calibration.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sigma_v03_Tref": sigmas,
        "reference_N": REFERENCE_N,
        "best_seed": best["seed"],
        "A0": a0,
        "epsilon_003L": epsilon,
        "artifact_path": str(path),
    }, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("calibrate",), default="calibrate")
    ap.parse_args()
    calibrate()
