from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pym.core.v03 import CoreV03Config, CoreV03State, PYMEngineV03, discrete_memory_work


OUT = Path("artifacts_core_v03")
DEFAULT_DTS = (0.004, 0.002, 0.001)


def harmonic_force(state: CoreV03State, k: float = 1.0) -> np.ndarray:
    return -k * state.position


def mechanical_energy(state: CoreV03State, k: float = 1.0) -> float:
    kinetic = float(np.dot(state.momentum, state.momentum) / (2.0 * state.mass))
    potential = float(0.5 * k * np.dot(state.position, state.position))
    return kinetic + potential


def run_v03(dt: float, T: float, config: CoreV03Config, q0=0.7, p0=-0.1):
    steps = int(round(T / dt))
    if not np.isclose(steps * dt, T, rtol=0.0, atol=1e-12):
        raise ValueError("T must be an integer multiple of dt")
    engine = PYMEngineV03()
    state = CoreV03State([q0], [p0])
    trace = np.empty((steps + 1, 4), dtype=float)
    trace[0] = (0.0, state.position[0], state.momentum[0], state.memory)
    E0 = mechanical_energy(state)
    work_memory = 0.0
    for i in range(steps):
        old = state
        state = engine.step(state, dt, harmonic_force, config)
        work_memory += discrete_memory_work(old, state, config)
        trace[i + 1] = ((i + 1) * dt, state.position[0], state.momentum[0], state.memory)
    E1 = mechanical_energy(state)
    return {
        "dt": dt,
        "steps": steps,
        "trace": trace,
        "endpoint": trace[-1, 1:].tolist(),
        "mechanical_energy_initial": E0,
        "mechanical_energy_final": E1,
        "mechanical_energy_change": E1 - E0,
        "memory_work_discrete": work_memory,
        "work_balance_residual": (E1 - E0) - work_memory,
    }


def common_grid_error(coarse, fine):
    ratio = int(round(coarse["dt"] / fine["dt"]))
    if not np.isclose(ratio * fine["dt"], coarse["dt"], rtol=0.0, atol=1e-15):
        raise ValueError("dt ratios must be integer")
    a = coarse["trace"][:, 1:]
    b = fine["trace"][::ratio, 1:]
    if a.shape != b.shape:
        raise ValueError("common-grid traces do not align")
    d = a - b
    return {
        "rms_qpm": float(np.sqrt(np.mean(d * d))),
        "max_abs_qpm": float(np.max(np.abs(d))),
        "endpoint_l2_qpm": float(np.linalg.norm(d[-1])),
    }


def convergence_report(T=1.0, dts=DEFAULT_DTS):
    cfg = CoreV03Config(tau_M=1.0, lambda_pym=0.05, memory_force_scale=0.1)
    runs = [run_v03(dt, T, cfg) for dt in dts]
    e01 = common_grid_error(runs[0], runs[1])
    e12 = common_grid_error(runs[1], runs[2])
    observed_order = None
    if e01["rms_qpm"] > 0.0 and e12["rms_qpm"] > 0.0:
        observed_order = float(np.log(e01["rms_qpm"] / e12["rms_qpm"]) / np.log(2.0))
    return {
        "experiment": "PYM-Core-v0.3-convergence",
        "T": T,
        "config": {
            "tau_M": cfg.tau_M,
            "lambda_pym": cfg.lambda_pym,
            "memory_force_scale": cfg.memory_force_scale,
            "signal_law": cfg.signal_law.value,
        },
        "dts": list(dts),
        "pair_errors": {
            f"{dts[0]}_vs_{dts[1]}": e01,
            f"{dts[1]}_vs_{dts[2]}": e12,
        },
        "observed_order_rms_qpm": observed_order,
        "energy_work": [
            {k: v for k, v in r.items() if k != "trace"} for r in runs
        ],
        "interpretation_guardrail":
            "Observed order is empirical for this coupled split scheme; it is not assumed to be exactly second order. "
            "Mechanical-energy change is compared with discrete work by the phenomenological memory force, so nonzero "
            "mechanical-energy change is not by itself numerical drift.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    args = ap.parse_args()
    report = convergence_report(T=args.T)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "convergence_v03.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
