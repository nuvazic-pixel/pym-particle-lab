from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pym.core.v03 import (
    CoreV03Config, CoreV03State, PYMEngineV03, SignalLaw,
    memory_exact_piecewise_constant, memory_force,
)
from pym.physics.standard import harmonic_force


OUT = Path("artifacts_core_v03")
DTS = (0.004, 0.002, 0.001)


def _legacy_style_J(state: CoreV03State, dt: float) -> float:
    # Reproduce the legacy source definition while keeping the surrounding
    # state/config explicit: J_legacy = ||F_standard|| * dt.
    return float(np.linalg.norm(harmonic_force(state)) * dt)


def _step_euler_exact_memory(state, dt, cfg, signal_mode):
    f_std = harmonic_force(state)
    f_mem = memory_force(state, cfg.memory_force_scale)
    p1 = state.momentum + dt * (f_std + cfg.lambda_pym * f_mem)
    q1 = state.position + dt * p1 / state.mass
    if signal_mode == "LEGACY_FORCE_DT":
        J = _legacy_style_J(state, dt)
    elif signal_mode == "LINEAR_Q":
        J = float(state.position[0])
    else:
        raise ValueError(signal_mode)
    M1 = memory_exact_piecewise_constant(state.memory, J, dt, cfg.tau_M)
    return CoreV03State(q1, p1, state.mass, M1, J)


def _step_symmetric(state, dt, cfg, signal_mode):
    if signal_mode == "LINEAR_Q":
        return PYMEngineV03().step(state, dt, harmonic_force, cfg)

    # Same symmetric q,p ordering as v0.3, but with the legacy source law.
    J0 = _legacy_style_J(state, dt)
    Mh = memory_exact_piecewise_constant(state.memory, J0, 0.5 * dt, cfg.tau_M)
    s0 = CoreV03State(state.position, state.momentum, state.mass, Mh, J0)
    p_half = state.momentum + 0.5 * dt * (
        harmonic_force(s0) + cfg.lambda_pym * memory_force(s0, cfg.memory_force_scale)
    )
    q1 = state.position + dt * p_half / state.mass
    sm = CoreV03State(q1, p_half, state.mass, Mh, J0)
    J1 = _legacy_style_J(sm, dt)
    M1 = memory_exact_piecewise_constant(Mh, J1, 0.5 * dt, cfg.tau_M)
    s1 = CoreV03State(q1, p_half, state.mass, M1, J1)
    p1 = p_half + 0.5 * dt * (
        harmonic_force(s1) + cfg.lambda_pym * memory_force(s1, cfg.memory_force_scale)
    )
    return CoreV03State(q1, p1, state.mass, M1, J1)


def _run(dt, T, integrator, signal_mode):
    cfg = CoreV03Config(tau_M=1.0, lambda_pym=0.05, memory_force_scale=0.1,
                        signal_law=SignalLaw.LINEAR_Q)
    s = CoreV03State([0.7], [-0.1])
    n = int(round(T / dt))
    tr = np.empty((n + 1, 3))
    tr[0] = (s.position[0], s.momentum[0], s.memory)
    for k in range(n):
        if integrator == "EULER_CROMER":
            s = _step_euler_exact_memory(s, dt, cfg, signal_mode)
        elif integrator == "SYMMETRIC":
            s = _step_symmetric(s, dt, cfg, signal_mode)
        else:
            raise ValueError(integrator)
        tr[k + 1] = (s.position[0], s.momentum[0], s.memory)
    return tr


def _metrics(a, b):
    d = a - b
    return {
        "rms_qp": float(np.sqrt(np.mean(d[:, :2] ** 2))),
        "rms_qpm": float(np.sqrt(np.mean(d ** 2))),
        "endpoint_qp": float(np.linalg.norm(d[-1, :2])),
        "endpoint_memory": float(abs(d[-1, 2])),
    }


def controls(T=1.0, dts=DTS):
    integrator_only = []
    model_definition = []
    for dt in dts:
        # CONTROL A: hold J=LINEAR_Q and the exact memory recurrence fixed.
        # Only q,p integration ordering changes.
        ec_q = _run(dt, T, "EULER_CROMER", "LINEAR_Q")
        sy_q = _run(dt, T, "SYMMETRIC", "LINEAR_Q")
        integrator_only.append({"dt": dt, **_metrics(ec_q, sy_q)})

        # CONTROL B: hold the symmetric integrator and memory recurrence fixed.
        # Only the source definition changes.
        sy_legacy_j = _run(dt, T, "SYMMETRIC", "LEGACY_FORCE_DT")
        model_definition.append({"dt": dt, **_metrics(sy_legacy_j, sy_q)})

    return {
        "experiment": "PYM-Core-v0.3-isolation-controls",
        "T": T,
        "dts": list(dts),
        "control_A_integrator_only": {
            "fixed": ["J=LINEAR_Q", "tau_M=1", "exact piecewise-constant memory recurrence",
                      "lambda_pym=0.05", "memory_force_scale=0.1"],
            "varied": "Euler-Cromer ordering vs symmetric kick-drift-kick",
            "rows": integrator_only,
        },
        "control_B_model_definition": {
            "fixed": ["symmetric kick-drift-kick", "tau_M=1",
                      "exact piecewise-constant memory recurrence",
                      "lambda_pym=0.05", "memory_force_scale=0.1"],
            "varied": "J=||F_standard||*dt vs J=q[0]",
            "rows": model_definition,
        },
        "guardrail":
            "Control B deliberately preserves the legacy dt factor inside J. Because that source itself changes "
            "with resolution, its dt->0 behavior is part of the legacy model definition being diagnosed. "
            "Neither control alone establishes physical equivalence or a universal continuous limit.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    a = ap.parse_args()
    result = controls(a.T)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "isolation_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
