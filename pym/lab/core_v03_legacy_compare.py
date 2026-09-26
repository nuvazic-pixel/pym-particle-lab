from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pym.core.engine import PYMEngine
from pym.core.state import PYMState
from pym.core.v03 import CoreV03Config, CoreV03State, PYMEngineV03, SignalLaw
from pym.physics.pym_forces import memory_force as legacy_memory_force
from pym.physics.standard import harmonic_force


OUT = Path("artifacts_core_v03")
DEFAULT_DTS = (0.004, 0.002, 0.001)


def run_legacy(dt: float, T: float, q0=0.7, p0=-0.1, lambda_pym=0.05, tau_M=1.0):
    steps = int(round(T / dt))
    state = PYMState(
        position=np.array([q0]), momentum=np.array([p0]), memory=0.0,
        lambda_pym=lambda_pym, alpha=float(np.exp(-dt / tau_M)),
    )
    engine = PYMEngine()
    trace = np.empty((steps + 1, 3))
    trace[0] = (state.position[0], state.momentum[0], state.memory)
    for k in range(steps):
        state = engine.step(state, dt, harmonic_force, legacy_memory_force)
        trace[k + 1] = (state.position[0], state.momentum[0], state.memory)
    return trace


def run_v03(dt: float, T: float, q0=0.7, p0=-0.1, lambda_pym=0.05, tau_M=1.0):
    steps = int(round(T / dt))
    state = CoreV03State([q0], [p0], memory=0.0)
    config = CoreV03Config(
        tau_M=tau_M, lambda_pym=lambda_pym, memory_force_scale=0.1,
        signal_law=SignalLaw.LINEAR_Q,
    )
    engine = PYMEngineV03()
    trace = np.empty((steps + 1, 3))
    trace[0] = (state.position[0], state.momentum[0], state.memory)
    for k in range(steps):
        state = engine.step(state, dt, harmonic_force, config)
        trace[k + 1] = (state.position[0], state.momentum[0], state.memory)
    return trace


def discrepancy(a, b):
    d = a - b
    return {
        "rms_qpm": float(np.sqrt(np.mean(d * d))),
        "rms_qp": float(np.sqrt(np.mean(d[:, :2] * d[:, :2]))),
        "endpoint_l2_qpm": float(np.linalg.norm(d[-1])),
        "endpoint_l2_qp": float(np.linalg.norm(d[-1, :2])),
        "endpoint_abs_memory": float(abs(d[-1, 2])),
    }


def report(T=1.0, dts=DEFAULT_DTS):
    rows = []
    for dt in dts:
        legacy = run_legacy(dt, T)
        v03 = run_v03(dt, T)
        rows.append({"dt": dt, **discrepancy(legacy, v03)})

    qp = [r["rms_qp"] for r in rows]
    qpm = [r["rms_qpm"] for r in rows]

    def ratios(values):
        return [
            float(values[i] / values[i + 1]) if values[i + 1] > 0 else None
            for i in range(len(values) - 1)
        ]

    qp_ratios = ratios(qp)
    qpm_ratios = ratios(qpm)

    # This gate deliberately does not infer a mathematical dt->0 limit from
    # three finite resolutions. It classifies only the observed refinement trend.
    if qp[-1] < qp[0] and qp_ratios[0] > 1.0 and qp_ratios[1] > 1.0:
        trend = "DECREASING_OVER_TESTED_REFINEMENT"
    elif qp[-1] > qp[0]:
        trend = "NONDECREASING_OR_DIVERGING_OVER_TESTED_REFINEMENT"
    else:
        trend = "INCONCLUSIVE_OVER_TESTED_REFINEMENT"

    return {
        "experiment": "PYM-Core-v0.3-legacy-comparison",
        "T": T,
        "dts": list(dts),
        "matched": {
            "q0": 0.7, "p0": -0.1, "lambda_pym": 0.05,
            "tau_M": 1.0, "memory_force_scale": 0.1,
        },
        "rows": rows,
        "refinement_ratios_rms_qp": qp_ratios,
        "refinement_ratios_rms_qpm": qpm_ratios,
        "observed_trend": trend,
        "critical_model_difference":
            "Legacy J = ||F_standard||*dt; v0.3 J = q[0]. Therefore this harness "
            "tests the implemented models as defined, not merely two integrators "
            "of an identical continuous equation.",
        "interpretation_guardrail":
            "Three finite dt values cannot establish a nonzero asymptotic limit or "
            "prove equivalence. A discrepancy that persists under refinement is evidence "
            "that the implementations are not behaving as the same numerical approximation "
            "over the tested range; the explicit J-law difference is already a model-definition change.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    a = ap.parse_args()
    result = report(a.T)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_vs_v03.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
