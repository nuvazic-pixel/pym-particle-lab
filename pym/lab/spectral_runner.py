from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from pym.lab.equivalence import trajectory_loss
from pym.lab.runner import IC, pym_trajectory
from pym.physics.multimode_bath import BathParameters

BEST_SEED = 42026
N_MODES = 16
HF_THRESHOLD = 999.0
SWEEP_VALUES = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def load_best_theta(path="artifacts_input/bounds_002.jsonl"):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [r for r in records if r["n_modes"] == N_MODES and r["seed"] == BEST_SEED]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one N=16 seed={BEST_SEED} record, found {len(matches)}")
    r = matches[0]
    return np.asarray(r["omega_opt"], float), np.asarray(r["coupling_opt"], float), float(r["loss_final"])


def residual(omega, coupling, dt, total_time=0.6):
    steps = int(round(total_time / dt))
    ic = IC("IC_1", (1.0,), (0.15,), 2.0)
    target_q, target_p = pym_trajectory(ic, steps, dt)
    params = BathParameters(np.ones(len(omega)), np.asarray(omega, float), np.asarray(coupling, float))
    return trajectory_loss(params, target_q, target_p, np.asarray(ic.q0), np.asarray(ic.p0), dt)


def run(path="artifacts_input/bounds_002.jsonl"):
    omega, coupling, recorded_loss = load_best_theta(path)
    hf = omega >= HF_THRESHOLD
    if int(hf.sum()) != 5:
        raise RuntimeError(f"Expected 5 HF modes, found {int(hf.sum())}")

    r_base = residual(omega, coupling, 0.001)
    out = {
        "source": {"n_modes": 16, "seed": BEST_SEED, "recorded_loss": recorded_loss},
        "hf_indices": np.flatnonzero(hf).tolist(),
        "baseline": {"dt": 0.001, "residual": r_base},
    }

    o = omega.copy(); o[hf] = 100.0
    r_freeze = residual(o, coupling, 0.001)
    out["freeze_100"] = {"residual": r_freeze, "rel_shift": (r_freeze-r_base)/r_base}

    c = coupling.copy(); c[hf] = 0.0
    r_ablate = residual(omega, c, 0.001)
    out["ablation_c0"] = {"residual": r_ablate, "rel_shift": (r_ablate-r_base)/r_base}

    sweep = {}
    for value in SWEEP_VALUES:
        o = omega.copy(); o[hf] = value
        r = residual(o, coupling, 0.001)
        sweep[str(int(value))] = {"residual": r, "rel_shift": (r-r_base)/r_base}
    out["sweep"] = sweep

    r_half = residual(omega, coupling, 0.0005)
    out["dt_half_control"] = {
        "dt": 0.0005, "steps": 1200, "residual": r_half,
        "delta_from_base": r_half-r_base, "rel_shift": (r_half-r_base)/r_base,
    }
    return out


def export(result, out_dir="artifacts_spectral"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / "spectral_002.json"
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"baseline: {result['baseline']['residual']:.12e}",
        f"freeze_100: {result['freeze_100']['residual']:.12e} rel={result['freeze_100']['rel_shift']:.6e}",
        f"ablation_c0: {result['ablation_c0']['residual']:.12e} rel={result['ablation_c0']['rel_shift']:.6e}",
    ]
    for k,v in result["sweep"].items():
        lines.append(f"sweep_{k}: {v['residual']:.12e} rel={v['rel_shift']:.6e}")
    d=result["dt_half_control"]
    lines.append(f"dt_half: {d['residual']:.12e} rel={d['rel_shift']:.6e}")
    (out/"spectral_002_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return p


if __name__ == "__main__":
    result = run()
    export(result)
    print((Path("artifacts_spectral")/"spectral_002_summary.txt").read_text())
