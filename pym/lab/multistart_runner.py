from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pym.lab.equivalence import trajectory_loss
from pym.lab.runner import IC, pym_trajectory
from pym.physics.multimode_bath import BathParameters


OMEGA_BOUNDS = (1e-2, 1e2)
COUPLING_BOUNDS = (1e-4, 1e1)
K_STARTS = 50
BASE_SEED = 42000


@dataclass(frozen=True)
class MultiStartRecord:
    n_modes: int
    start_index: int
    seed: int
    success: bool
    nit: int
    message: str
    loss_final: float
    omega_opt: list[float]
    coupling_opt: list[float]


def unpack_fixed_mass(log_params: np.ndarray, n_modes: int) -> BathParameters:
    omega = np.exp(log_params[:n_modes])
    coupling = np.exp(log_params[n_modes:])
    return BathParameters(np.ones(n_modes), omega, coupling)


def fit_one_start(target_q, target_p, q0, p0, dt, n_modes, seed, maxiter=150):
    rng = np.random.default_rng(seed)
    log_omega_bounds = (np.log(OMEGA_BOUNDS[0]), np.log(OMEGA_BOUNDS[1]))
    log_coupling_bounds = (np.log(COUPLING_BOUNDS[0]), np.log(COUPLING_BOUNDS[1]))
    x0 = np.concatenate([
        rng.uniform(*log_omega_bounds, size=n_modes),
        rng.uniform(*log_coupling_bounds, size=n_modes),
    ])
    bounds = [log_omega_bounds] * n_modes + [log_coupling_bounds] * n_modes

    def objective(x):
        return trajectory_loss(
            unpack_fixed_mass(x, n_modes), target_q, target_p, q0, p0, dt
        )

    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter},
    )
    params = unpack_fixed_mass(result.x, n_modes)
    return result, params


def run_multistart(
    modes=(1, 2, 4, 8, 16, 32),
    k_starts=K_STARTS,
    steps=600,
    dt=0.001,
    maxiter=150,
):
    train_ic = IC("IC_1", (1.0,), (0.15,), 2.0)
    target_q, target_p = pym_trajectory(train_ic, steps, dt)
    q0 = np.asarray(train_ic.q0, dtype=float)
    p0 = np.asarray(train_ic.p0, dtype=float)

    records = []
    for n_modes in modes:
        for start_index in range(k_starts):
            seed = BASE_SEED + start_index
            result, params = fit_one_start(
                target_q, target_p, q0, p0, dt, n_modes, seed, maxiter=maxiter
            )
            records.append(MultiStartRecord(
                n_modes=n_modes,
                start_index=start_index,
                seed=seed,
                success=bool(result.success),
                nit=int(result.nit),
                message=str(result.message),
                loss_final=float(result.fun),
                omega_opt=params.omega.tolist(),
                coupling_opt=params.coupling.tolist(),
            ))
    return records


def export_multistart(records, out_dir="artifacts_multistart"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "multistart_002.jsonl"
    summary_path = out / "multistart_002_summary.md"

    with raw_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")

    modes = sorted({r.n_modes for r in records})
    lines = [
        "| N | starts | converged | R_train* | median | p90 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in modes:
        group = [r for r in records if r.n_modes == n]
        losses = np.asarray([r.loss_final for r in group], dtype=float)
        lines.append(
            f"| {n} | {len(group)} | {sum(r.success for r in group)} | "
            f"{np.min(losses):.9e} | {np.median(losses):.9e} | "
            f"{np.quantile(losses, 0.9):.9e} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    records = run_multistart()
    raw, summary = export_multistart(records)
    print(summary.read_text(encoding="utf-8"))
    print(f"JSONL: {raw}")
