from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pym.lab.equivalence import trajectory_loss
from pym.lab.runner import IC, pym_trajectory
from pym.physics.multimode_bath import BathParameters


OMEGA_BOUNDS = (1e-2, 1e3)
COUPLING_BOUNDS = (1e-4, 1e2)
K_STARTS = 50
BASE_SEED = 42000
MODES = (16, 32)


@dataclass(frozen=True)
class BoundsRecord:
    n_modes: int
    start_index: int
    seed: int
    success: bool
    nit: int
    message: str
    loss_final: float
    omega_opt: list[float]
    coupling_opt: list[float]
    omega_at_min: int
    omega_at_max: int
    coupling_at_min: int
    coupling_at_max: int


def unpack_fixed_mass(log_params: np.ndarray, n_modes: int) -> BathParameters:
    omega = np.exp(log_params[:n_modes])
    coupling = np.exp(log_params[n_modes:])
    return BathParameters(np.ones(n_modes), omega, coupling)


def boundary_counts(params: BathParameters, rtol: float = 1e-6) -> tuple[int, int, int, int]:
    return (
        int(np.sum(np.isclose(params.omega, OMEGA_BOUNDS[0], rtol=rtol))),
        int(np.sum(np.isclose(params.omega, OMEGA_BOUNDS[1], rtol=rtol))),
        int(np.sum(np.isclose(params.coupling, COUPLING_BOUNDS[0], rtol=rtol))),
        int(np.sum(np.isclose(params.coupling, COUPLING_BOUNDS[1], rtol=rtol))),
    )


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
    return result, unpack_fixed_mass(result.x, n_modes)


def run_bounds(
    modes=MODES, k_starts=K_STARTS, steps=600, dt=0.001, maxiter=150
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
            omin, omax, cmin, cmax = boundary_counts(params)
            records.append(BoundsRecord(
                n_modes=n_modes, start_index=start_index, seed=seed,
                success=bool(result.success), nit=int(result.nit),
                message=str(result.message), loss_final=float(result.fun),
                omega_opt=params.omega.tolist(),
                coupling_opt=params.coupling.tolist(),
                omega_at_min=omin, omega_at_max=omax,
                coupling_at_min=cmin, coupling_at_max=cmax,
            ))
    return records


def export_bounds(records, out_dir="artifacts_bounds"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "bounds_002.jsonl"
    summary_path = out / "bounds_002_summary.md"
    with raw_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True, separators=(",", ":")) + "\n")

    lines = [
        "| N | starts | converged | R_train* | best_seed | omega_min/max hits | c_min/max hits |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({r.n_modes for r in records}):
        group = [r for r in records if r.n_modes == n]
        best = min(group, key=lambda r: r.loss_final)
        lines.append(
            f"| {n} | {len(group)} | {sum(r.success for r in group)} | "
            f"{best.loss_final:.9e} | {best.seed} | "
            f"{best.omega_at_min}/{best.omega_at_max} | "
            f"{best.coupling_at_min}/{best.coupling_at_max} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    records = run_bounds()
    raw, summary = export_bounds(records)
    print(summary.read_text(encoding="utf-8"))
    print(f"JSONL: {raw}")
