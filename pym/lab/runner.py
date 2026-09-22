from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from pym.core.engine import PYMEngine
from pym.core.state import PYMState
from pym.lab.equivalence import fit_bath, oos_loss
from pym.physics.pym_forces import memory_force
from pym.physics.standard import harmonic_force


@dataclass(frozen=True)
class IC:
    name: str
    q0: tuple[float, ...]
    p0: tuple[float, ...]
    memory0: float


@dataclass(frozen=True)
class EquivalenceRow:
    n_modes: int
    train_loss: float
    oos_losses: dict[str, float]
    optimization_status: str
    interpretation: str


def pym_trajectory(ic: IC, steps: int, dt: float, lambda_pym: float = 0.05):
    state = PYMState(
        position=np.asarray(ic.q0, dtype=float),
        momentum=np.asarray(ic.p0, dtype=float),
        memory=ic.memory0,
        lambda_pym=lambda_pym,
        alpha=float(np.exp(-dt / 1.0)),
    )
    engine = PYMEngine()
    qs, ps = [], []
    for _ in range(steps):
        state = engine.step(state, dt, harmonic_force, memory_force)
        qs.append(state.position.copy())
        ps.append(state.momentum.copy())
    return np.asarray(qs), np.asarray(ps)


def classify(rows: list[EquivalenceRow], index: int) -> str:
    row = rows[index]
    if not row.optimization_status == "CONVERGED":
        return "Optimization unresolved"
    if index == 0:
        return "Initial model capacity"
    prev = rows[index - 1]
    prev_oos = float(np.mean(list(prev.oos_losses.values())))
    curr_oos = float(np.mean(list(row.oos_losses.values())))
    if curr_oos < 0.5 * prev_oos:
        return "OOS residual decreasing"
    if abs(curr_oos - prev_oos) <= 0.05 * max(prev_oos, 1e-15):
        return "Candidate plateau; requires stronger controls"
    return "No monotonic equivalence trend"


def run_equivalence_challenge(
    modes=(1, 2, 4, 8, 16, 32),
    steps: int = 600,
    dt: float = 0.001,
    maxiter: int = 150,
):
    train_ic = IC("IC_1", (1.0,), (0.15,), 2.0)
    test_ics = [
        IC("IC_2", (0.8,), (-0.10,), 0.5),
        IC("IC_3", (1.2,), (0.05,), 3.0),
    ]
    tq, tp = pym_trajectory(train_ic, steps, dt)
    rows: list[EquivalenceRow] = []

    for n in modes:
        fit = fit_bath(tq, tp, np.asarray(train_ic.q0), np.asarray(train_ic.p0), dt, n, maxiter=maxiter)
        losses = {}
        for ic in test_ics:
            oq, op = pym_trajectory(ic, steps, dt)
            losses[ic.name] = oos_loss(
                fit, oq, op, np.asarray(ic.q0), np.asarray(ic.p0), dt
            )
        status = "CONVERGED" if fit.success else "OPTIMIZER_WARNING"
        row = EquivalenceRow(n, fit.train_loss, losses, status, "")
        rows.append(row)
        rows[-1] = EquivalenceRow(
            row.n_modes, row.train_loss, row.oos_losses, row.optimization_status,
            classify(rows, len(rows) - 1)
        )
    return rows


def export_results(rows: list[EquivalenceRow], out_dir="artifacts"):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    jsonl = path / "experiment_002.jsonl"
    md = path / "experiment_002.md"
    with jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n")

    names = sorted({k for row in rows for k in row.oos_losses})
    header = ["N", "R_train", *[f"R_OOS({n})" for n in names], "Optimization", "Interpretation"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        values = [str(row.n_modes), f"{row.train_loss:.6e}"]
        values += [f"{row.oos_losses[n]:.6e}" for n in names]
        values += [row.optimization_status, row.interpretation]
        lines.append("| " + " | ".join(values) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl, md


if __name__ == "__main__":
    results = run_equivalence_challenge()
    j, m = export_results(results)
    print(m.read_text(encoding="utf-8"))
    print(f"JSONL: {j}")
