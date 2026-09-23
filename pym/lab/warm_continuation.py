from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.prehistory_runner import (
    CASES, DT, OMEGA_BOUNDS, COUPLING_BOUNDS,
    loss_from_prepared, pym_target, params_from_log,
)

PROVENANCE = Path(__file__).parent / "provenance" / "theta_32_seed43003.json"
OUT = Path("artifacts_warm_continuation")
MAXITER = 600
GTOL = 1e-6
FTOL = 1e-9


def main():
    src=json.loads(PROVENANCE.read_text(encoding="utf-8"))
    omega=np.asarray(src["omega"],dtype=float)
    coupling=np.asarray(src["coupling"],dtype=float)
    n=int(src["n_modes"])
    if n != 32 or omega.size != n or coupling.size != n:
        raise ValueError("provenance vector must contain exactly 32 omega and 32 coupling values")

    x0=np.log(np.concatenate([omega,coupling]))
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    bounds=[ob]*n+[cb]*n

    targets={c.name:pym_target(c) for c in CASES}
    train=CASES[0]
    tq,tp,_=targets[train.name]

    def objective(x):
        return loss_from_prepared(params_from_log(x,n),train,tq,tp)

    baseline=float(objective(x0))
    res=minimize(
        objective,x0,method="L-BFGS-B",bounds=bounds,
        options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL},
    )
    params=params_from_log(res.x,n)
    final=float(res.fun)
    oos={}
    for case in CASES[1:]:
        q,p,_=targets[case.name]
        oos[case.name]=loss_from_prepared(params,case,q,p)

    result={
        "source_seed":src["source"]["seed"],
        "source_artifact_id":src["source"]["artifact_id"],
        "source_train_loss_recorded":src["train_loss"],
        "baseline_recomputed":baseline,
        "maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL,
        "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev),
        "message":str(res.message),
        "train_loss_final":final,
        "delta_loss":final-baseline,
        "relative_change":(final-baseline)/baseline,
        "oos_losses":oos,
        "omega":params.omega.tolist(),
        "coupling":params.coupling.tolist(),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"warm_continuation_43003.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary=(
        f"# Experiment 003 Warm Continuation — seed 43003\n\n"
        f"- baseline recomputed: {baseline:.12e}\n"
        f"- final train loss: {final:.12e}\n"
        f"- delta: {final-baseline:.12e}\n"
        f"- relative change: {(final-baseline)/baseline:.6%}\n"
        f"- success: {bool(res.success)}\n"
        f"- nit: {int(res.nit)}\n"
        f"- nfev: {int(res.nfev)}\n"
        f"- message: {res.message}\n"
        f"- OOS IC_2: {oos['IC_2']:.12e}\n"
        f"- OOS IC_3: {oos['IC_3']:.12e}\n"
    )
    (OUT/"warm_continuation_43003_summary.md").write_text(summary,encoding="utf-8")
    print(summary)


if __name__=="__main__":
    main()
