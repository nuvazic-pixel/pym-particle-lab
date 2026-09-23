from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target
from pym.lab.pruning import evaluate, passes
from pym.physics.multimode_bath import BathParameters

OUT=Path("artifacts_boundary_multistart")
N_VALUES=(5,6)
K_STARTS=50
BASE_SEED=44000
OMEGA_BOUNDS=(1e-2,1e3)
COUPLING_BOUNDS=(1e-4,1e2)
MAXITER=300
GTOL=1e-6
FTOL=1e-9

# Frozen from Experiment 003D / 003B preregistration.
THRESHOLDS={
    "train":0.01993909959175232,
    "IC_2":0.0023943529651270124,
    "IC_3":0.015479953672192508,
}

def fit_one(n:int, seed:int, targets):
    rng=np.random.default_rng(seed)
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    x0=np.concatenate([rng.uniform(*ob,n),rng.uniform(*cb,n)])
    bounds=[ob]*n+[cb]*n
    tq,tp,_=targets["IC_1"]

    def obj(x):
        pars=BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))
        return loss_from_prepared(pars,CASES[0],tq,tp)

    res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL})
    omega=np.exp(res.x[:n]); coupling=np.exp(res.x[n:])
    losses=evaluate(omega,coupling,targets)
    return {
        "n_eff":n,"seed":seed,"success":bool(res.success),
        "nit":int(res.nit),"nfev":int(res.nfev),"message":str(res.message),
        "losses":losses,"passes_5pct":passes(losses,THRESHOLDS),
        "omega":omega.tolist(),"coupling":coupling.tolist(),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,required=True,choices=N_VALUES)
    ap.add_argument("--start-index",type=int,default=0)
    ap.add_argument("--k-starts",type=int,default=10)
    args=ap.parse_args()

    targets={case.name:pym_target(case) for case in CASES}
    rows=[]
    for k in range(args.start_index,args.start_index+args.k_starts):
        rows.append(fit_one(args.n,BASE_SEED+k,targets))

    best=min(rows,key=lambda r:r["losses"]["IC_1"])
    payload={
        "experiment":"003E Independent Multistart Boundary Check",
        "n_eff":args.n,
        "start_index":args.start_index,
        "k_starts":args.k_starts,
        "base_seed":BASE_SEED,
        "bounds":{"omega":OMEGA_BOUNDS,"coupling":COUPLING_BOUNDS},
        "optimizer":{"method":"L-BFGS-B","maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL},
        "thresholds":THRESHOLDS,
        "rows":rows,
        "best_by_train":best,
        "n_passing_all_three":sum(r["passes_5pct"] for r in rows),
        "scope_note":"Independent random log-space starts; no warm start from the pruning path. Failure to pass is evidence only under this finite optimizer/search budget, not algebraic impossibility."
    }
    out=OUT/f"N{args.n}_start{args.start_index}"
    out.mkdir(parents=True,exist_ok=True)
    (out/"multistart_boundary_003e.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "n_eff":args.n,"start_index":args.start_index,"k_starts":args.k_starts,
        "best_seed":best["seed"],"best_losses":best["losses"],
        "best_passes_5pct":best["passes_5pct"],
        "passing_all_three":payload["n_passing_all_three"],
        "converged":sum(r["success"] for r in rows)
    },indent=2))

if __name__=="__main__":
    main()
