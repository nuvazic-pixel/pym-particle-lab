from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.minimal_search import ABLATION_ORDER
from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target
from pym.lab.pruning import PROV, evaluate, passes, recover, OMEGA_BOUNDS, COUPLING_BOUNDS
from pym.physics.multimode_bath import BathParameters

OUT=Path("artifacts_recursive_descent")
MAXITER=600
GTOL=1e-6
FTOL=1e-9

def optimize(omega,coupling,targets):
    n=len(omega)
    x0=np.log(np.r_[omega,coupling])
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    bounds=[ob]*n+[cb]*n
    tq,tp,_=targets["IC_1"]
    def obj(x):
        pars=BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))
        return loss_from_prepared(pars,CASES[0],tq,tp)
    res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL})
    o=np.exp(res.x[:n]); c=np.exp(res.x[n:])
    return o,c,res

def main():
    src=json.loads(PROV.read_text())
    thresholds=src["preregistered_thresholds"]
    targets={case.name:pym_target(case) for case in CASES}
    omega32=np.asarray(src["omega"],float)
    coupling32=np.asarray(src["coupling"],float)

    # Reconstruct the validated N=6 continuation solution, then deterministically
    # choose the best passing 6->5 projection by train loss only. OOS is never
    # used to rank candidates; it is only a hard pass/fail audit.
    keep6=list(ABLATION_ORDER[26:])
    o6,c6,res6=recover(omega32[keep6],coupling32[keep6],targets)
    base6=evaluate(o6,c6,targets)
    if not passes(base6,thresholds):
        raise RuntimeError(f"N=6 reference failed reconstruction: {base6}")

    initial5=[]
    for i,source_mode in enumerate(keep6):
        mask=np.ones(6,dtype=bool); mask[i]=False
        o,c,res=optimize(o6[mask],c6[mask],targets)
        losses=evaluate(o,c,targets)
        initial5.append({
            "parent_n":6,"child_n":5,"removed_parent_index":i,
            "removed_source_mode":int(source_mode),"losses":losses,
            "passes_5pct":passes(losses,thresholds),"success":bool(res.success),
            "nit":int(res.nit),"nfev":int(res.nfev),
            "omega":o.tolist(),"coupling":c.tolist()
        })
    passing5=[r for r in initial5 if r["passes_5pct"]]
    if not passing5:
        raise RuntimeError("No passing N=5 projection; 003F result not reproduced.")
    current=min(passing5,key=lambda r:(r["losses"]["IC_1"],r["removed_parent_index"]))
    current_o=np.asarray(current["omega"],float)
    current_c=np.asarray(current["coupling"],float)

    levels=[{"n":5,"selected":current,"all_candidates":initial5}]
    stopping=None

    while len(current_o)>1:
        parent_n=len(current_o)
        candidates=[]
        for i in range(parent_n):
            mask=np.ones(parent_n,dtype=bool); mask[i]=False
            o,c,res=optimize(current_o[mask],current_c[mask],targets)
            losses=evaluate(o,c,targets)
            candidates.append({
                "parent_n":parent_n,"child_n":parent_n-1,
                "removed_parent_index":i,"losses":losses,
                "passes_5pct":passes(losses,thresholds),
                "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev),
                "omega":o.tolist(),"coupling":c.tolist()
            })

        passing=[r for r in candidates if r["passes_5pct"]]
        if not passing:
            stopping={
                "parent_n":parent_n,
                "attempted_child_n":parent_n-1,
                "reason":"all single-mode projections failed at least one preregistered threshold",
                "candidates":candidates
            }
            break

        selected=min(passing,key=lambda r:(r["losses"]["IC_1"],r["removed_parent_index"]))
        levels.append({"n":parent_n-1,"selected":selected,"all_candidates":candidates})
        current_o=np.asarray(selected["omega"],float)
        current_c=np.asarray(selected["coupling"],float)

    smallest=min(level["n"] for level in levels)
    result={
        "experiment":"003G Recursive Projection Descent",
        "thresholds":thresholds,
        "selection_rule":"At each level, enumerate every one-mode deletion; optimize IC_1 only; retain passing candidates using all frozen thresholds; among passers choose minimum IC_1 train loss, tie-break by removed index.",
        "levels":levels,
        "smallest_passing_n_on_recursive_path":smallest,
        "stopping_boundary":stopping,
        "scope_note":"This is the smallest passing N found along the preregistered recursive one-mode projection/continuation path. It is not a proof of globally minimal algebraic realization or physical irreducibility."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"recursive_descent_003g.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "thresholds":thresholds,
        "path":[{"n":x["n"],"losses":x["selected"]["losses"],
                 "removed_parent_index":x["selected"]["removed_parent_index"]}
                for x in levels],
        "smallest_passing_n_on_recursive_path":smallest,
        "stopping_boundary":None if stopping is None else {
            "parent_n":stopping["parent_n"],
            "attempted_child_n":stopping["attempted_child_n"],
            "candidate_losses":[{"removed_parent_index":r["removed_parent_index"],
                                 "losses":r["losses"],"passes_5pct":r["passes_5pct"],
                                 "success":r["success"],"nit":r["nit"],"nfev":r["nfev"]}
                                for r in stopping["candidates"]]
        }
    },indent=2))

if __name__=="__main__":
    main()
