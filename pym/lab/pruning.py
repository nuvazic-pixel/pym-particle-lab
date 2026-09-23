from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target
from pym.physics.multimode_bath import BathParameters

PROV=Path(__file__).parent/"provenance"/"theta_32_003b.json"
OUT=Path("artifacts_pruning")
RECOVERY_N=(24,20,16,12,8)
OMEGA_BOUNDS=(1e-2,1e3)
COUPLING_BOUNDS=(1e-4,1e2)
MAXITER=300
GTOL=1e-6
FTOL=1e-9

def evaluate(omega,coupling,targets):
    params=BathParameters(np.ones(len(omega)),np.asarray(omega,float),np.asarray(coupling,float))
    losses={}
    for case in CASES:
        q,p,_=targets[case.name]
        losses[case.name]=float(loss_from_prepared(params,case,q,p))
    return losses

def passes(losses,thresholds):
    return (losses["IC_1"]<=thresholds["train"] and
            losses["IC_2"]<=thresholds["IC_2"] and
            losses["IC_3"]<=thresholds["IC_3"])

def recover(omega,coupling,targets):
    n=len(omega)
    x0=np.log(np.r_[omega,coupling])
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    bounds=[ob]*n+[cb]*n
    train=CASES[0]; tq,tp,_=targets[train.name]
    def obj(x):
        pars=BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))
        return loss_from_prepared(pars,train,tq,tp)
    res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL})
    o=np.exp(res.x[:n]); c=np.exp(res.x[n:])
    return o,c,res

def main():
    src=json.loads(PROV.read_text())
    omega=np.asarray(src["omega"],float)
    coupling=np.asarray(src["coupling"],float)
    thresholds=src["preregistered_thresholds"]
    targets={case.name:pym_target(case) for case in CASES}
    baseline=evaluate(omega,coupling,targets)

    single=[]
    for i in range(32):
        c=coupling.copy(); c[i]=0.0
        losses=evaluate(omega,c,targets)
        single.append({"mode":i,"omega":float(omega[i]),"coupling":float(coupling[i]),
                       "losses":losses,"delta_train":losses["IC_1"]-baseline["IC_1"]})
    order=[r["mode"] for r in sorted(single,key=lambda r:(r["delta_train"],r["mode"]))]

    frozen=[]
    keep=list(range(32))
    frozen.append({"n_eff":32,"removed":[],"losses":baseline,"passes_5pct":passes(baseline,thresholds)})
    for mode in order:
        keep.remove(mode)
        losses=evaluate(omega[keep],coupling[keep],targets) if keep else {"IC_1":1e12,"IC_2":1e12,"IC_3":1e12}
        frozen.append({"n_eff":len(keep),"removed":order[:32-len(keep)],
                       "losses":losses,"passes_5pct":passes(losses,thresholds)})

    recovery=[]
    for n_eff in RECOVERY_N:
        keep=order[32-n_eff:]
        o0=omega[keep]; c0=coupling[keep]
        o,c,res=recover(o0,c0,targets)
        losses=evaluate(o,c,targets)
        recovery.append({"n_eff":n_eff,"kept_modes":keep,"removed_modes":order[:32-n_eff],
                         "losses":losses,"passes_5pct":passes(losses,thresholds),
                         "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev),
                         "message":str(res.message),"omega":o.tolist(),"coupling":c.tolist()})

    passing=[r["n_eff"] for r in recovery if r["passes_5pct"]]
    result={"baseline":baseline,"thresholds":thresholds,
            "single_mode_ablation":single,"static_ablation_order":order,
            "frozen_pruning":frozen,"recovery":recovery,
            "minimal_preregistered_recovery_n":min(passing) if passing else None,
            "method_note":"Frozen curve uses the preregistered static order from single-mode train-loss ablation; recovery reoptimizes surviving modes at N=24,20,16,12,8."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"pruning_003c.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    summary={"baseline":baseline,"thresholds":thresholds,"ablation_order":order,
             "recovery":[{"n_eff":r["n_eff"],"losses":r["losses"],"passes_5pct":r["passes_5pct"],
                          "success":r["success"],"nit":r["nit"]} for r in recovery],
             "minimal_preregistered_recovery_n":result["minimal_preregistered_recovery_n"]}
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
