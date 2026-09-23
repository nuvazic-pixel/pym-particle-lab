from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target, params_from_log

PROV=Path(__file__).parent/"provenance"/"theta_32_warm_converged_meta.json"
OUT=Path("artifacts_warm_boundary")
N=32
R0=0.026481298702114797
OB=(1e-2,1e3)
CB=(1e-4,1e2)

def count_hits(values,bounds):
    a=np.asarray(values,float); lo,hi=bounds
    return {"min":int(np.sum(np.isclose(a,lo,rtol=1e-8,atol=1e-12))),
            "max":int(np.sum(np.isclose(a,hi,rtol=1e-8,atol=1e-12)))}

def main():
    src=json.loads(PROV.read_text())
    x0=np.log(np.r_[src["omega"],src["coupling"]])
    bounds=[(np.log(OB[0]),np.log(OB[1]))]*N+[(np.log(CB[0]),np.log(CB[1]))]*N
    targets={case.name:pym_target(case) for case in CASES}
    train=CASES[0]; tq,tp,_=targets[train.name]
    def objective(x):
        return loss_from_prepared(params_from_log(x,N),train,tq,tp)
    baseline=float(objective(x0))
    res=minimize(objective,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":300,"gtol":1e-6,"ftol":1e-9})
    pars=params_from_log(res.x,N)
    final=float(res.fun)
    oos={}
    for case in CASES[1:]:
        q_target,p_target,_=targets[case.name]
        oos[case.name]=loss_from_prepared(pars,case,q_target,p_target)
    result={"preregistered_R0":R0,"baseline_recomputed":baseline,
      "train_loss_final":final,"delta_loss":final-baseline,
      "relative_change":(final-baseline)/baseline,"success":bool(res.success),
      "nit":int(res.nit),"nfev":int(res.nfev),"message":str(res.message),
      "omega_hits":count_hits(pars.omega,OB),"coupling_hits":count_hits(pars.coupling,CB),
      "oos_losses":oos,"omega":pars.omega.tolist(),"coupling":pars.coupling.tolist()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"warm_boundary_003b.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in ["baseline_recomputed","train_loss_final","delta_loss","relative_change","success","nit","nfev","message","omega_hits","coupling_hits","oos_losses"]},indent=2))

if __name__=="__main__":
    main()
