from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.prehistory_runner import CASES, PREP_TIME
from pym.lab.prehistory import prepare_pym, prepare_bath, release_pym, release_bath
from pym.lab.spectral_dissection import reconstruct_n3
from pym.physics.multimode_bath import BathParameters

DT=0.001
HORIZONS=(0.3,0.6,1.0,2.0,3.0,5.0)
TIERS=((1,2,3,4,6,8),(12,16),(24,32))
K=20
BASE_SEED=47000
OMEGA_BOUNDS=(1e-2,1e3)
COUPLING_BOUNDS=(1e-4,1e2)
MAXITER=300
GTOL=1e-6
FTOL=1e-9
ANCHOR_NRMSE=1.0215428270266633
ANCHOR_MARGIN=1.05
EPSILON=1.0726199683779965
ANCHOR_T=0.6
ANCHOR_SIGMA={"IC_1":0.13582885408422112,"IC_2":0.11707011915998188,"IC_3":0.18485292671545464}
OUT=Path("artifacts_horizon_scaling")

def targets_for(T):
    steps=int(round(T/DT)); prep=int(round(PREP_TIME/DT)); out={}
    for case in CASES:
        s=prepare_pym(case.q0,case.p0,0.05,DT,prep,case.protocol)
        q,p=release_pym(s,steps,DT)
        out[case.name]=(q,p)
    return out

def raw_loss(params,case,target,T):
    prep=int(round(PREP_TIME/DT)); steps=int(round(T/DT))
    # Infrastructure guard only: invalid numerical trajectories receive the
    # same finite sentinel and never participate in scientific PASS decisions.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            b=prepare_bath(params,case.q0,case.p0,DT,prep,case.protocol)
            if b is None: return 1e12
            q,p=release_bath(b,params,steps,DT)
            if q is None or not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))): return 1e12
            tq,tp=target
            d=(q-tq)**2+(p-tp)**2
            value=float(np.mean(np.sum(d,axis=1)))
            return value if np.isfinite(value) else 1e12
    except (FloatingPointError, OverflowError, ValueError):
        return 1e12

def sigma_target(target):
    q,p=target
    qc=q-np.mean(q,axis=0); pc=p-np.mean(p,axis=0)
    return float(np.sqrt(np.mean(np.sum(qc*qc,axis=1)+np.sum(pc*pc,axis=1))))

def metrics(params,T,targets):
    ans={}
    for case in CASES:
        r=raw_loss(params,case,targets[case.name],T)
        sig_dynamic=sigma_target(targets[case.name])
        sig_anchor=ANCHOR_SIGMA[case.name]
        ans[case.name]={
            "raw_loss":r,
            "nrmse_fixed":float(np.sqrt(r)/sig_anchor),
            "sigma_anchor":sig_anchor,
            "nrmse_dynamic_diagnostic":float(np.sqrt(r)/max(sig_dynamic,1e-15)),
            "sigma_pym_dynamic_diagnostic":sig_dynamic,
        }
    return ans

def is_pass(m,epsilon=EPSILON):
    return all(m[c.name]["nrmse_fixed"]<=epsilon for c in CASES)

def optimize_n(n,T,targets,x0,seed):
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    bounds=[ob]*n+[cb]*n
    train=CASES[0]
    def obj(x):
        pars=BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))
        return raw_loss(pars,train,targets[train.name],T)
    res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL})
    pars=BathParameters(np.ones(n),np.exp(res.x[:n]),np.exp(res.x[n:]))
    return {"seed":seed,"success":bool(res.success),"nit":int(res.nit),
            "omega":pars.omega.tolist(),"coupling":pars.coupling.tolist(),
            "metrics":metrics(pars,T,targets),"pass":is_pass(metrics(pars,T,targets))}

def random_x0(n,seed):
    rng=np.random.default_rng(seed)
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    return np.r_[rng.uniform(*ob,n),rng.uniform(*cb,n)]

def calibrate():
    T=.6; targets=targets_for(T)
    # reconstruct_n3 expects the historical target tuple shape q,p,state
    historical={}
    prep=int(round(PREP_TIME/DT)); steps=int(round(T/DT))
    for case in CASES:
        s=prepare_pym(case.q0,case.p0,0.05,DT,prep,case.protocol)
        q,p=release_pym(s,steps,DT); historical[case.name]=(q,p,s)
    o,c,_,path=reconstruct_n3(historical)
    m=metrics(BathParameters(np.ones(3),o,c),T,targets)
    ref=max(m[k]["nrmse_fixed"] for k in m)
    result={"experiment_version":"003K-v3","anchor_source":"Frozen N=3 at T=0.6 s (historical 003G-derived reconstruction)","anchor_T":ANCHOR_T,"anchor_sigma_per_ic":ANCHOR_SIGMA,"anchor_nrmse":ANCHOR_NRMSE,"anchor_margin":ANCHOR_MARGIN,"epsilon_frozen":EPSILON,"n3_reference_nrmse_max":ref,
            "reference_metrics":m,"calibration_pass":bool(ref<=EPSILON),
            "policy":"003K-v3: PASS uses per-IC sigma anchors frozen at T=0.6 s. epsilon is frozen at 1.0726199683779965 for every horizon. Dynamic sigma/NRMSE are diagnostic only."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"calibration_003k.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not np.isclose(ref,ANCHOR_NRMSE,rtol=0.0,atol=1e-12):
        raise SystemExit(f"PROVENANCE GATE FAILED: reconstructed anchor {ref} != frozen anchor {ANCHOR_NRMSE}")
    if ref>EPSILON:
        raise SystemExit("CALIBRATION GATE FAILED: frozen anchor exceeds 003K-v2 epsilon")

def independent_cell(T,n):
    """One preregistered (T,N) cell. K/bounds/optimizer/metric are unchanged."""
    targets=targets_for(T); starts=[]
    for k in range(K):
        seed=BASE_SEED+k
        starts.append(optimize_n(n,T,targets,random_x0(n,seed),seed))
    best=min(starts,key=lambda r:r["metrics"]["IC_1"]["raw_loss"])
    result={"phase":"003K-A-v3.1-cell","T":T,"n":n,"epsilon":EPSILON,
            "anchor_nrmse":ANCHOR_NRMSE,"anchor_margin":ANCHOR_MARGIN,
            "anchor_sigma_per_ic":ANCHOR_SIGMA,"K":K,"best":best,"starts":starts}
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f"cell_T{T:g}_N{n}.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"T":T,"n":n,"pass":best["pass"],
                      "train_loss":best["metrics"]["IC_1"]["raw_loss"]},indent=2))

def independent(T):
    # Retained for reproducibility of v3; v3.1 workflow uses independent_cell.
    targets=targets_for(T); rows=[]
    for tier in TIERS:
        tier_pass=False
        for n in tier:
            starts=[]
            for k in range(K):
                seed=BASE_SEED+k
                starts.append(optimize_n(n,T,targets,random_x0(n,seed),seed))
            best=min(starts,key=lambda r:r["metrics"]["IC_1"]["raw_loss"])
            rows.append({"n":n,"best":best,"starts":starts})
            tier_pass=tier_pass or best["pass"]
        if tier_pass: break
    passing=[r["n"] for r in rows if r["best"]["pass"]]
    result={"phase":"003K-A-v3","T":T,"epsilon":EPSILON,"anchor_nrmse":ANCHOR_NRMSE,"anchor_margin":ANCHOR_MARGIN,"anchor_sigma_per_ic":ANCHOR_SIGMA,"K":K,"results":rows,
            "n_min_independent":min(passing) if passing else None}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f"independent_T{T:g}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"T":T,"n_min_independent":result["n_min_independent"],
                      "tested_n":[r["n"] for r in rows]},indent=2))

def resize_theta(prev,n,seed):
    po=np.asarray(prev["omega"]); pc=np.asarray(prev["coupling"])
    if len(po)==n: return np.log(np.r_[po,pc])
    rng=np.random.default_rng(seed)
    if len(po)>n:
        # deterministic keep: largest kernel weights
        keep=np.argsort(pc*pc/(po*po))[-n:]
        return np.log(np.r_[po[keep],pc[keep]])
    add=n-len(po)
    oo=np.r_[po,np.exp(rng.uniform(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]),add))]
    cc=np.r_[pc,np.exp(rng.uniform(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]),add))]
    return np.log(np.r_[oo,cc])

def continuation():
    chain=[]; previous={}
    for T in HORIZONS:
        targets=targets_for(T); rows=[]
        for tier in TIERS:
            tier_pass=False
            for n in tier:
                starts=[]
                # one true continuation start if available, plus deterministic independent starts.
                if n in previous:
                    starts.append(optimize_n(n,T,targets,resize_theta(previous[n],n,BASE_SEED),-1))
                for k in range(K):
                    seed=BASE_SEED+k
                    starts.append(optimize_n(n,T,targets,random_x0(n,seed),seed))
                best=min(starts,key=lambda r:r["metrics"]["IC_1"]["raw_loss"])
                rows.append({"n":n,"best":best})
                previous[n]=best
                tier_pass=tier_pass or best["pass"]
            if tier_pass: break
        passing=[r["n"] for r in rows if r["best"]["pass"]]
        chain.append({"T":T,"n_min_continuation":min(passing) if passing else None,"results":rows})
    result={"phase":"003K-B-v3","epsilon":EPSILON,"anchor_nrmse":ANCHOR_NRMSE,"anchor_margin":ANCHOR_MARGIN,"anchor_sigma_per_ic":ANCHOR_SIGMA,"chain":chain,
            "note":"Continuation candidates are never ranked by OOS; IC2/IC3 are audit-only."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"continuation_003k.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps([{"T":x["T"],"n_min_continuation":x["n_min_continuation"]} for x in chain],indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("calibrate","independent","cell","continuation"),required=True)
    ap.add_argument("--T",type=float)
    ap.add_argument("--N",type=int)
    a=ap.parse_args()
    if a.phase=="calibrate": calibrate()
    elif a.phase=="independent":
        if a.T not in HORIZONS: raise SystemExit("--T must be one of preregistered horizons")
        independent(a.T)
    elif a.phase=="cell":
        if a.T not in HORIZONS: raise SystemExit("--T must be one of preregistered horizons")
        allowed={n for tier in TIERS for n in tier}
        if a.N not in allowed: raise SystemExit("--N must be a preregistered capacity")
        independent_cell(a.T,a.N)
    else: continuation()
