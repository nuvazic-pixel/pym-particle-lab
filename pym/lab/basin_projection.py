from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.minimal_search import ABLATION_ORDER
from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target
from pym.lab.pruning import PROV, evaluate, passes, recover, OMEGA_BOUNDS, COUPLING_BOUNDS
from pym.physics.multimode_bath import BathParameters

OUT=Path("artifacts_basin_projection")
SIGMAS=(0.01,0.05,0.10,0.20,0.50)
K_PERTURB=20
BASE_SEED=45000
MAXITER_PROJECTION=600
MAXITER_PERTURB=300
GTOL=1e-6
FTOL=1e-9

def optimize(omega,coupling,targets,maxiter):
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
                 options={"maxiter":maxiter,"gtol":GTOL,"ftol":FTOL})
    o=np.exp(res.x[:n]); c=np.exp(res.x[n:])
    return o,c,res

def main():
    src=json.loads(PROV.read_text())
    omega32=np.asarray(src["omega"],float)
    coupling32=np.asarray(src["coupling"],float)
    thresholds=src["preregistered_thresholds"]
    targets={case.name:pym_target(case) for case in CASES}

    # Reconstruct the exact demonstrated N=6 solution from the frozen 003D path.
    keep6=list(ABLATION_ORDER[26:])
    o6,c6,res6=recover(omega32[keep6],coupling32[keep6],targets)
    base6=evaluate(o6,c6,targets)
    if not passes(base6,thresholds):
        raise RuntimeError(f"Reconstructed N=6 reference does not pass: {base6}")

    projections=[]
    for local_i,source_mode in enumerate(keep6):
        mask=np.ones(6,dtype=bool); mask[local_i]=False
        o5,c5,res=optimize(o6[mask],c6[mask],targets,MAXITER_PROJECTION)
        losses=evaluate(o5,c5,targets)
        projections.append({
            "removed_local_index":local_i,"removed_source_mode":int(source_mode),
            "losses":losses,"passes_5pct":passes(losses,thresholds),
            "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev),
            "message":str(res.message),"omega":o5.tolist(),"coupling":c5.tolist()
        })

    perturb=[]
    log6=np.log(np.r_[o6,c6])
    n=6
    lo=np.r_[np.full(n,np.log(OMEGA_BOUNDS[0])),np.full(n,np.log(COUPLING_BOUNDS[0]))]
    hi=np.r_[np.full(n,np.log(OMEGA_BOUNDS[1])),np.full(n,np.log(COUPLING_BOUNDS[1]))]
    for sidx,sigma in enumerate(SIGMAS):
        for k in range(K_PERTURB):
            seed=BASE_SEED+sidx*100+k
            rng=np.random.default_rng(seed)
            x=np.clip(log6+sigma*rng.normal(size=2*n),lo,hi)
            op=np.exp(x[:n]); cp=np.exp(x[n:])
            initial=evaluate(op,cp,targets)
            o,c,res=optimize(op,cp,targets,MAXITER_PERTURB)
            final=evaluate(o,c,targets)
            perturb.append({
                "sigma":sigma,"seed":seed,
                "initial_losses":initial,
                "final_losses":final,
                "final_passes_5pct":passes(final,thresholds),
                "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev)
            })

    by_sigma={}
    for sigma in SIGMAS:
        rows=[r for r in perturb if r["sigma"]==sigma]
        by_sigma[str(sigma)]={
            "k":len(rows),
            "recovered_pass":sum(r["final_passes_5pct"] for r in rows),
            "optimizer_success":sum(r["success"] for r in rows),
            "best_train":min(r["final_losses"]["IC_1"] for r in rows)
        }

    result={
        "experiment":"003F Local Basin Perturbation / Projection Audit",
        "thresholds":thresholds,
        "n6_reference":{"kept_source_modes":keep6,"losses":base6,
                        "success":bool(res6.success),"nit":int(res6.nit),
                        "omega":o6.tolist(),"coupling":c6.tolist()},
        "n6_to_n5_projections":projections,
        "n5_projection_pass_count":sum(r["passes_5pct"] for r in projections),
        "perturbation":{"sigmas":SIGMAS,"k_per_sigma":K_PERTURB,
                        "summary":by_sigma,"rows":perturb},
        "scope_note":"Projection failure is local/path-specific, not proof of algebraic impossibility. Perturbation recovery frequency estimates optimizer-basin accessibility under the specified perturbation distribution; it is not a physical radius."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"basin_projection_003f.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "n6_reference_losses":base6,
        "n5_projections":[{"removed_source_mode":r["removed_source_mode"],"losses":r["losses"],
                           "passes_5pct":r["passes_5pct"],"success":r["success"],
                           "nit":r["nit"],"nfev":r["nfev"]} for r in projections],
        "n5_projection_pass_count":result["n5_projection_pass_count"],
        "perturbation_summary":by_sigma
    },indent=2))

if __name__=="__main__":
    main()
