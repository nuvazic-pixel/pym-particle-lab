from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.minimal_search import ABLATION_ORDER
from pym.lab.prehistory_runner import CASES, loss_from_prepared, pym_target
from pym.lab.pruning import PROV, evaluate, passes, recover, OMEGA_BOUNDS, COUPLING_BOUNDS

OUT=Path("artifacts_n2_assault")
SIGMAS=(0.05,0.10,0.20,0.50)
K_PER_PROJECTION=50
BASE_SEED=46000
MAXITER=600
GTOL=1e-7
FTOL=1e-9

def optimize(omega,coupling,targets):
    from pym.physics.multimode_bath import BathParameters
    n=len(omega)
    x0=np.log(np.r_[omega,coupling])
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    bounds=[ob]*n+[cb]*n
    tq,tp,_=targets["IC_1"]
    def obj(x):
        p=BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))
        return loss_from_prepared(p,CASES[0],tq,tp)
    res=minimize(obj,x0,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL})
    o=np.exp(res.x[:n]); c=np.exp(res.x[n:])
    return o,c,res

def reconstruct_n3(targets):
    src=json.loads(PROV.read_text())
    o=np.asarray(src["omega"],float)
    c=np.asarray(src["coupling"],float)
    keep=list(ABLATION_ORDER[26:])
    o,c,_=recover(o[keep],c[keep],targets)
    # Reproduce 003G selection rule from 6->5->4->3.
    for _ in (5,4,3):
        cand=[]
        for i in range(len(o)):
            mask=np.ones(len(o),dtype=bool); mask[i]=False
            oo,cc,res=optimize(o[mask],c[mask],targets)
            losses=evaluate(oo,cc,targets)
            if passes(losses,src["preregistered_thresholds"]):
                cand.append((losses["IC_1"],i,oo,cc))
        if not cand:
            raise RuntimeError("Failed to reconstruct 003G path.")
        _,_,o,c=min(cand,key=lambda x:(x[0],x[1]))
    return o,c,src["preregistered_thresholds"]

def main():
    targets={case.name:pym_target(case) for case in CASES}
    o3,c3,thresholds=reconstruct_n3(targets)
    n3_losses=evaluate(o3,c3,targets)
    if not passes(n3_losses,thresholds):
        raise RuntimeError(f"Reconstructed N=3 does not pass: {n3_losses}")

    projections=[]
    for removed in range(3):
        mask=np.ones(3,dtype=bool); mask[removed]=False
        o2,c2,res=optimize(o3[mask],c3[mask],targets)
        projections.append({
            "removed_parent_index":removed,
            "omega":o2.tolist(),"coupling":c2.tolist(),
            "losses":evaluate(o2,c2,targets),
            "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev)
        })

    attacks=[]
    # Exactly 150 directed attacks total: 50 per N=2 projection.
    # Sigma cycles deterministically across the requested set (13,13,12,12).
    for pidx,p in enumerate(projections):
        base_o=np.asarray(p["omega"],float)
        base_c=np.asarray(p["coupling"],float)
        for k in range(K_PER_PROJECTION):
            sigma=SIGMAS[k % len(SIGMAS)]
            seed=BASE_SEED + 1000*pidx + k
            rng=np.random.default_rng(seed)
            o0=np.clip(base_o*np.exp(sigma*rng.normal(size=2)),OMEGA_BOUNDS[0],OMEGA_BOUNDS[1])
            c0=np.clip(base_c*np.exp(sigma*rng.normal(size=2)),COUPLING_BOUNDS[0],COUPLING_BOUNDS[1])
            o,c,res=optimize(o0,c0,targets)
            losses=evaluate(o,c,targets)
            attacks.append({
                "projection":pidx,"removed_parent_index":p["removed_parent_index"],
                "k":k,"seed":seed,"sigma":sigma,
                "losses":losses,"passes_5pct":passes(losses,thresholds),
                "success":bool(res.success),"nit":int(res.nit),"nfev":int(res.nfev),
                "omega":o.tolist(),"coupling":c.tolist()
            })

    passing=[r for r in attacks if r["passes_5pct"]]
    best=min(attacks,key=lambda r:r["losses"]["IC_1"])
    result={
        "experiment":"003H N=2 Boundary Assault",
        "thresholds":thresholds,
        "n3_reference_losses":n3_losses,
        "n2_projections":projections,
        "protocol":{
            "total_directed_attacks":len(attacks),
            "attacks_per_projection":K_PER_PROJECTION,
            "sigmas":SIGMAS,
            "sigma_allocation_per_projection":{"0.05":13,"0.10":13,"0.20":12,"0.50":12},
            "omega_bounds":OMEGA_BOUNDS,"coupling_bounds":COUPLING_BOUNDS,
            "maxiter":MAXITER,"gtol":GTOL,"ftol":FTOL,
            "objective":"IC_1 train only; IC_2/IC_3 audit only"
        },
        "attacks":attacks,
        "pass_count":len(passing),
        "best_train_attack":best,
        "scope_note":"If all directed N=2 attacks fail, this strengthens a practical N=3 boundary under this tested family, domain, perturbation distribution, optimizer budget, observables, horizon, ICs and thresholds. It does not prove algebraic impossibility, physical irreducibility, or a universal minimum."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"n2_assault_003h.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "thresholds":thresholds,
        "n3_reference_losses":n3_losses,
        "projection_losses":[p["losses"] for p in projections],
        "attacks":len(attacks),"pass_count":len(passing),
        "best_train_attack":{"projection":best["projection"],"seed":best["seed"],"sigma":best["sigma"],"losses":best["losses"],"success":best["success"],"nit":best["nit"],"nfev":best["nfev"]}
    },indent=2))

if __name__=="__main__":
    main()
