from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from pym.lab.prehistory import PrepProtocol, prepare_pym, prepare_bath, release_pym, release_bath
from pym.physics.multimode_bath import BathParameters

MODES=(1,2,4,8,16,32)
K_STARTS=50
BASE_SEED=43000
OMEGA_BOUNDS=(1e-2,1e2)
COUPLING_BOUNDS=(1e-4,1e1)
DT=0.001
PREP_TIME=1.0
EVAL_TIME=0.6
PENALTY=1e12


@dataclass(frozen=True)
class PreparedCase:
    name: str
    q0: tuple[float,...]
    p0: tuple[float,...]
    protocol: PrepProtocol


CASES=(
    PreparedCase("IC_1",(1.0,),(0.15,),PrepProtocol(0.50,2.0,0.0)),
    PreparedCase("IC_2",(0.8,),(-0.10,),PrepProtocol(0.35,3.0,0.4)),
    PreparedCase("IC_3",(1.2,),(0.05,),PrepProtocol(0.65,1.5,-0.3)),
)


@dataclass(frozen=True)
class PrehistoryFit:
    n_modes:int
    seed:int
    success:bool
    nit:int
    train_loss:float
    omega:list[float]
    coupling:list[float]
    oos_losses:dict[str,float]


def params_from_log(x,n):
    return BathParameters(np.ones(n),np.exp(x[:n]),np.exp(x[n:]))


def loss_from_prepared(params,case,target_q,target_p,dt=DT):
    prep_steps=int(round(PREP_TIME/dt)); eval_steps=int(round(EVAL_TIME/dt))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        b0=prepare_bath(params,case.q0,case.p0,dt,prep_steps,case.protocol)
        if b0 is None:
            return PENALTY
        q,p=release_bath(b0,params,eval_steps,dt)
        if q is None or p is None:
            return PENALTY
        dq=q-target_q; dp=p-target_p
        loss=float(np.mean(np.sum(dq*dq,axis=1)+np.sum(dp*dp,axis=1)))
    return loss if np.isfinite(loss) else PENALTY


def pym_target(case,dt=DT):
    prep_steps=int(round(PREP_TIME/dt)); eval_steps=int(round(EVAL_TIME/dt))
    s=prepare_pym(case.q0,case.p0,0.05,dt,prep_steps,case.protocol)
    q,p=release_pym(s,eval_steps,dt)
    return q,p,s


def fit_one(n,seed,targets,maxiter=150):
    rng=np.random.default_rng(seed)
    ob=(np.log(OMEGA_BOUNDS[0]),np.log(OMEGA_BOUNDS[1]))
    cb=(np.log(COUPLING_BOUNDS[0]),np.log(COUPLING_BOUNDS[1]))
    x0=np.concatenate([rng.uniform(*ob,n),rng.uniform(*cb,n)])
    bounds=[ob]*n+[cb]*n
    train=CASES[0]; tq,tp,_=targets[train.name]
    def objective(x):
        return loss_from_prepared(params_from_log(x,n),train,tq,tp)
    res=minimize(objective,x0,method="L-BFGS-B",bounds=bounds,options={"maxiter":maxiter})
    params=params_from_log(res.x,n)
    oos={}
    for case in CASES[1:]:
        q,p,_=targets[case.name]
        oos[case.name]=loss_from_prepared(params,case,q,p)
    return PrehistoryFit(n,seed,bool(res.success),int(res.nit),float(res.fun),
                         params.omega.tolist(),params.coupling.tolist(),oos)


def run(modes=MODES,k_starts=K_STARTS,maxiter=150):
    targets={c.name:pym_target(c) for c in CASES}
    fits=[]
    for n in modes:
        for k in range(k_starts):
            fits.append(fit_one(n,BASE_SEED+k,targets,maxiter))
    return fits,targets


def export(fits,targets,out_dir="artifacts_prehistory"):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"prehistory_003.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for r in fits:
            f.write(json.dumps(asdict(r),sort_keys=True,separators=(",",":"))+"\n")
    prep={}
    for case in CASES:
        s=targets[case.name][2]
        prep[case.name]={"pym_q0_release":s.position.tolist(),"pym_p0_release":s.momentum.tolist(),"pym_memory0_release":float(s.memory),
                         "protocol":asdict(case.protocol)}
    (out/"prehistory_003_prepared_states.json").write_text(json.dumps(prep,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["| N | converged | R_train* | best_seed | R_OOS(IC_2) | R_OOS(IC_3) |",
           "|---:|---:|---:|---:|---:|---:|"]
    for n in sorted({r.n_modes for r in fits}):
        g=[r for r in fits if r.n_modes==n]; best=min(g,key=lambda r:r.train_loss)
        lines.append(f"| {n} | {sum(r.success for r in g)}/{len(g)} | {best.train_loss:.9e} | {best.seed} | {best.oos_losses['IC_2']:.9e} | {best.oos_losses['IC_3']:.9e} |")
    (out/"prehistory_003_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,choices=MODES)
    ap.add_argument("--k-starts",type=int,default=K_STARTS)
    ap.add_argument("--maxiter",type=int,default=150)
    ap.add_argument("--start-index",type=int,default=0)
    args=ap.parse_args()
    modes=(args.n,) if args.n else MODES
    if args.start_index:
        global BASE_SEED
        BASE_SEED = BASE_SEED + args.start_index
    fits,targets=run(modes=modes,k_starts=args.k_starts,maxiter=args.maxiter)
    suffix=f"_start{args.start_index}" if args.start_index else ""
    out=export(fits,targets,out_dir=f"artifacts_prehistory/N{args.n}{suffix}" if args.n else "artifacts_prehistory")
    print((out/"prehistory_003_summary.md").read_text())
