from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pym.lab.spectral_dissection import reconstruct_n3, TAU_M, DT
from pym.lab.prehistory_runner import CASES, PREP_TIME
from pym.lab.prehistory import prepare_pym, prepare_bath, release_pym, release_bath
from pym.physics.multimode_bath import BathParameters

OUT=Path("artifacts_emulation_horizon")
HORIZONS=(0.6,1.0,1.5,2.0,3.0,5.0)
NOMINAL_T=0.6
REL_THRESHOLDS=(0.05,0.50)

def first_crossing(rel_series,times,threshold):
    idx=np.flatnonzero(rel_series>threshold)
    return None if len(idx)==0 else float(times[int(idx[0])])

def main():
    # Frozen N=3 reconstruction; no fit/refit occurs in this experiment.
    nominal_targets={}
    for case in CASES:
        s=prepare_pym(case.q0,case.p0,0.05,DT,int(round(PREP_TIME/DT)),case.protocol)
        q,p=release_pym(s,int(round(NOMINAL_T/DT)),DT)
        nominal_targets[case.name]=(q,p,s)
    omega,coupling,thresholds,path=reconstruct_n3(nominal_targets)
    params=BathParameters(np.ones(3),omega,coupling)

    tmax=max(HORIZONS)
    steps=int(round(tmax/DT))
    # Kernel diagnostics use the same frozen single amplitude scaler from 003I's
    # nominal 0.6 s window; it is NOT refit at longer horizons.
    t=np.arange(0.0,tmax+0.5*DT,DT)
    A=coupling**2/(omega**2)
    kb=np.sum(A[:,None]*np.cos(omega[:,None]*t[None,:]),axis=0)
    kp=(1.0/TAU_M)*np.exp(-t/TAU_M)
    n_nom=int(round(NOMINAL_T/DT))+1
    a_diag=float(np.dot(kp[:n_nom],kb[:n_nom])/np.dot(kb[:n_nom],kb[:n_nom]))
    kbs=a_diag*kb

    case_series={}
    for case in CASES:
        prep_steps=int(round(PREP_TIME/DT))
        ps=prepare_pym(case.q0,case.p0,0.05,DT,prep_steps,case.protocol)
        bs=prepare_bath(params,case.q0,case.p0,DT,prep_steps,case.protocol)
        q1,p1=release_pym(ps,steps,DT)
        q2,p2=release_bath(bs,params,steps,DT)
        point=np.sum((q2-q1)**2,axis=1)+np.sum((p2-p1)**2,axis=1)
        cum=np.cumsum(point)/np.arange(1,steps+1)
        case_series[case.name]={"point":point,"cum":cum}

    rows=[]
    for T in HORIZONS:
        nk=int(round(T/DT))+1
        ne=int(round(T/DT))
        pearson=float(np.corrcoef(kp[:nk],kb[:nk])[0,1])
        rmse_scaled=float(np.sqrt(np.mean((kbs[:nk]-kp[:nk])**2)))
        row={"T_eval":T,"kernel_pearson":pearson,"kernel_rmse_scaled_fixed_a":rmse_scaled}
        for case in CASES:
            row[f"{case.name}_loss"]=float(case_series[case.name]["cum"][ne-1])
        rows.append(row)

    # Trajectory threshold crossing is relative to each case's frozen nominal
    # 0.6 s loss. This makes +5%/+50% operational and avoids mixing it with the
    # preregistered absolute PASS thresholds from 003B.
    crossings={}
    times=np.arange(1,steps+1)*DT
    n06=int(round(NOMINAL_T/DT))
    for case in CASES:
        nominal=float(case_series[case.name]["cum"][n06-1])
        rel=case_series[case.name]["cum"]/nominal-1.0
        # Search only after the nominal window.
        rel_after=rel[n06:]; time_after=times[n06:]
        crossings[case.name]={
            "nominal_loss_0p6":nominal,
            "first_plus_5pct_sec":first_crossing(rel_after,time_after,0.05),
            "first_plus_50pct_sec":first_crossing(rel_after,time_after,0.50)
        }

    result={"experiment":"003J Memory Emulation Horizon","horizons_sec":HORIZONS,
            "frozen_n3":{"omega":omega.tolist(),"coupling":coupling.tolist(),
                         "reconstruction_path":path},
            "kernel":{"tau_M":TAU_M,"a_diag_frozen_from_0p6":a_diag},
            "horizon_metrics":rows,"trajectory_crossings":crossings,
            "scope_note":"No parameters are refit beyond the frozen N=3 reconstruction. Kernel scaling a_diag is frozen from 0.6 s. Pearson decay and trajectory-loss growth diagnose bounded-horizon emulation; they do not prove a universal physical identity or non-identity."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"emulation_horizon_003j.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (OUT/"horizon_metrics_003j.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    plt.figure(figsize=(8,5)); plt.plot(t,kp,label="PYM exponential"); plt.plot(t,kbs,label="N=3 bath, fixed 0.6s scale"); plt.axvline(0.6,linestyle="--",label="nominal 0.6s"); plt.xlabel("t [s]"); plt.ylabel("Kernel"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"kernel_horizon_003j.png",dpi=180); plt.close()
    plt.figure(figsize=(8,5))
    for case in CASES: plt.plot(times,case_series[case.name]["cum"],label=case.name)
    plt.axvline(0.6,linestyle="--",label="nominal 0.6s"); plt.xlabel("t [s]"); plt.ylabel("Cumulative trajectory loss"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"trajectory_loss_horizon_003j.png",dpi=180); plt.close()

    print(json.dumps({"a_diag_frozen":a_diag,"horizon_metrics":rows,"trajectory_crossings":crossings},indent=2))

if __name__=="__main__":
    main()
