from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pym.lab.minimal_search import ABLATION_ORDER
from pym.lab.prehistory_runner import CASES, pym_target
from pym.lab.pruning import PROV, evaluate, passes, recover, OMEGA_BOUNDS, COUPLING_BOUNDS
from pym.lab.recursive_descent import optimize

OUT=Path("artifacts_spectral_dissection")
TAU_M=1.0
DT=0.001
TOTAL_TIME=0.6

def reconstruct_n3(targets):
    src=json.loads(PROV.read_text())
    thresholds=src["preregistered_thresholds"]
    o=np.asarray(src["omega"],float); c=np.asarray(src["coupling"],float)
    keep=list(ABLATION_ORDER[26:])
    o,c,_=recover(o[keep],c[keep],targets)
    path=[]
    for child_n in (5,4,3):
        cand=[]
        for i in range(len(o)):
            mask=np.ones(len(o),dtype=bool); mask[i]=False
            oo,cc,res=optimize(o[mask],c[mask],targets)
            losses=evaluate(oo,cc,targets)
            if passes(losses,thresholds):
                cand.append((losses["IC_1"],i,oo,cc,losses))
        if not cand:
            raise RuntimeError(f"No passing reconstruction at N={child_n}")
        _,removed,o,c,losses=min(cand,key=lambda x:(x[0],x[1]))
        path.append({"n":child_n,"removed_parent_index":removed,"losses":losses})
    return o,c,thresholds,path

def main():
    targets={case.name:pym_target(case) for case in CASES}
    omega,coupling,thresholds,path=reconstruct_n3(targets)
    base_losses=evaluate(omega,coupling,targets)
    if not passes(base_losses,thresholds):
        raise RuntimeError(f"N=3 reconstruction failed thresholds: {base_losses}")

    mass=np.ones_like(omega)
    periods=2*np.pi/omega
    amplitudes=coupling**2/(mass*omega**2)
    fractions=amplitudes/amplitudes.sum()
    spectral=[{"mode_index":i,"omega_rad_s":float(omega[i]),
               "coupling":float(coupling[i]),"period_sec":float(periods[i]),
               "A_i":float(amplitudes[i]),"fraction_A":float(fractions[i])}
              for i in range(3)]

    t=np.arange(0.0,TOTAL_TIME+0.5*DT,DT)
    contributions=np.asarray([amplitudes[i]*np.cos(omega[i]*t) for i in range(3)])
    k_bath=contributions.sum(axis=0)
    k_pym=(1.0/TAU_M)*np.exp(-t/TAU_M)
    denom=float(np.dot(k_bath,k_bath))
    a_diag=float(np.dot(k_pym,k_bath)/denom) if denom>0 else float("nan")
    k_scaled=a_diag*k_bath
    rmse_raw=float(np.sqrt(np.mean((k_bath-k_pym)**2)))
    rmse_scaled=float(np.sqrt(np.mean((k_scaled-k_pym)**2)))
    corr=float(np.corrcoef(k_bath,k_pym)[0,1])

    lomo=[]
    for i in range(3):
        c=coupling.copy(); c[i]=0.0
        losses=evaluate(omega,c,targets)
        delta={k:float(losses[k]-base_losses[k]) for k in base_losses}
        k_without=k_bath-contributions[i]
        kernel_rmse=float(np.sqrt(np.mean((k_without-k_bath)**2)))
        lomo.append({"ablated_mode":i,"losses":losses,"delta_losses":delta,
                     "passes_5pct":passes(losses,thresholds),
                     "kernel_delta_rmse":kernel_rmse})

    result={"experiment":"003I Spectral Dissection","tau_M":TAU_M,"dt":DT,
            "total_time":TOTAL_TIME,"thresholds":thresholds,
            "n3_reconstruction_path":path,"n3_losses":base_losses,
            "spectral_table":spectral,
            "kernel_diagnostics":{"a_diag":a_diag,"rmse_raw":rmse_raw,
                                  "rmse_scaled":rmse_scaled,"pearson_corr":corr},
            "lomo_ablation":lomo,
            "scope_note":"Kernel overlay is diagnostic. Similar trajectory residuals do not imply literal equality of the PYM exponential kernel and the finite conservative bath cosine kernel. T_i=2*pi/omega_i is an oscillation period, not a relaxation time."}

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"spectral_dissection_003i.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (OUT/"spectral_table_003i.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=spectral[0].keys()); w.writeheader(); w.writerows(spectral)
    with (OUT/"kernel_003i.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["t","K_pym","K_bath","K_bath_scaled","mode_0","mode_1","mode_2"])
        for j,x in enumerate(t): w.writerow([x,k_pym[j],k_bath[j],k_scaled[j],*contributions[:,j]])

    plt.figure(figsize=(8,5)); plt.plot(t,k_pym,label="PYM exponential"); plt.plot(t,k_bath,label="Bath raw"); plt.plot(t,k_scaled,label="Bath scaled diagnostic"); plt.xlabel("t [s]"); plt.ylabel("Kernel"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"kernel_overlay_003i.png",dpi=180); plt.close()
    plt.figure(figsize=(8,5)); plt.stem(omega,amplitudes); plt.xlabel("omega [rad/s]"); plt.ylabel("A_i = c_i^2/(m_i omega_i^2)"); plt.tight_layout(); plt.savefig(OUT/"spectral_weights_003i.png",dpi=180); plt.close()
    plt.figure(figsize=(8,5))
    for i in range(3): plt.plot(t,contributions[i],label=f"mode {i}")
    plt.xlabel("t [s]"); plt.ylabel("A_i cos(omega_i t)"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"mode_contributions_003i.png",dpi=180); plt.close()

    print(json.dumps({"n3_losses":base_losses,"spectral_table":spectral,
                      "kernel_diagnostics":result["kernel_diagnostics"],
                      "lomo_ablation":lomo},indent=2))

if __name__=="__main__":
    main()
