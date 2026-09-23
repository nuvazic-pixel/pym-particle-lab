from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from pym.lab.prehistory_runner import CASES, pym_target
from pym.lab.pruning import PROV, evaluate, passes, recover

OUT=Path("artifacts_minimal")
SEARCH_N=(7,6,5,4,3,2,1)

# Frozen by Experiment 003C; do not recompute/re-rank after seeing 003D outcomes.
ABLATION_ORDER=(29,13,18,17,31,22,30,15,5,2,16,1,20,7,24,26,
                25,12,14,23,10,4,11,8,9,21,19,0,3,27,6,28)

def main():
    src=json.loads(PROV.read_text())
    omega=np.asarray(src["omega"],float)
    coupling=np.asarray(src["coupling"],float)
    thresholds=src["preregistered_thresholds"]
    targets={case.name:pym_target(case) for case in CASES}
    baseline=evaluate(omega,coupling,targets)

    rows=[]
    for n_eff in SEARCH_N:
        keep=list(ABLATION_ORDER[32-n_eff:])
        o0=omega[keep]; c0=coupling[keep]
        o,c,res=recover(o0,c0,targets)
        losses=evaluate(o,c,targets)
        rows.append({
            "n_eff":n_eff,
            "kept_modes":keep,
            "removed_modes":list(ABLATION_ORDER[:32-n_eff]),
            "losses":losses,
            "passes_5pct":passes(losses,thresholds),
            "success":bool(res.success),
            "nit":int(res.nit),
            "nfev":int(res.nfev),
            "message":str(res.message),
            "omega":o.tolist(),
            "coupling":c.tolist()
        })

    passing=[r["n_eff"] for r in rows if r["passes_5pct"]]
    n_min=min(passing) if passing else None
    result={
        "experiment":"003D Minimal Realization Search",
        "baseline_n32":baseline,
        "thresholds":thresholds,
        "search_n":list(SEARCH_N),
        "ablation_order_source":"Experiment 003C static single-mode train-loss ablation order",
        "ablation_order":list(ABLATION_ORDER),
        "results":rows,
        "n_min_under_tested_protocol":n_min,
        "definition":"Minimum tested N satisfying all three preregistered +5% thresholds simultaneously.",
        "scope_note":"N_min is protocol/observable/domain/optimizer specific; it is not a universal physical equivalence claim."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"minimal_003d.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "baseline_n32":baseline,
        "thresholds":thresholds,
        "results":[{"n_eff":r["n_eff"],"losses":r["losses"],
                    "passes_5pct":r["passes_5pct"],"success":r["success"],
                    "nit":r["nit"],"nfev":r["nfev"]} for r in rows],
        "n_min_under_tested_protocol":n_min
    },indent=2))

if __name__=="__main__":
    main()
