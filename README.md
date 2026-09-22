# PYM Particle Lab

**v0.1 — Information-Memory Particle Simulator**

PYM Particle Lab is a falsifiable computational sandbox for comparing standard dynamics with an explicit history-dependent memory hypothesis.

> **Scientific status:** PYM is a computational hypothesis, not a discovered particle or established physical theory.

## Research question

Can an explicit PYM memory term produce behavior that is distinguishable, numerically robust, and eventually distinguishable from conventional hidden-environment / reduced-system memory?

## Models

- **M0 — Standard:** ordinary harmonic dynamics; PYM coupling disabled.
- **M1 — PYM:** standard dynamics plus an explicit memory-dependent force.
- **M2 — Conventional memory comparator:** reserved for v0.1.x; an auxiliary hidden degree of freedom / bath comparator.

The PYM limit is controlled by `lambda_pym`:

`lambda_pym = 0` → standard baseline.

Memory is updated from a neutral **interaction signal J**, not assumed to be physical information:

`M[k+1] = alpha*M[k] + (1-alpha)*J[k]`

## Experiment 001 — Double History

Two particles start with identical observable physical state but different initial memory. The control run must remain identical when `lambda_pym=0`. With coupling enabled, any divergence is reported as a model effect — not evidence of new physics.

Run:

```bash
python -m pip install -r requirements.txt
python main.py
pytest -q
```

## Validation gates

- baseline position divergence < 1e-12
- baseline momentum divergence < 1e-12
- deterministic replay
- finite-state checks
- energy drift reported explicitly

## Provenance

Fields are classified as `MODEL_INPUT`, `DERIVED`, `HYPOTHESIS`, `NUMERICAL`, `EMPIRICAL`, or `UNKNOWN`.

## Roadmap

v0.1: deterministic simulator + Double-History Experiment  
v0.1.x: timestep/integrator convergence + conventional memory comparator  
v0.2: extended-phase-space / Hamiltonian embedding

## Interpretation rule

A conservation violation is a **model/implementation failure to explain**, not a discovery signal. PYM claims stay hypothetical until a distinct, falsifiable observable survives numerical controls and comparison with conventional models.
