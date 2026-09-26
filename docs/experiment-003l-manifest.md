# Experiment 003L — Model-Class Ladder Manifest

**Status:** PREREGISTERED BEFORE LEVEL-1 IMPLEMENTATION  
**Branch:** `feature/003l-model-class-ladder`  
**Lineage anchor:** PYM-Core v0.3 isolation-gate commit `88d994be8039edccdb6a19c1a68d985600c55b15`  
**Isolation run:** `36230579881`  
**Isolation artifact digest:** `sha256:dae2b34ca055a2a0a503281410d9f8b8e3e161b9d321c13235ee12aa80e587aa`

## 1. Scientific question

To what extent can the phenomenological non-Markovian PYM-Core v0.3 dynamics be emulated by finite-dimensional conventional latent-environment model classes over bounded observation horizons?

003L is a computational model-comparison experiment. It is not a claim of new physics. PASS establishes bounded-horizon numerical emulation under the frozen protocol only. FAIL establishes mismatch/accessibility failure for the tested model class, capacity, bounds, optimizer budget, histories, observables, metric, and horizons only.

## 2. Target model and frozen observable

The target is **PYM-Core v0.3**, not PYM-Legacy.

Target configuration:
- signal law: `LINEAR_Q`, J = q[0];
- tau_M = 1.0;
- lambda_pym = 0.05;
- memory-force scale = 0.1;
- standard observable force: harmonic, F_std = -q;
- v0.3 symmetric kick-drift-kick ordering and exact piecewise-constant memory substep as specified in `docs/pym-core-v0.3-spec.md`;
- primary observable trajectory: (q(t), p(t));
- latent/memory states are diagnostic and are not fitted as observable targets.

The exact piecewise-constant memory recurrence is not described as an exact solver for the complete coupled trajectory.

## 3. Shared prehistory and train/OOS split

All model classes receive the same observable initial conditions and the same external preparation protocol from t=-1.0 s to t=0. Latent states begin from the neutral state defined by each class; no latent state may be hand-set at release to match the PYM target.

Preparation cases are frozen from the 003 prehistory protocol:

| Case | Role | q(-1) | p(-1) | forcing u(t) |
|---|---|---:|---:|---|
| IC_1 | TRAIN | 1.0 | 0.15 | 0.50 sin(2.0 t + 0.0) |
| IC_2 | OOS only | 0.8 | -0.10 | 0.35 sin(3.0 t + 0.4) |
| IC_3 | OOS only | 1.2 | 0.05 | 0.65 sin(1.5 t - 0.3) |

At t=0 the external preparation force is removed and the autonomous release trajectory is evaluated. IC_2 and IC_3 must never influence parameter selection, optimizer ranking, stopping, or hyperparameter tuning.

**Implementation requirement:** the PYM target preparation/release path for 003L must use PYM-Core v0.3. The legacy `prepare_pym/release_pym` path from Experiment 003K is provenance reference only and must not be used to generate 003L targets.

## 4. Numerical grid and horizons

Primary dt = 0.001 s.

Frozen release horizons:
T in {0.3, 0.6, 1.0, 2.0, 3.0, 5.0} s.

Numerical-convergence controls are reported separately and must not be silently substituted for the primary grid.

## 5. Metric and normalization

For case j and horizon T:

R_j(T) = mean_t [ ||q_model-q_PYM||^2 + ||p_model-p_PYM||^2 ]

NRMSE_j(T) = sqrt(R_j(T)) / sigma_j*

where sigma_j* is a **fixed PYM-Core v0.3 target scale calibrated at T_ref=0.6 s** using the same centering rule used in 003K:

sigma_j* = sqrt(mean_t[||q-q_bar||^2 + ||p-p_bar||^2]).

The fixed sigma_j* is then reused at every horizon.

### Anti-contamination rule

The 003K constants
`ANCHOR_SIGMA={0.13582885408422112, 0.11707011915998188, 0.18485292671545464}`
and `epsilon=1.0726199716099455` belong to **PYM-Legacy** and are not valid 003L pass/fail constants merely because the numerical values already exist.

003L therefore freezes the **calibration rule before Level-1 implementation**, not a legacy-derived number:

1. generate v0.3 targets at T_ref=0.6 s;
2. freeze the three v0.3 sigma_j* values;
3. run Level 0 calibration under the optimization protocol below;
4. define A0 as the maximum OOS-inclusive fixed NRMSE of the preregistered Level-0 reference selected by TRAIN loss only;
5. freeze epsilon_003L = 1.05 * A0;
6. write sigma_j*, A0, epsilon_003L, run ID, commit SHA, and artifact digest to a calibration artifact **before any Level-1 result is generated**.

No threshold may be changed after Level-1 execution begins.

PASS at a tested (class,N,T) means every IC satisfies NRMSE_j(T) <= epsilon_003L. OOS values are audit-only until final PASS evaluation; they may not select parameters.

Dynamic-horizon sigma/NRMSE may be reported as diagnostics only and cannot determine PASS.

## 6. Shared optimizer budget

Independent optimization per tested cell:
- K = 20 deterministic random starts;
- base seed = 47000, starts seed = 47000+k;
- initialization: log-uniform within each parameter's frozen positive bounds;
- optimizer: L-BFGS-B;
- MAXITER = 300;
- gtol = 1e-6;
- ftol = 1e-9;
- candidate ranking: IC_1 TRAIN raw loss only;
- invalid/non-finite trajectories receive the common numerical sentinel and cannot constitute scientific PASS.

The same K and MAXITER budget is used across levels. Parameter-count differences are reported explicitly; equal optimizer calls do not imply equal effective search difficulty.

## 7. Capacity grid

Primary latent-capacity grid:
N in {1,2,3,4,6,8,12,16,24,32}.

All scheduled capacities are retained in the audit record. A failed independent search at one N does not prove algebraic incapacity, especially if larger/smaller N exhibits non-monotonic accessibility.

The reported quantity is **protocol-accessible capacity**, not a theorem about global minimal latent dimension.

## 8. Model-class ladder

### Level 0 — finite conservative linear bath

A finite Caldeira-Leggett-style linear auxiliary-oscillator bath with N real modes:

q_ddot = -q + sum_i c_i (y_i - q)  
y_i_ddot = -omega_i^2 (y_i - q)

with positive real omega_i and c_i under the repository's existing finite-bath parameterization and masses frozen to 1 unless the implementation audit shows a different exact existing force convention. The code-level equation must be copied verbatim into the Level-0 implementation report before results are interpreted.

Level 0 is conservative. Energy accounting applies to the complete particle+bath system, subject to numerical convergence.

Frozen positive bounds inherited from 003K:
- omega_i in [1e-2, 1e3]
- coupling_i in [1e-4, 1e2]

### Level 1 — damped/relaxing linear latent bath

Level 1 adds one non-negative damping/relaxation rate gamma_i per Level-0 latent mode while retaining linear coupling. The intended real-valued form is:

q_ddot = F_obs(q) + F_latent(q,y)  
y_i_ddot + 2 gamma_i y_i_dot + omega_i^2 (y_i - q) = 0

with real omega_i > 0 and gamma_i >= 0.

Complex frequencies may be used analytically to describe eigenvalues, but **complex omega is not an independently optimized physical parameter** in the preregistered Level-1 parameterization.

Before Level-1 execution, the implementation must freeze:
- the exact force convention, matching Level 0 when gamma=0;
- gamma bounds and initialization distribution;
- the numerical integrator for damped latent dynamics;
- reservoir/dissipation accounting.

The gamma=0 reduction must reproduce Level 0 to numerical tolerance on matched parameters before Level-1 optimization is allowed.

Because Level 1 is dissipative, a naive conservative-energy gate is prohibited. The report must separate mechanical/bath energy change from dissipative work/reservoir accounting.

### Level 2 — linear continuous latent state-space

Reserved, not implemented before Level 0/1 results are frozen:

z_dot = A z + B u,  
F_latent = C z.

Its parameterization, stability constraints, capacity mapping, and optimizer budget require a separate preregistration amendment before execution.

### Level 3 — nonlinear latent model

Reserved only if justified after Levels 0–2. No Level-3 architecture may be chosen using Level-3 test/OOS results.

## 9. Required gates before scientific comparison

1. PYM-Core v0.3 target replay is deterministic in the same environment.
2. T=0 release state is produced by shared prehistory, not hand-set latent memory.
3. IC_2/IC_3 do not participate in fitting or candidate ranking.
4. Level-0 complete-system numerical sanity/convergence is documented.
5. Level-1 gamma=0 reduction agrees with Level 0 on matched parameters.
6. Level-1 dissipation/reservoir accounting is explicit.
7. Same primary dt, horizons, cases, observable metric, K, optimizer, MAXITER, and capacity grid are used unless a preregistered class-specific necessity is documented before seeing comparative results.
8. Raw per-start results, failures, optimizer status, bounds hits, and selected TRAIN winner are archived.
9. Each official run records commit SHA, workflow run ID, environment, seeds, artifact ID, and SHA-256 artifact digest.

## 10. Interpretation policy

Allowed conclusions are scoped to the tested protocol.

A PASS supports: "This finite-dimensional model class numerically emulates the tested PYM-Core v0.3 observable trajectories over the stated histories, horizon, tolerance, and optimization protocol."

A FAIL supports only: "No passing emulator was found for this tested class/capacity/bounds/optimizer budget under this protocol."

Forbidden without additional proof:
- universal equivalence or universal non-equivalence;
- algebraic reduction inferred from numerical fit alone;
- a globally minimal latent dimension inferred from finite multistart optimization;
- a nonzero dt->0 limit inferred from a few finite refinements;
- new-physics claims;
- interpreting optimizer failure as structural impossibility.

## 11. Freeze boundary

This manifest is frozen before Level-1 implementation/results.

The only permitted pre-Level-1 calibration amendment is the mechanical insertion of the v0.3 T_ref sigma_j*, Level-0 A0, epsilon_003L, and their provenance generated by Section 5. Any other change to metric, histories, optimizer budget, model equations, bounds, PASS rule, or interpretation policy requires a new manifest version and must be made before inspecting the affected comparative results.
