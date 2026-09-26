# Experiment 003L — Level-1 preregistration amendment

**Status:** FROZEN BEFORE ANY LEVEL-1 OPTIMIZATION RESULT  
**Parent manifest:** `docs/experiment-003l-manifest.md` at commit `dfbb87d45aecb5cec9d18f0e1f84fefd41c4aca3`  
**Level-0 calibration:** SEALED at commit `3efeb8a82eb6129773c115fd62c604527d4780d6`, run `36239111717`, artifact `10904513491`, digest `sha256:330b7850f72472fe7ac9344b87bdbc8cc322fe671eeb829d40686f3cfd822a82`.

Frozen native scales:
- sigma*(IC_1) = 0.196199716029399
- sigma*(IC_2) = 0.15779818649314112
- sigma*(IC_3) = 0.2509655101723989
- A0 = 0.004734304658737562
- epsilon_003L = 0.00497101989167444

This amendment supplies the class-specific choices that the parent manifest explicitly required before Level-1 execution. It does not change the target, histories, metric, PASS rule, capacity grid, K, optimizer, or interpretation policy.

## L1. Exact implemented class

Level 1 uses the existing Level-0 shifted-oscillator force convention unchanged:

V_i = 0.5 m_i omega_i^2 ( y_i - c_i q/(m_i omega_i^2) )^2.

The conservative forces are exactly those in `pym.physics.multimode_bath.forces`. Level 1 adds only

d p_yi / dt |_damp = -2 gamma_i p_yi,  gamma_i >= 0.

The numerical step is the committed Strang split in `pym.physics.damped_multimode_bath`:

1. exact damping half-step;
2. the existing Level-0 velocity-Verlet step;
3. exact damping half-step.

For gamma_i=0, the implementation delegates directly to the Level-0 step. The reduction gate requires exact array equality in the same environment.

The already executed CI before this amendment contained unit/gate tests only; no Level-1 parameter optimization or comparative Level-1 result had been generated.

## L2. Frozen gamma search domain

The primary Level-1 damping domain is

gamma_i in [1e-3, 1e2] s^-1.

Initialization is independent log-uniform sampling:

log(gamma_i) ~ Uniform(log(1e-3), log(1e2)).

Rationale fixed before optimization: this spans damping e-folding time constants for the latent momentum term from approximately 500 s down to 0.005 s because p_y decays as exp(-2 gamma t). The tested preparation+release windows are O(1 s), so the interval deliberately includes effectively weak damping and strongly damped behavior without using gamma=0 as a log-space endpoint.

**Boundary rule:** gamma=0 remains a structural reduction/control value only. It is not a random-start optimization value. A Level-1 fit may approach but cannot equal the Level-0 subspace under the primary positive log-bound.

No bound expansion, contraction, or alternate gamma initialization may be introduced after Level-1 results are inspected under this preregistration. A later bound study, if scientifically needed, must be explicitly labeled a new/post-hoc experiment.

## L3. Other parameter domains

For every mode:
- mass_i = 1, frozen;
- omega_i in [1e-2, 1e3], log-uniform initialization;
- coupling_i in [1e-4, 1e2], log-uniform initialization;
- gamma_i in [1e-3, 1e2], log-uniform initialization.

All positive parameters are optimized in log coordinates with L-BFGS-B.

## L4. Optimization and selection

For each preregistered (T,N) cell:
- N in {1,2,3,4,6,8,12,16,24,32};
- T in {0.3,0.6,1.0,2.0,3.0,5.0} s;
- dt = 0.001 s;
- T_prep = 1.0 s;
- K = 20 independent starts;
- seed_k = 47000+k;
- MAXITER = 300;
- gtol = 1e-6;
- ftol = 1e-9;
- ranking and parameter selection use IC_1 raw loss only;
- IC_2 and IC_3 are never used to rank starts, tune bounds, stop optimization, or choose hyperparameters.

To keep a deterministic per-seed initialization while extending the parameter vector, a single RNG seeded by seed_k draws, in order, N log-omega values, then N log-coupling values, then N log-gamma values from their respective uniform log bounds.

## L5. PASS and reported capacity

For each selected TRAIN winner:

NRMSE_j(T) = sqrt(R_j(T)) / sigma*_j.

PASS iff all three cases satisfy

NRMSE_j(T) <= 0.00497101989167444.

The smallest passing N found by the frozen finite search is reported as **protocol-accessible capacity**, not a globally minimal latent dimension. Non-monotonic PASS/FAIL across N is treated as optimizer-accessibility evidence and does not establish representational impossibility.

## L6. Dissipation accounting gate

The reservoir diagnostic uses

P_res = sum_i 2 gamma_i |p_yi|^2 / m_i >= 0.

Reservoir energy is accumulated by trapezoidal power quadrature. It is explicitly a convergence-tested numerical diagnostic, not an exact discrete conservation identity.

Level 1 must report mechanical+bath energy and reservoir accumulation separately. Naive mechanical-energy conservation is not a Level-1 gate.

## L7. Execution freeze

Before the first Level-1 optimization workflow is launched:
1. unit tests for gamma>=0, non-negative dissipation power, and gamma=0 exact Level-0 reduction must pass;
2. the optimization harness must encode the constants in this amendment;
3. workflow artifacts must preserve all K starts, optimizer status, selected TRAIN winner, OOS metrics, bounds hits/parameters, commit SHA, run ID, artifact ID and digest.

After the first Level-1 optimization result exists, this protocol is frozen. Any altered gamma bounds, seeds, metric, threshold, histories, optimizer budget, integrator, or model equation defines a separate experiment/version.
