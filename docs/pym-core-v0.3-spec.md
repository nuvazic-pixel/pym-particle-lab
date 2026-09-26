# PYM-Core v0.3 numerical specification

Status: preregistered implementation specification. The legacy v0.1 core remains untouched so Experiments 003G–003K stay reproducible.

## Scope

v0.3 is a numerical-hygiene revision of the phenomenological PYM model. It is not a claim of new physics and it is not an exact continuous-time solver for the full coupled system.

## Memory law

Define a neutral interaction signal J(q,p,t) and

M(t) = integral_0^t [exp(-(t-s)/tau_M)/tau_M] J(s) ds.

For a step of length dt, assuming J is held constant on that step,

M_{k+1} = a M_k + (1-a) J_k,
a = exp(-dt/tau_M).

This memory update is exact for piecewise-constant J. It does not make the coupled q,p,M trajectory an exact continuous solution.

Supported signal laws in v0.3:
- LINEAR_Q: J = q[0]
- LINEAR_P: J = p[0]
- NONLINEAR_Q_DOT_P: J = q dot p
- Q_SQUARED: J = q dot q

The signal remains a model hypothesis, not physical information.

## q,p integration and ordering

Use a symmetric kick-drift-kick (velocity-Verlet-like) step.

1. Evaluate J_k from (q_k,p_k).
2. Compute M_half using the exact exponential update over dt/2 with J_k.
3. Evaluate the force at the old state with M_half and apply a half kick.
4. Drift q for dt using the half-step momentum.
5. Evaluate J_{k+1} from the drifted q and half-step p.
6. Complete the memory update over dt/2 using J_{k+1}.
7. Evaluate force at the new position with M_{k+1} and apply the second half kick.

Because the memory force depends on a history state and J may depend on momentum, this splitting is a declared numerical scheme, not a proof of global second-order accuracy. Convergence is measured empirically at dt, dt/2 and dt/4.

## Configuration

v0.3 uses explicit tau_M > 0 rather than alpha. alpha remains legacy-only.

The memory-force scale is explicit. Dimensional interpretation must be declared by an experiment; v0.3 does not silently assign physical units to J or M.

## Required gates

1. lambda_pym=0: memory-force coupling has no effect on q,p.
2. Deterministic replay: identical configuration gives identical arrays in the same environment.
3. Memory recurrence agrees with the analytic piecewise-constant-J recurrence.
4. dt refinement: endpoint/trajectory differences are reported for dt, dt/2, dt/4.
5. Work accounting: report the discrete memory-force work separately from conservative mechanical energy.
6. Legacy regression: existing v0.1 modules and tests are not modified by the v0.3 implementation.

## Comparison gate before 003L

Run legacy and v0.3 references at dt, dt/2 and dt/4 under matched initial conditions. If their discrepancy tends to zero, part of the prior difference is attributable to discretization. If it approaches a nonzero limit, v0.3 defines a changed model and 003K and 003L must not be described as direct like-for-like comparisons.

## 003L model-class ladder

Only after the comparison gate:

- Level 0: finite conservative linear bath (003K class).
- Level 1: damped/relaxing latent bath. Dissipation/reservoir accounting replaces a naive conservative-energy gate.
- Level 2: linear latent continuous state-space Markovianization, z_dot = A z + B u, F_rep = C z.
- Level 3: nonlinear latent/memory model, only if Levels 0–2 leave a robust residual.

All levels use the same prehistory/OOS split and frozen evaluation metric where applicable. Capacity and optimizer budgets must be reported. Failure means mismatch for the tested class/protocol only.
