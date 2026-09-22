# Interpretation Policy

The PYM Particle Lab reports evidence against explicitly tested model classes, never absolute irreducibility.

## Misspecification gap

A plateau in out-of-sample residuals is interpreted as a candidate misspecification gap relative to the tested bath family and fitting protocol.

It is not, by itself, evidence for new physics.

Before a plateau can be treated as structural, the project must test:

- timestep and integrator convergence
- optimizer multi-start and failure modes
- wider frequency/coupling bounds
- increasing bath capacity
- alternative spectral-density parameterizations
- held-out initial conditions
- nonlinear conventional bath families
- uncertainty / numerical precision

## Linear vs nonlinear signals

A linear PYM signal can often be represented by an appropriate linear memory realization, but the repository does **not** claim a universal finite-N equivalence theorem without specifying the kernel, admissible bath class, initial-state preparation, and approximation norm.

Likewise, a nonlinear PYM signal producing a plateau against a linear bath demonstrates mismatch with that linear ansatz. It does not establish that no conventional nonlinear Hamiltonian environment can represent the same reduced dynamics.
