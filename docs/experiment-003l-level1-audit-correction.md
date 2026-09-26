# Experiment 003L Level-1 — Implementation Audit Correction

Status: **pre-result implementation correction**

## Scope

The preregistered Level-1 equation is

```text
d p_yi / dt |_damp = -2 gamma_i p_yi
```

with exact propagation over an interval `h`:

```text
p_yi(t+h) = p_yi(t) exp(-2 gamma_i h)
```

An implementation audit performed before scientific interpretation of the
Level-1 matrix found that the original Strang helper applied
`exp(-gamma*h)` while callers supplied `h = dt/2`. Two half-steps therefore
implemented a full-step damping factor `exp(-gamma*dt)`, inconsistent with
the preregistered ODE and with the reservoir-power diagnostic
`2 gamma |p_y|^2 / m`.

## Disposition of the original Level-1 run

The original artifacts are retained for provenance. They characterize the
implemented effective damping convention, but they are **not admissible as
the preregistered Level-1 result**.

No scientific conclusion from the original Level-1 matrix was used to choose
or tune this correction.

## Correction

The damping propagator now applies

```text
exp(-2 gamma h)
```

over every supplied interval `h`. The Strang calls remain at `h=dt/2`,
so two damping-only half-steps compose to `exp(-2 gamma dt)`.

## Frozen protocol invariants

This correction does **not** change:

- target PYM-Core v0.3 configuration;
- horizons or capacity grid;
- dt or preparation duration;
- omega, coupling, or gamma bounds;
- K=20 starts or seed schedule;
- L-BFGS-B settings;
- fixed sigma values, A0, or epsilon_003L;
- TRAIN-only winner selection or OOS PASS rule.

## Regression gates

1. Exact analytic damping propagation for `d p_y/dt=-2 gamma p_y`.
2. Composition of two half damping steps equals the exact full damping step.
3. `gamma=0` Level-1 prehistory is array-identical to Level-0
   `prepare_bath` for every preregistered IC/preparation protocol.
4. Existing `gamma=0` release-step reduction gate remains in force.

A fresh Level-1-v2 matrix must be generated from the corrected, tested commit
before Level-1 scientific interpretation.
