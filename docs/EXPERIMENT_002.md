# Experiment 002 — PYM Equivalence Challenge

## Question

Can one parameter set fitted to a PYM trajectory generalize to unseen initial conditions without refitting?

## Protocol

1. Generate a PYM training trajectory from IC-1.
2. Fit an N-mode conventional hidden-environment model on IC-1.
3. Freeze all fitted bath parameters.
4. Generate PYM trajectories for IC-2...IC-K.
5. Evaluate the frozen bath on those same observable initial conditions.
6. Record train and out-of-sample residuals for N = 1, 2, 4, 8, 16, 32 where computationally practical.
7. Repeat with tighter dt and multiple deterministic optimizer initializations before interpreting saturation.

## Metrics

Loss combines position and momentum residuals:

L = mean(||q_PYM-q_bath||^2 + beta ||p_PYM-p_bath||^2)

Report train loss and every OOS loss separately. Do not collapse them into a discovery score.

## Interpretation boundary

- Falling OOS residual with increasing model capacity supports representability within the tested bath family.
- Saturating OOS residual is evidence only against the **tested model class and fitting protocol**.
- It does not prove irreducibility against every Hamiltonian environment, nonlinear bath, continuum spectral density, or hidden-variable realization.
- Optimization failure must be separated from model-class failure.

## Controls

- positive bath parameters enforced in log-space
- deterministic initialization
- no OOS refitting
- total energy accounting for the full conservative bath
- dt convergence
- later: multi-start fitting, held-out IC suite, spectral-density families, uncertainty intervals
