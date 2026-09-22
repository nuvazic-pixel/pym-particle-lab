import numpy as np
from pym.lab.equivalence import fit_bath, oos_loss
from pym.physics.multimode_bath import BathParameters, simulate


def test_multimode_energy_convergence():
    params = BathParameters(np.ones(2), np.array([0.7, 1.4]), np.array([0.1, 0.15]))
    q0, p0 = np.array([1.0]), np.array([0.0])
    _, _, coarse = simulate(params, q0, p0, 1000, 1e-3)
    _, _, fine = simulate(params, q0, p0, 2000, 5e-4)
    assert fine <= coarse + 1e-14


def test_fit_and_oos_are_deterministic():
    true_params = BathParameters(np.ones(1), np.array([1.2]), np.array([0.2]))
    q0, p0 = np.array([1.0]), np.array([0.1])
    tq, tp, _ = simulate(true_params, q0, p0, 80, 2e-3)
    fit1 = fit_bath(tq, tp, q0, p0, 2e-3, 1, maxiter=20)
    fit2 = fit_bath(tq, tp, q0, p0, 2e-3, 1, maxiter=20)
    assert fit1.train_loss == fit2.train_loss
    q1, p1 = np.array([0.8]), np.array([-0.05])
    oq, op, _ = simulate(true_params, q1, p1, 80, 2e-3)
    assert oos_loss(fit1, oq, op, q1, p1, 2e-3) >= 0.0
