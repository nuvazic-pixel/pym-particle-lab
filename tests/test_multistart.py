import numpy as np

from pym.lab.multistart_runner import (
    COUPLING_BOUNDS,
    OMEGA_BOUNDS,
    fit_one_start,
    unpack_fixed_mass,
)


def test_fixed_mass_unpack_and_bounds():
    x = np.log(np.array([0.1, 10.0, 1e-3, 2.0]))
    params = unpack_fixed_mass(x, 2)
    assert np.all(params.mass == 1.0)
    assert np.allclose(params.omega, [0.1, 10.0])
    assert np.allclose(params.coupling, [1e-3, 2.0])


def test_multistart_single_seed_is_deterministic_and_bounded():
    target_q = np.zeros((8, 1))
    target_p = np.zeros((8, 1))
    args = (target_q, target_p, np.array([0.0]), np.array([0.0]), 0.001, 1, 42000)
    r1, p1 = fit_one_start(*args, maxiter=2)
    r2, p2 = fit_one_start(*args, maxiter=2)
    assert r1.fun == r2.fun
    assert np.array_equal(p1.omega, p2.omega)
    assert np.array_equal(p1.coupling, p2.coupling)
    assert OMEGA_BOUNDS[0] <= p1.omega[0] <= OMEGA_BOUNDS[1]
    assert COUPLING_BOUNDS[0] <= p1.coupling[0] <= COUPLING_BOUNDS[1]
