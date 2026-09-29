import numpy as np
import pytest

from pym.physics.damped_multimode_bath import (
    DampedBathParameters,
    _damping_step,
    instantaneous_dissipation_power,
    step_damped_bath,
    step_with_reservoir_accounting,
)
from pym.physics.multimode_bath import (
    BathParameters,
    MultiModeState,
    step_velocity_verlet,
    total_energy,
)


def _state():
    return MultiModeState(
        q=np.array([0.7]),
        p=np.array([-0.1]),
        y=np.array([[0.2], [-0.3]]),
        py=np.array([[0.4], [-0.2]]),
        particle_mass=1.0,
    )


def _level0():
    return BathParameters(
        mass=np.ones(2),
        omega=np.array([2.0, 5.0]),
        coupling=np.array([0.4, 0.8]),
    )


def test_gamma_zero_is_exact_level0_reduction():
    s = _state()
    p0 = _level0()
    p1 = DampedBathParameters(p0.mass, p0.omega, p0.coupling, np.zeros(2))
    a = step_velocity_verlet(s, p0, 0.001)
    b = step_damped_bath(s, p1, 0.001)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.p, b.p)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.py, b.py)


def test_gamma_must_be_nonnegative():
    p0 = _level0()
    with pytest.raises(ValueError):
        DampedBathParameters(p0.mass, p0.omega, p0.coupling, np.array([0.0, -0.1]))


def test_dissipation_power_is_nonnegative():
    p0 = _level0()
    p1 = DampedBathParameters(p0.mass, p0.omega, p0.coupling, np.array([0.3, 0.7]))
    assert instantaneous_dissipation_power(_state(), p1) >= 0.0


def test_reservoir_accounting_accumulates_nonnegative_energy():
    p0 = _level0()
    p1 = DampedBathParameters(p0.mass, p0.omega, p0.coupling, np.array([0.3, 0.7]))
    s = _state()
    e0 = total_energy(s, p0)
    s1, reservoir, e1 = step_with_reservoir_accounting(s, p1, 0.001)
    assert reservoir >= 0.0
    assert np.isfinite(e1)
    assert np.isfinite(e1 + reservoir - e0)


def test_damping_step_matches_preregistered_exact_solution():
    py = np.array([[0.4], [-0.2]])
    gamma = np.array([0.3, 0.7])
    h = 0.017
    got = _damping_step(py, gamma, h)
    expected = py * np.exp(-2.0 * gamma[:, None] * h)
    assert np.array_equal(got, expected)


def test_two_half_damping_steps_equal_full_exact_propagation():
    py = np.array([[0.4], [-0.2]])
    gamma = np.array([0.3, 0.7])
    dt = 0.01
    half = _damping_step(py, gamma, 0.5 * dt)
    got = _damping_step(half, gamma, 0.5 * dt)
    expected = py * np.exp(-2.0 * gamma[:, None] * dt)
    assert np.allclose(got, expected, rtol=1e-15, atol=0.0)


def test_damping_substep_matches_analytic_decay():
    from pym.physics.damped_multimode_bath import _damping_half_step
    py = np.array([[2.0], [-3.0]])
    gamma = np.array([0.25, 1.5])
    h = 0.037
    got = _damping_half_step(py, gamma, h)
    expected = py * np.exp(-2.0 * gamma[:, None] * h)
    assert np.array_equal(got, expected)


def test_gamma_zero_prehistory_is_bit_identical_to_level0():
    from pym.lab.experiment_003l_level0 import CASES, DT, PREP_TIME
    from pym.lab.prehistory import prepare_bath
    from pym.lab.experiment_003l_level1 import prepare_level1

    p0 = _level0()
    p1 = DampedBathParameters(p0.mass, p0.omega, p0.coupling, np.zeros(2))
    _, q0, mom0, protocol = CASES[0]
    a = prepare_bath(p0, q0, mom0, DT, int(round(PREP_TIME / DT)), protocol)
    b = prepare_level1(p1, q0, mom0, protocol)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.p, b.p)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.py, b.py)
