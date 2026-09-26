import numpy as np
import pytest

from pym.physics.damped_multimode_bath import (
    DampedBathParameters,
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
