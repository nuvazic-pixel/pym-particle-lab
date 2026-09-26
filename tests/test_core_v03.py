import numpy as np

from pym.core.v03 import (
    CoreV03Config, CoreV03State, PYMEngineV03, SignalLaw,
    memory_exact_piecewise_constant,
)


def zero_force(s):
    return np.zeros_like(s.position)


def harmonic_force(s):
    return -s.position


def test_exact_memory_constant_signal():
    M0, J, tau, dt = 0.3, 2.0, 1.7, 0.4
    got = memory_exact_piecewise_constant(M0, J, dt, tau)
    expected = J + (M0 - J) * np.exp(-dt / tau)
    assert np.isclose(got, expected, rtol=0.0, atol=1e-15)


def test_lambda_zero_decouples_memory_from_trajectory():
    e = PYMEngineV03()
    a = CoreV03State([1.0], [0.2], memory=0.0)
    b = CoreV03State([1.0], [0.2], memory=9.0)
    cfg = CoreV03Config(tau_M=1.0, lambda_pym=0.0, signal_law=SignalLaw.Q_SQUARED)
    for _ in range(100):
        a = e.step(a, 0.01, harmonic_force, cfg)
        b = e.step(b, 0.01, harmonic_force, cfg)
    assert np.array_equal(a.position, b.position)
    assert np.array_equal(a.momentum, b.momentum)


def test_deterministic_replay_same_environment():
    e = PYMEngineV03()
    cfg = CoreV03Config(tau_M=0.8, lambda_pym=0.05)
    def run():
        s = CoreV03State([0.7], [-0.1])
        trace = []
        for _ in range(200):
            s = e.step(s, 0.002, harmonic_force, cfg)
            trace.append((s.position[0], s.momentum[0], s.memory))
        return np.asarray(trace)
    assert np.array_equal(run(), run())


def test_zero_force_free_particle_when_lambda_zero():
    e = PYMEngineV03()
    s = CoreV03State([1.0], [2.0], mass=2.0)
    cfg = CoreV03Config(lambda_pym=0.0)
    out = e.step(s, 0.25, zero_force, cfg)
    assert np.isclose(out.position[0], 1.25)
    assert np.isclose(out.momentum[0], 2.0)
