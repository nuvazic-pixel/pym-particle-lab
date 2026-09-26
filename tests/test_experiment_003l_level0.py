import numpy as np

from pym.lab.experiment_003l_level0 import (
    CASES, DT, K, MAXITER, REFERENCE_N, T_REF, V03_CONFIG,
    prepare_v03, release_v03, sigma_target,
)


def test_003l_calibration_protocol_is_frozen():
    assert DT == 0.001
    assert T_REF == 0.6
    assert K == 20
    assert MAXITER == 300
    assert REFERENCE_N == 3
    assert [c[0] for c in CASES] == ["IC_1", "IC_2", "IC_3"]
    assert V03_CONFIG.signal_law.value == "LINEAR_Q"


def test_003l_target_is_native_v03_and_finite():
    name, q0, p0, protocol = CASES[0]
    s = prepare_v03(q0, p0, protocol)
    q, p = release_v03(s, T=0.01)
    assert q.shape == p.shape == (10, 1)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(p))
    assert np.isfinite(s.memory)


def test_003l_sigma_rule_positive():
    name, q0, p0, protocol = CASES[0]
    s = prepare_v03(q0, p0, protocol)
    q, p = release_v03(s, T=0.01)
    assert sigma_target((q, p)) > 0.0
