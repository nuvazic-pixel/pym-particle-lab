import numpy as np

from pym.lab.experiment_003l_level1 import (
    CAPACITIES, EPSILON_003L, GAMMA_BOUNDS, HORIZONS, K, MAXITER,
    artifact_path, horizon_token, params_from_log, random_x0, prepare_level1,
)


def test_level1_protocol_constants_are_frozen():
    assert HORIZONS == (0.3, 0.6, 1.0, 2.0, 3.0, 5.0)
    assert CAPACITIES == (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert GAMMA_BOUNDS == (1e-3, 1e2)
    assert K == 20
    assert MAXITER == 300
    assert EPSILON_003L == 0.00497101989167444


def test_level1_rng_is_deterministic_and_block_ordered():
    n = 3
    a = random_x0(n, 47000)
    b = random_x0(n, 47000)
    assert np.array_equal(a, b)
    p = params_from_log(a, n)
    assert np.all((p.omega >= 1e-2) & (p.omega <= 1e3))
    assert np.all((p.coupling >= 1e-4) & (p.coupling <= 1e2))
    assert np.all((p.gamma >= 1e-3) & (p.gamma <= 1e2))


def test_level1_artifact_names_have_no_decimal_horizon():
    assert horizon_token(0.3) == "T0300ms"
    assert horizon_token(5.0) == "T5000ms"
    assert artifact_path(0.6, 3).name == "cell_T0600ms_N03.json"
    assert "." not in artifact_path(0.6, 3).stem


def test_gamma_zero_prehistory_is_exact_level0_reduction():
    from pym.lab.experiment_003l_level0 import CASES, DT, PREP_TIME
    from pym.lab.prehistory import prepare_bath
    from pym.physics.damped_multimode_bath import DampedBathParameters
    from pym.physics.multimode_bath import BathParameters

    p0 = BathParameters(
        mass=np.ones(3),
        omega=np.array([2.0, 5.0, 9.0]),
        coupling=np.array([0.4, 0.8, 0.3]),
    )
    p1 = DampedBathParameters(
        p0.mass, p0.omega, p0.coupling, np.zeros(3)
    )
    prep_steps = int(round(PREP_TIME / DT))

    for _, q0, p0_obs, protocol in CASES:
        expected = prepare_bath(p0, q0, p0_obs, DT, prep_steps, protocol)
        got = prepare_level1(p1, q0, p0_obs, protocol)
        assert expected is not None and got is not None
        assert np.array_equal(got.q, expected.q)
        assert np.array_equal(got.p, expected.p)
        assert np.array_equal(got.y, expected.y)
        assert np.array_equal(got.py, expected.py)
