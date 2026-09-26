import numpy as np

from pym.lab.experiment_003l_level1 import (
    CAPACITIES, EPSILON_003L, GAMMA_BOUNDS, HORIZONS, K, MAXITER,
    artifact_path, horizon_token, params_from_log, random_x0,
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
