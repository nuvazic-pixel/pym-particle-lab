import numpy as np
from pym.core.engine import PYMEngine
from pym.core.state import PYMState
from pym.lab.experiment import run_double_history_experiment
from pym.physics.pym_forces import memory_force
from pym.physics.standard import harmonic_force


def test_lambda_zero_erases_history_effect():
    result = run_double_history_experiment(steps=500, dt=0.01)
    assert result.control.baseline_pass
    assert result.control.baseline_position_divergence < 1e-12
    assert result.control.baseline_momentum_divergence < 1e-12


def test_pym_history_produces_model_divergence():
    result = run_double_history_experiment(steps=500, dt=0.01, lambda_pym=0.1)
    assert result.pym_position_divergence > 0.0


def test_deterministic_replay():
    r1 = run_double_history_experiment(steps=300, dt=0.01, lambda_pym=0.1)
    r2 = run_double_history_experiment(steps=300, dt=0.01, lambda_pym=0.1)
    assert r1 == r2


def test_state_validation():
    try:
        PYMState(np.zeros(3), np.zeros(2))
        assert False
    except ValueError:
        pass
