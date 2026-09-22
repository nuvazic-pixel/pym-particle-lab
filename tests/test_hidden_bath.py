from pym.lab.model_comparison import run_hidden_bath_pair


def test_hidden_bath_is_deterministic():
    assert run_hidden_bath_pair(steps=200, dt=1e-3) == run_hidden_bath_pair(steps=200, dt=1e-3)


def test_hidden_bath_total_energy_converges_with_dt():
    coarse = run_hidden_bath_pair(steps=1000, dt=1e-3)
    fine = run_hidden_bath_pair(steps=2000, dt=5e-4)
    assert fine.relative_total_energy_drift <= coarse.relative_total_energy_drift
