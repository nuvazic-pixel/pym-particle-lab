from pym.lab.runner import run_equivalence_challenge


def test_runner_small_matrix_is_deterministic():
    a = run_equivalence_challenge(modes=(1,), steps=30, dt=0.002, maxiter=3)
    b = run_equivalence_challenge(modes=(1,), steps=30, dt=0.002, maxiter=3)
    assert a == b
    assert set(a[0].oos_losses) == {"IC_2", "IC_3"}


def test_no_oos_refit_surface():
    rows = run_equivalence_challenge(modes=(1,), steps=20, dt=0.002, maxiter=2)
    assert rows[0].train_loss >= 0.0
    assert all(v >= 0.0 for v in rows[0].oos_losses.values())
