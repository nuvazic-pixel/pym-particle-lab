import numpy as np

from pym.lab.core_v03_convergence import convergence_report, run_v03
from pym.core.v03 import CoreV03Config


def test_lambda_zero_harmonic_energy_refines():
    cfg = CoreV03Config(lambda_pym=0.0)
    coarse = run_v03(0.01, 1.0, cfg)
    fine = run_v03(0.005, 1.0, cfg)
    assert abs(fine["mechanical_energy_change"]) < abs(coarse["mechanical_energy_change"])


def test_coupled_convergence_error_decreases():
    r = convergence_report(T=0.4, dts=(0.004, 0.002, 0.001))
    e1 = r["pair_errors"]["0.004_vs_0.002"]["rms_qpm"]
    e2 = r["pair_errors"]["0.002_vs_0.001"]["rms_qpm"]
    assert e2 < e1


def test_work_balance_refines():
    cfg = CoreV03Config(tau_M=1.0, lambda_pym=0.05)
    coarse = run_v03(0.004, 0.4, cfg)
    fine = run_v03(0.002, 0.4, cfg)
    assert abs(fine["work_balance_residual"]) < abs(coarse["work_balance_residual"])
