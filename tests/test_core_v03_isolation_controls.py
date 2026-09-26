from pym.lab.core_v03_isolation_controls import controls


def test_controls_separate_one_variable_at_a_time():
    r = controls(T=0.2)
    a = r["control_A_integrator_only"]
    b = r["control_B_model_definition"]
    assert "ordering" in a["varied"]
    assert "J=" in b["varied"]
    assert len(a["rows"]) == 3
    assert len(b["rows"]) == 3


def test_integrator_only_difference_refines():
    r = controls(T=0.4)
    rows = r["control_A_integrator_only"]["rows"]
    assert rows[-1]["rms_qp"] < rows[0]["rms_qp"]


def test_model_definition_control_is_not_mislabeled_integrator_error():
    r = controls(T=0.2)
    assert r["control_B_model_definition"]["varied"] == "J=||F_standard||*dt vs J=q[0]"
    assert "legacy dt factor" in r["guardrail"]
