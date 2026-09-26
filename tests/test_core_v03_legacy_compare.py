from pym.lab.core_v03_legacy_compare import report


def test_legacy_v03_report_is_finite_and_explicit_about_model_change():
    r = report(T=0.2, dts=(0.004, 0.002, 0.001))
    assert len(r["rows"]) == 3
    assert "Legacy J" in r["critical_model_difference"]
    for row in r["rows"]:
        assert row["rms_qp"] >= 0.0
        assert row["rms_qpm"] >= 0.0


def test_comparison_does_not_claim_asymptotic_equivalence():
    r = report(T=0.2)
    assert r["observed_trend"] in {
        "DECREASING_OVER_TESTED_REFINEMENT",
        "NONDECREASING_OR_DIVERGING_OVER_TESTED_REFINEMENT",
        "INCONCLUSIVE_OVER_TESTED_REFINEMENT",
    }
    assert "cannot establish" in r["interpretation_guardrail"]
