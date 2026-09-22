from pym.lab.experiment import run_double_history_experiment


def main() -> None:
    result = run_double_history_experiment()
    print("PYM Particle Lab — Experiment 001: Double History")
    print(f"Control Δx: {result.control.baseline_position_divergence:.12e}")
    print(f"Control Δp: {result.control.baseline_momentum_divergence:.12e}")
    print(f"Control gate: {'PASS' if result.control.baseline_pass else 'FAIL'}")
    print(f"PYM Δx:     {result.pym_position_divergence:.12e}")
    print(f"PYM Δp:     {result.pym_momentum_divergence:.12e}")
    print(f"Control energy drift A: {result.control_energy_drift_a:.12e}")
    print(f"PYM physical-subsystem energy drift A: {result.pym_energy_drift_a:.12e}")
    print("Interpretation: divergence is a property of the explicit hypothesis model, not evidence of new physics.")


if __name__ == "__main__":
    main()
