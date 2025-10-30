use logosq::circuits::Circuit;
use logosq::optimization::ansatz::*;
use logosq::states::State;
use std::f64::consts::PI;

mod tests {
    use super::*;
    // ============================================================================
    // ANSATZ TESTS
    // ============================================================================

    #[test]
    fn test_hardware_efficient_ansatz_creation() {
        let ansatz =
            HardwareEfficientAnsatz::new(3, 2, EntanglingGate::CNOT, EntanglingPattern::Linear);

        assert_eq!(ansatz.num_qubits(), 3);
        assert_eq!(ansatz.num_parameters(), 9); // 3 qubits × (2 layers + 1 final) = 9
        assert_eq!(ansatz.depth, 2);
    }

    #[test]
    fn test_hardware_efficient_ansatz_linear_pattern() {
        let ansatz =
            HardwareEfficientAnsatz::new(4, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 8 parameters
        let circuit = ansatz.build_circuit(&params);

        assert_eq!(circuit.num_qubits, 4);
        // Should have: 4 RY gates + 3 CNOTs + 4 RY gates = 11 gates
        // assert!(circuit.num_gates() > 0);
    }

    #[test]
    fn test_hardware_efficient_ansatz_circular_pattern() {
        let ansatz =
            HardwareEfficientAnsatz::new(3, 1, EntanglingGate::CNOT, EntanglingPattern::Circular);

        assert_eq!(ansatz.num_parameters(), 6); // 3 × (1 + 1)
    }

    #[test]
    fn test_hardware_efficient_ansatz_full_pattern() {
        let ansatz =
            HardwareEfficientAnsatz::new(3, 1, EntanglingGate::CNOT, EntanglingPattern::Full);

        let params = vec![0.1; 6];
        let circuit = ansatz.build_circuit(&params);

        // Full pattern on 3 qubits: 3 RY + 3 CNOTs (all pairs) + 3 RY
        assert_eq!(circuit.operations.len(), 9);
    }

    #[test]
    fn test_hardware_efficient_ansatz_apply() {
        let ansatz =
            HardwareEfficientAnsatz::new(2, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let params = vec![PI / 2.0; 4]; // 2 × (1 + 1)
        let mut state = State::zero_state(2);
        ansatz.apply(&mut state, &params);

        // Check state is normalized
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_amplitudes_ansatz() {
        let ansatz = RealAmplitudesAnsatz::new(3, 2);

        assert_eq!(ansatz.num_qubits(), 3);
        assert_eq!(ansatz.num_parameters(), 9); // 3 × (2 + 1)

        let params = vec![0.1; 9];
        let circuit = ansatz.build_circuit(&params);
        assert_eq!(circuit.num_qubits, 3);
    }

    #[test]
    fn test_efficient_su2_ansatz() {
        let ansatz = EfficientSU2Ansatz::new(2, 1);

        assert_eq!(ansatz.num_qubits(), 2);
        assert_eq!(ansatz.num_parameters(), 6); // 2 qubits × 3 rotations × 1 layer

        let params = vec![0.1; 6];
        let mut state = State::zero_state(2);
        ansatz.apply(&mut state, &params);

        // assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_parameterized_circuit_custom() {
        let builder = |params: &[f64]| -> Circuit {
            let mut circuit = Circuit::new(2);
            circuit.ry(0, params[0]);
            circuit.ry(1, params[1]);
            circuit.cnot(0, 1);
            circuit.ry(0, params[2]);
            circuit
        };

        let ansatz = ParameterizedCircuit::new(2, 3, builder);
        assert_eq!(ansatz.num_parameters(), 3);

        let params = vec![0.5, 1.0, 1.5];
        let circuit = ansatz.build_circuit(&params);
        assert_eq!(circuit.num_qubits, 2);
    }

    #[test]
    #[should_panic(expected = "Parameter count mismatch")]
    fn test_ansatz_wrong_parameter_count() {
        let ansatz =
            HardwareEfficientAnsatz::new(2, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let wrong_params = vec![0.1, 0.2]; // Should be 4 parameters
        let _circuit = ansatz.build_circuit(&wrong_params);
    }
}
