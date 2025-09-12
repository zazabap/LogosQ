use logosq::states::State;
use logosq::MatrixGate;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

mod tests {
    use super::*;

    #[test]
    fn test_state_creation_and_normalization() {
        // Create a non-normalized state
        let vector = Array1::from_vec(vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]);

        let state = State::new(vector, Some(1));

        // Should be normalized
        assert!((state.vector[0].norm_sqr() - 1.0).abs() < 1e-10);
        assert_eq!(state.num_qubits, 1);

        // Test automatic qubit count calculation
        let vector = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);

        let state = State::new(vector, None);
        assert_eq!(state.num_qubits, 2); // 4 = 2^2 amplitudes

        // Test more than 2 qubits
        let vector_1 = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0), // |000⟩
            Complex64::new(0.0, 0.0), // |001⟩
            Complex64::new(0.0, 0.0), // |010⟩
            Complex64::new(0.0, 0.0), // |011⟩
            Complex64::new(0.0, 0.0), // |100⟩
            Complex64::new(0.0, 0.0), // |101⟩
            Complex64::new(0.0, 0.0), // |110⟩
            Complex64::new(0.0, 0.0), // |111⟩
        ]);

        let state_1 = State::new(vector_1, Some(3));
        assert!((state_1.vector[0].norm_sqr() - 1.0).abs() < 1e-10);
        assert_eq!(state_1.num_qubits, 3); // 8 = 2^3
    }

    #[test]
    fn test_zero_state() {
        let state = State::zero_state(2);

        assert_eq!(state.vector.len(), 4);
        assert_eq!(state.num_qubits, 2);
        assert!((state.vector[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        for i in 1..4 {
            assert!((state.vector[i] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_one_state() {
        let state = State::one_state(2);

        assert_eq!(state.vector.len(), 4);
        assert_eq!(state.num_qubits, 2);
        assert!((state.vector[3] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        for i in 0..3 {
            assert!((state.vector[i] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_plus_state() {
        let state = State::plus_state(2);
        let expected_amplitude = 0.5; // 1/√4 = 1/2

        assert_eq!(state.vector.len(), 4);
        assert_eq!(state.num_qubits, 2);

        for i in 0..4 {
            assert!((state.vector[i] - Complex64::new(expected_amplitude, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_bell_state() {
        let state = State::bell_state();
        let expected_amplitude = 1.0 / SQRT_2;

        assert_eq!(state.vector.len(), 4);
        assert_eq!(state.num_qubits, 2);

        assert!((state.vector[0] - Complex64::new(expected_amplitude, 0.0)).norm() < 1e-10);
        assert!((state.vector[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state.vector[2] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state.vector[3] - Complex64::new(expected_amplitude, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_deterministic_measurement() {
        // State |0⟩
        let state = State::zero_state(1);
        assert_eq!(state.measure(), 0);

        // State |1⟩
        let state = State::one_state(1);
        assert_eq!(state.measure(), 1);
    }

    #[test]
    fn test_measure_qubit() {
        // For |0⟩, measuring qubit 0 should always give 0
        let mut state = State::zero_state(1);
        assert_eq!(state.measure_qubit(0), 0);

        // For |1⟩, measuring qubit 0 should always give 1
        let mut state = State::one_state(1);
        assert_eq!(state.measure_qubit(0), 1);

        // For |+⟩ state, after measurement, state should collapse to |0⟩ or |1⟩
        let mut state = State::plus_state(1);
        let result = state.measure_qubit(0);
        assert!(result == 0 || result == 1);

        if result == 0 {
            assert!((state.vector[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
            assert!((state.vector[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        } else {
            assert!((state.vector[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
            assert!((state.vector[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_probability() {
        // For |0⟩, P(|0⟩) = 1, P(|1⟩) = 0
        let state = State::zero_state(1);
        assert!((state.probability(0) - 1.0).abs() < 1e-10);
        assert!(state.probability(1).abs() < 1e-10);

        // For |+⟩, P(|0⟩) = P(|1⟩) = 0.5
        let state = State::plus_state(1);
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);

        // For Bell state, P(|00⟩) = P(|11⟩) = 0.5, P(|01⟩) = P(|10⟩) = 0
        let state = State::bell_state();
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1).abs() < 1e-10);
        assert!(state.probability(2).abs() < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_product() {
        // |0⟩ ⊗ |0⟩ = |00⟩
        let state0 = State::zero_state(1);
        let state00 = state0.tensor_product(&state0);

        assert_eq!(state00.num_qubits, 2);
        assert!((state00.vector[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        for i in 1..4 {
            assert!((state00.vector[i] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        }

        // |0⟩ ⊗ |1⟩ = |01⟩
        let state0 = State::zero_state(1);
        let state1 = State::one_state(1);
        let state01 = state0.tensor_product(&state1);

        assert_eq!(state01.num_qubits, 2);
        assert!((state01.vector[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        for i in [0, 2, 3] {
            assert!((state01.vector[i] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        }
    }

    #[test]
    fn test_gate_application() {
        // Create X-gate (Pauli-X, NOT gate)
        let x_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let x_gate = MatrixGate { matrix: x_matrix };

        // Apply X to |0⟩, should get |1⟩
        let mut state = State::zero_state(1);
        state.apply_gate(&x_gate);

        assert!((state.vector[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state.vector[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);

        // Apply X to |1⟩, should get |0⟩
        let mut state = State::one_state(1);
        state.apply_gate(&x_gate);

        assert!((state.vector[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((state.vector[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_print() {
        let state = State::plus_state(1);
        let output = state.print();
        println!("{}", output);
        assert!(output.contains("State: 1 qubit"));
        // assert!(output.contains("State: 1 qubit"));
        assert!(output.contains("|0⟩ : 0.7071+0.0000i (p=0.5000)"));
        assert!(output.contains("|1⟩ : 0.7071+0.0000i (p=0.5000)"));

        let state = State::bell_state();
        let output = state.print();
        // Explicit result for the bell state printout
        println!("{}", output);
        assert!(output.contains("State: 2 qubits"));
        assert!(output.contains("|00⟩ : 0.7071+0.0000i (p=0.5000)"));
        assert!(output.contains("|11⟩ : 0.7071+0.0000i (p=0.5000)"));
    }
}
