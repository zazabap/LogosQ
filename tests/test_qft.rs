use approx::assert_relative_eq;
use logosq::algorithms::qft;
use logosq::circuits::Circuit;
use logosq::states::State;
use std::f64::consts::PI;

mod tests {
    use super::*;

    #[test]
    fn test_qft_circuit_creation() {
        let num_qubits = 3;
        let circuit = qft::create_circuit(num_qubits);

        // Check circuit properties
        assert_eq!(circuit.num_qubits, num_qubits);
        assert!(circuit.name.is_some());
        assert!(circuit.name.unwrap().contains("Quantum Fourier Transform"));

        // Verify circuit has operations (specific count depends on implementation)
        assert!(!circuit.operations.is_empty());
    }

    #[test]
    fn test_inverse_qft_circuit_creation() {
        let num_qubits = 3;
        let circuit = qft::create_inverse_circuit(num_qubits);

        // Check circuit properties
        assert_eq!(circuit.num_qubits, num_qubits);
        assert!(circuit.name.is_some());
        assert!(circuit.name.unwrap().contains("Inverse"));

        // Verify circuit has operations
        assert!(!circuit.operations.is_empty());
    }

    #[test]
    fn test_qft_on_zero_state() {
        // QFT on |0⟩ should create an equal superposition of all states
        let mut state = State::zero_state(2);
        qft::apply(&mut state);

        // Expected result: equal superposition with specific phases
        let n = state.vector.len();
        for i in 0..n {
            assert_relative_eq!(
                state.vector[i].norm(),
                1.0 / (n as f64).sqrt(),
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn test_qft_on_basis_state() {
        // Create a basis state |1⟩
        let mut state = State::zero_state(1);
        let mut circuit = Circuit::new(1);
        circuit.x(0);
        circuit.execute(&mut state);

        // Apply QFT
        qft::apply(&mut state);

        // For 1 qubit, QFT of |1⟩ = |0⟩ - |1⟩)/√2
        assert_relative_eq!(state.vector[0].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-5);
        assert_relative_eq!(state.vector[0].im, 0.0, epsilon = 1e-5);

        assert_relative_eq!(state.vector[1].re, -1.0 / 2.0_f64.sqrt(), epsilon = 1e-5);
        assert_relative_eq!(state.vector[1].im, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_qft_inverse_qft_identity() {
        // Test that QFT followed by inverse QFT returns the original state

        // Test on a few different states
        for n_qubits in 1..=4 {
            println!("Testing {}-qubit state", n_qubits);
            // Start with a non-trivial state (apply some gates to |0⟩)
            let mut state = State::zero_state(n_qubits);
            println!("Initial state ({} qubits): ", state.num_qubits);
            let mut prep_circuit = Circuit::new(n_qubits);

            // Apply some gates to create a test state
            for i in 0..n_qubits {
                if i % 2 == 0 {
                    prep_circuit.h(i);
                } else {
                    prep_circuit.x(i);
                }
            }
            prep_circuit.execute(&mut state);

            // Save original state
            let original_state = state.clone();

            println!("Original state ({} qubits): ", original_state.num_qubits);

            // Apply QFT followed by inverse QFT
            qft::apply(&mut state);
            println!("Finished QFT and inverse QFT on {} qubits", n_qubits);

            qft::apply_inverse(&mut state);

            // Verify state is the same as original
            for i in 0..state.vector.len() {
                assert_relative_eq!(
                    state.vector[i].re,
                    original_state.vector[i].re,
                    epsilon = 1e-5
                );
                assert_relative_eq!(
                    state.vector[i].im,
                    original_state.vector[i].im,
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_qft_known_transformation() {
        // Test QFT on a known state with known output

        // Create state |0⟩ + |1⟩ (equivalent to applying H to |0⟩)
        let mut state = State::zero_state(1);
        let mut prep_circuit = Circuit::new(1);
        prep_circuit.h(0);
        prep_circuit.execute(&mut state);

        // Apply QFT
        qft::apply(&mut state);

        // Expected: |0⟩
        assert_relative_eq!(state.vector[0].norm(), 1.0, epsilon = 1e-5);
        assert_relative_eq!(state.vector[1].norm(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_qft_multi_qubit_transformation() {
        // Test QFT on a 2-qubit state
        let mut state = State::zero_state(2);

        // Create state |00⟩ + |01⟩ + |10⟩ + |11⟩ with equal amplitudes
        let mut prep_circuit = Circuit::new(2);
        prep_circuit.h(0);
        prep_circuit.h(1);
        prep_circuit.execute(&mut state);

        // Apply QFT
        qft::apply(&mut state);

        // Expected result: |00⟩
        assert_relative_eq!(state.vector[0].norm(), 1.0, epsilon = 1e-5);
        assert_relative_eq!(state.vector[1].norm(), 0.0, epsilon = 1e-5);
        assert_relative_eq!(state.vector[2].norm(), 0.0, epsilon = 1e-5);
        assert_relative_eq!(state.vector[3].norm(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_controlled_phase_gate() {
        // Test the controlled phase gate with various angles

        for angle in [PI / 4.0, PI / 2.0, PI] {
            let mut circuit = Circuit::new(2);
            qft::controlled_phase(&mut circuit, 0, 1, angle);

            // Verify the effect on a superposition state
            let mut state = State::zero_state(2);
            circuit.h(0);
            circuit.h(1);
            circuit.execute(&mut state);

            // The transformation should maintain equal probability for all states
            // but introduce relative phases
            for i in 0..4 {
                assert_relative_eq!(state.vector[i].norm(), 0.5, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_qft_correctness() {
        // Test QFT on a specific 3-qubit input and check expected output

        // Create state |001⟩
        let mut state = State::zero_state(3);
        let mut prep_circuit = Circuit::new(3);
        prep_circuit.x(2); // Note: qubit 2 is the least significant in the state vector
        prep_circuit.execute(&mut state);

        // Apply QFT
        qft::apply(&mut state);

        // Expected result: (1/√8) * (|0⟩ + e^(2πi*1/8)|1⟩ + e^(2πi*2/8)|2⟩ + ... + e^(2πi*7/8)|7⟩)
        let n = state.vector.len();
        let norm_factor = 1.0 / (n as f64).sqrt();

        // Check amplitudes - all should be 1/√8 in magnitude
        for i in 0..n {
            assert_relative_eq!(state.vector[i].norm(), norm_factor, epsilon = 1e-5);
        }

        // Check phases - the phase of the i-th amplitude should be 2πi*k/8
        // where k is the input state (1 in this case)
        let k = 1;
        for i in 0..n {
            let expected_phase = 2.0 * PI * (i as f64) * (k as f64) / (n as f64);
            let actual_phase = state.vector[i].arg();

            // Normalize phase comparison to account for 2π periodicity
            let phase_diff = (expected_phase - actual_phase).abs() % (2.0 * PI);
            let normalized_diff = phase_diff.min(2.0 * PI - phase_diff);

            assert_relative_eq!(normalized_diff, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_qft_on_zero_states_multiple_sizes() {
        // Test QFT on |0⟩ for increasing qubit counts
        // For large qubit counts (>8), we'll only verify a subset of amplitudes to save memory and time

        println!("Testing QFT on |0⟩ states with varying qubit counts:");

        for num_qubits in 1..=10 {
            println!("  Testing {}-qubit zero state", num_qubits);

            // Create zero state with specified number of qubits
            let mut state = State::zero_state(num_qubits);

            // Apply QFT
            qft::apply(&mut state);

            // Expected result: equal superposition with specific phases
            let n = state.vector.len();
            let expected_magnitude = 1.0 / (n as f64).sqrt();

            // For smaller states, verify all amplitudes
            if num_qubits <= 8 {
                for i in 0..n {
                    assert_relative_eq!(
                        state.vector[i].norm(),
                        expected_magnitude,
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            } else {
                // For larger states, sample a few amplitudes to verify
                let sample_indices = [0, 1, n / 4, n / 2, n - 2, n - 1];
                for &i in &sample_indices {
                    assert_relative_eq!(
                        state.vector[i].norm(),
                        expected_magnitude,
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }

            // Additional test: QFT on |0⟩ should have all real, positive amplitudes
            if num_qubits <= 6 {
                // Limit detailed phase checking to smaller states
                for i in 0..n {
                    // All amplitudes should be real and positive
                    assert_relative_eq!(state.vector[i].im, 0.0, epsilon = 1e-5);
                    assert!(state.vector[i].re > 0.0);
                }
            }

            println!("  ✓ {}-qubit zero state QFT verified", num_qubits);
        }
    }
}
