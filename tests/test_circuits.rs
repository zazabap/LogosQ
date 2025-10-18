use logosq::circuits::Circuit;
use logosq::gates::matrix::*;
use logosq::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

mod tests {
    use super::*;
    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::new(2);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.operations.len(), 0);
        assert_eq!(circuit.name, None);

        let named_circuit = Circuit::new(3).with_name("Test Circuit");
        assert_eq!(named_circuit.num_qubits, 3);
        assert_eq!(named_circuit.name, Some("Test Circuit".to_string()));
    }

    #[test]
    fn test_adding_operations() {
        let mut circuit = Circuit::new(2);
        let x = x_gate();

        // Add X gate to qubit 0
        circuit.add_operation(x.clone(), vec![0], "X on 0");
        assert_eq!(circuit.operations.len(), 1);
        assert_eq!(circuit.operations[0].qubits, vec![0]);
        assert_eq!(circuit.operations[0].name, "X on 0");

        // Add X gate to qubit 1
        circuit.add_operation(x.clone(), vec![1], "X on 1");
        assert_eq!(circuit.operations.len(), 2);
        assert_eq!(circuit.operations[1].qubits, vec![1]);
        assert_eq!(circuit.operations[1].name, "X on 1");
    }

    #[test]
    #[should_panic(expected = "Qubit index 2 out of range")]
    fn test_invalid_qubit_index() {
        let mut circuit = Circuit::new(2);
        let x = x_gate();

        // This should panic because qubit 2 is out of range for a 2-qubit circuit
        circuit.add_operation(x, vec![2], "X on 2");
    }

    #[test]
    fn test_execute_circuit() {
        let mut circuit = Circuit::new(1);
        let x = x_gate();

        // Add X gate to qubit 0
        circuit.add_operation(x, vec![0], "X on 0");

        // Execute on |0⟩ state
        let mut state = State::zero_state(1);
        circuit.execute(&mut state);

        // Should be |1⟩ state
        assert!(state.probability(0) < 1e-10);
        assert!((state.probability(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_circuit() {
        // Circuit to create Bell state |00⟩ + |11⟩ / sqrt(2)
        let mut circuit = Circuit::new(2);
        let h = h_gate();
        let cnot = cnot_gate();

        circuit.add_single_qubit_gate(h.matrix, 0, "H");
        circuit.add_two_qubit_gate(cnot.matrix, 0, 1, "CNOT");

        // Execute on |00⟩ state
        let mut state = State::zero_state(2);
        circuit.execute(&mut state);

        // Should be Bell state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_circuit_flipped() {
        // Circuit to create Bell state |00⟩ + |11⟩ / sqrt(2)
        let mut circuit = Circuit::new(2);
        let h = h_gate();
        let x = x_gate();
        let cnot = cnot_gate();

        // Create |10⟩ state first
        circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
        // Then create Bell state
        circuit.add_single_qubit_gate(h.matrix, 0, "H");

        circuit.add_two_qubit_gate(cnot.matrix, 0, 1, "CNOT");

        // Execute on |00⟩ state
        let mut state = State::zero_state(2);
        circuit.execute(&mut state);

        // Should be Bell state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_three_state_circuit() {
        // Circuit to create |000⟩ + |111⟩ / sqrt(2)
        let mut circuit = Circuit::new(3);
        let h = h_gate();
        let cnot = cnot_gate();

        // Create |000⟩ state
        circuit.add_single_qubit_gate(h.matrix, 0, "H");
        circuit.add_two_qubit_gate(cnot.matrix.clone(), 0, 1, "CNOT");
        circuit.add_two_qubit_gate(cnot.matrix.clone(), 0, 2, "CNOT");

        // Execute on |000⟩ state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state);

        // Should be |000⟩ + |111⟩ / sqrt(2)
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(7) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fluent_bell_state() {
        // Create Bell state using fluent interface
        let mut circuit = Circuit::new(2);
        circuit.h(0).cnot(0, 1);

        // Execute on |00⟩ state
        let mut state = State::zero_state(2);
        circuit.execute(&mut state);

        // Should be Bell state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_circuit_composition() {
        // First circuit: X on qubit 0
        let mut circuit1 = Circuit::new(2);
        circuit1.x(0);

        // Second circuit: H on qubit 1
        let mut circuit2 = Circuit::new(2);
        circuit2.h(1);

        // Compose them
        circuit1.compose(&circuit2);
        assert_eq!(circuit1.operations.len(), 2);

        // Execute the combined circuit
        let mut state = State::zero_state(2);
        circuit1.execute(&mut state);

        // Should have X on qubit 0 and H on qubit 1
        // |10⟩ + |11⟩ / sqrt(2)
        assert!(state.probability(0) < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!((state.probability(2) - 0.5).abs() < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_circuit_reversal() {
        // Create a circuit X -> H on a single qubit
        let mut circuit = Circuit::new(1);
        circuit.x(0).h(0);

        // Reverse it to get H -> X
        let reversed = circuit.reversed();
        assert_eq!(reversed.operations.len(), 2);
        assert_eq!(reversed.operations[0].name, "reversed_H");
        assert_eq!(reversed.operations[1].name, "reversed_X");

        // Execute original circuit on |0⟩
        let mut state1 = State::zero_state(1);
        circuit.execute(&mut state1);

        // Execute reversed circuit on |0⟩
        let mut state2 = State::zero_state(1);
        reversed.execute(&mut state2);

        // Results should be different
        assert!((state1.probability(0) - 0.5).abs() < 1e-10);
        assert!((state1.probability(1) - 0.5).abs() < 1e-10);

        assert!((state2.probability(0) - 0.5).abs() < 1e-10);
        assert!((state2.probability(1) - 0.5).abs() < 1e-10);
        // The phases would be different, but probabilities are the same
    }

    #[test]
    fn test_expectation_value() {
        // Create a circuit that prepares |+⟩ state
        let mut circuit = Circuit::new(1);
        circuit.h(0);

        // Measure expectation of X operator
        let x = x_gate();
        let expectation = circuit.expectation(&x);

        // <+|X|+> = 1
        assert!((expectation - 1.0).abs() < 1e-10);

        // Measure expectation of Z operator
        let z = z_gate();
        let expectation = circuit.expectation(&z);

        // <+|Z|+> = 0
        assert!(expectation.abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_gates() {
        let mut circuit = Circuit::new(1);

        // Test each single-qubit gate method
        circuit.x(0);
        circuit.y(0);
        circuit.z(0);
        circuit.h(0);
        circuit.s(0);
        circuit.t(0);
        circuit.rx(0, PI / 2.0);
        circuit.ry(0, PI / 2.0);
        circuit.rz(0, PI / 2.0);

        assert_eq!(circuit.operations.len(), 9);
    }

    #[test]
    fn test_two_qubit_gates() {
        let mut circuit = Circuit::new(2);

        // Test each two-qubit gate method
        circuit.cnot(0, 1);
        circuit.swap(0, 1);
        circuit.cz(0, 1);

        assert_eq!(circuit.operations.len(), 3);
    }

    #[test]
    fn test_toffoli_gate() {
        let mut circuit = Circuit::new(3);

        // Add Toffoli gate
        circuit.toffoli(0, 1, 2);

        assert_eq!(circuit.operations.len(), 1);

        // Test with |110⟩ state (should flip to |111⟩)
        let mut state = State::zero_state(3);

        // Apply X to first two qubits to get |110⟩
        let x = x_gate();
        let mut prep_circuit = Circuit::new(3);
        prep_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
        prep_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
        prep_circuit.execute(&mut state);

        // Apply Toffoli gate
        circuit.execute(&mut state);

        // Result should be |111⟩
        assert!(state.probability(7) > 0.99); // 111 in binary = 7 in decimal
    }

    #[test]
    fn test_ghz_state() {
        // Create a GHZ state |000⟩ + |111⟩ / sqrt(2)
        let mut circuit = Circuit::new(3);

        // Apply H to qubit 0
        circuit.h(0);

        // Manually create expanded CNOT(0,1) for a 3-qubit system
        let mut cnot01_full = Array2::zeros((8, 8));

        // Identity on states where control qubit 0 is |0⟩
        cnot01_full[[0, 0]] = Complex64::new(1.0, 0.0);
        cnot01_full[[1, 1]] = Complex64::new(1.0, 0.0);
        cnot01_full[[2, 2]] = Complex64::new(1.0, 0.0);
        cnot01_full[[3, 3]] = Complex64::new(1.0, 0.0);

        // CNOT action where control qubit 0 is |1⟩
        cnot01_full[[4, 6]] = Complex64::new(1.0, 0.0); // |100⟩ → |110⟩
        cnot01_full[[5, 7]] = Complex64::new(1.0, 0.0); // |101⟩ → |111⟩
        cnot01_full[[6, 4]] = Complex64::new(1.0, 0.0); // |110⟩ → |100⟩
        cnot01_full[[7, 5]] = Complex64::new(1.0, 0.0); // |111⟩ → |101⟩

        circuit.add_matrix_gate(cnot01_full, vec![0, 1, 2], "CNOT_01");

        // Manually create expanded CNOT(1,2) for a 3-qubit system
        let mut cnot12_full = Array2::zeros((8, 8));

        // Identity for states where control qubit 1 is |0⟩
        cnot12_full[[0, 0]] = Complex64::new(1.0, 0.0);
        cnot12_full[[1, 1]] = Complex64::new(1.0, 0.0);
        cnot12_full[[4, 4]] = Complex64::new(1.0, 0.0);
        cnot12_full[[5, 5]] = Complex64::new(1.0, 0.0);

        // CNOT action where control qubit 1 is |1⟩
        cnot12_full[[2, 3]] = Complex64::new(1.0, 0.0); // |010⟩ → |011⟩
        cnot12_full[[3, 2]] = Complex64::new(1.0, 0.0); // |011⟩ → |010⟩
        cnot12_full[[6, 7]] = Complex64::new(1.0, 0.0); // |110⟩ → |111⟩
        cnot12_full[[7, 6]] = Complex64::new(1.0, 0.0); // |111⟩ → |110⟩

        circuit.add_matrix_gate(cnot12_full, vec![0, 1, 2], "CNOT_12");

        // Execute on |000⟩ state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state);

        // Should be GHZ state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!(state.probability(3) < 1e-10);
        assert!(state.probability(4) < 1e-10);
        assert!(state.probability(5) < 1e-10);
        assert!(state.probability(6) < 1e-10);
        assert!((state.probability(7) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_execute_and_measure() {
        // Create a deterministic circuit that always gives |1⟩
        let mut circuit = Circuit::new(1);
        circuit.x(0);

        // Execute and measure
        let results = circuit.execute_and_measure();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1);

        // Create a circuit that creates an equal superposition
        let mut circuit = Circuit::new(1);
        circuit.h(0);

        // Execute and measure multiple times to check probabilistic behavior
        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            let results = circuit.execute_and_measure();
            if results[0] == 0 {
                zeros += 1;
            } else {
                ones += 1;
            }
        }

        // Should get roughly equal numbers (allowing some statistical variation)
        assert!(zeros > 30 && zeros < 70);
        assert!(ones > 30 && ones < 70);
    }

    #[test]
    fn test_circuit_clone() {
        // Create a circuit with some operations
        let mut original = Circuit::new(2);
        original.h(0).cnot(0, 1);

        // Clone it
        let cloned = original.clone();

        // Check they have the same properties
        assert_eq!(original.num_qubits, cloned.num_qubits);
        assert_eq!(original.operations.len(), cloned.operations.len());
        assert_eq!(original.name, cloned.name);

        // Modify original and check that cloned is unaffected
        original.x(0);
        assert_eq!(original.operations.len(), 3);
        assert_eq!(cloned.operations.len(), 2);
    }

    #[test]
    fn test_debug_format() {
        // Create a simple circuit
        let mut circuit = Circuit::new(1).with_name("Test Circuit");
        circuit.x(0).h(0);

        // Get the debug string
        let debug_str = format!("{:?}", circuit);

        // Check it contains the expected information
        assert!(debug_str.contains("Test Circuit"));
        assert!(debug_str.contains("(1 qubits)"));
        assert!(debug_str.contains("0: X"));
        assert!(debug_str.contains("1: H"));
    }

    #[test]
    fn test_swap_entangled_states() {
        // Test SWAP on entangled states
        let mut circuit = Circuit::new(3);

        // Create Bell state on qubits 0 and 1
        circuit.h(0).cnot(0, 1);
        // Now we have (|00⟩ + |11⟩)/√2 on first two qubits, and |0⟩ on third

        // Swap qubits 1 and 2
        circuit.swap(1, 2);

        // Execute circuit
        let mut state = State::zero_state(3);
        circuit.execute(&mut state);

        // Expected: Bell state now between qubits 0 and 2
        // State should be (|000⟩ + |101⟩)/√2
        assert!((state.probability(0) - 0.5).abs() < 1e-10); // |000⟩
        assert!(state.probability(1) < 1e-10); // |001⟩
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!((state.probability(5) - 0.5).abs() < 1e-10); // |101⟩
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!(state.probability(7) < 1e-10); // |111⟩
    }

    // Fully test the CNOT gate functionality including edge cases
    // ensuring it behaves as expected in various scenarios
}
