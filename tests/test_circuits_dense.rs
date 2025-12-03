//! Comprehensive tests for Circuit execution on dense state backend
//!
//! This file consolidates tests from multiple test files, removing redundancies
//! and organizing tests by functionality.

use logosq::circuits::Circuit;
use logosq::gates::matrix::*;
use logosq::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

mod tests {
    use super::*;

    // ========================================================================
    // BASIC CIRCUIT OPERATIONS
    // ========================================================================

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::new(2);
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.operations().len(), 0);
        assert_eq!(circuit.name(), None);

        let named_circuit = Circuit::new(3).with_name("Test Circuit");
        assert_eq!(named_circuit.num_qubits(), 3);
        assert_eq!(named_circuit.name(), Some("Test Circuit"));
    }

    #[test]
    fn test_adding_operations() {
        let mut circuit = Circuit::new(2);
        let x = x_gate();

        // Add X gate to qubit 0
        circuit.add_operation(x.clone(), vec![0], "X on 0").unwrap();
        assert_eq!(circuit.operations().len(), 1);
        assert_eq!(circuit.operations()[0].qubits(), &[0]);
        assert_eq!(circuit.operations()[0].name(), "X on 0");

        // Add X gate to qubit 1
        circuit.add_operation(x.clone(), vec![1], "X on 1").unwrap();
        assert_eq!(circuit.operations().len(), 2);
        assert_eq!(circuit.operations()[1].qubits(), &[1]);
        assert_eq!(circuit.operations()[1].name(), "X on 1");
    }

    #[test]
    #[should_panic(expected = "Invalid qubit index: 2 (must be < 2)")]
    fn test_invalid_qubit_index() {
        let mut circuit = Circuit::new(2);
        let x = x_gate();

        // This should panic because qubit 2 is out of range for a 2-qubit circuit
        circuit
            .add_operation(x, vec![2], "X on 2")
            .expect("Invalid qubit index: 2 (must be < 2)");
    }

    #[test]
    fn test_execute_circuit() {
        let mut circuit = Circuit::new(1);
        let x = x_gate();

        // Add X gate to qubit 0
        circuit.add_operation(x, vec![0], "X on 0").unwrap();

        // Execute on |0⟩ state
        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        // Should be |1⟩ state
        assert!(state.probability(0) < 1e-10);
        assert!((state.probability(1) - 1.0).abs() < 1e-10);
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
        circuit1.compose(&circuit2).unwrap();
        assert_eq!(circuit1.operations().len(), 2);

        // Execute the combined circuit
        let mut state = State::zero_state(2);
        circuit1.execute(&mut state).unwrap();

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
        assert_eq!(reversed.operations().len(), 2);
        assert_eq!(reversed.operations()[0].name(), "reversed_H");
        assert_eq!(reversed.operations()[1].name(), "reversed_X");

        // Execute original circuit on |0⟩
        let mut state1 = State::zero_state(1);
        circuit.execute(&mut state1).unwrap();

        // Execute reversed circuit on |0⟩
        let mut state2 = State::zero_state(1);
        reversed.execute(&mut state2).unwrap();

        // Results should be different
        assert!((state1.probability(0) - 0.5).abs() < 1e-10);
        assert!((state1.probability(1) - 0.5).abs() < 1e-10);

        assert!((state2.probability(0) - 0.5).abs() < 1e-10);
        assert!((state2.probability(1) - 0.5).abs() < 1e-10);
        // The phases would be different, but probabilities are the same
    }

    #[test]
    fn test_circuit_clone() {
        // Create a circuit with some operations
        let mut original = Circuit::new(2);
        original.h(0).cnot(0, 1);

        // Clone it
        let cloned = original.clone();

        // Check they have the same properties
        assert_eq!(original.num_qubits(), cloned.num_qubits());
        assert_eq!(original.operations().len(), cloned.operations().len());
        assert_eq!(original.name(), cloned.name());

        // Modify original and check that cloned is unaffected
        original.x(0);
        assert_eq!(original.operations().len(), 3);
        assert_eq!(cloned.operations().len(), 2);
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

    // ========================================================================
    // SINGLE-QUBIT GATES
    // ========================================================================

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

        assert_eq!(circuit.operations().len(), 9);
    }

    // ========================================================================
    // TWO-QUBIT GATES
    // ========================================================================

    #[test]
    fn test_two_qubit_gates() {
        let mut circuit = Circuit::new(2);

        // Test each two-qubit gate method
        circuit.cnot(0, 1);
        circuit.swap(0, 1);
        circuit.cz(0, 1);

        assert_eq!(circuit.operations().len(), 3);
    }

    // ========================================================================
    // CNOT GATE TESTS
    // ========================================================================

    #[test]
    fn test_cnot_basic_functionality() {
        // Create a 3-qubit circuit
        let mut circuit = Circuit::new(3);

        // Prepare initial state |100⟩ (first qubit is 1)
        circuit.x(0);

        // Add CNOT with control=0, target=2 (should flip qubit 2)
        circuit.cnot(0, 2);

        // Execute on initial zero state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected result: |101⟩ (first and third qubits are 1)
        assert!(state.probability(0) < 1e-10); // |000⟩
        assert!(state.probability(1) < 1e-10); // |001⟩
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!(state.probability(5) > 0.99); // |101⟩ (binary 101 = decimal 5)
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!(state.probability(7) < 1e-10); // |111⟩

        // Test CNOT with control=0 but now with control qubit in state |0⟩
        let mut circuit2 = Circuit::new(3);
        circuit2.x(0); // Set qubit 0 to |1⟩
        circuit2.cnot(0, 2); // Flip qubit 2 if qubit 0 is |1⟩
        circuit2.x(0); // Flip qubit 0 back to |0⟩
        circuit2.cnot(0, 2); // Should NOT flip qubit 2 as control is |0⟩

        let mut state2 = State::zero_state(3);
        circuit2.execute(&mut state2).unwrap();

        // Expected result: |001⟩ (only third qubit is 1, flipped once)
        assert!(state2.probability(1) > 0.99); // |001⟩
    }

    #[test]
    fn test_cnot_control_target_order() {
        // Test that CNOT(a,b) != CNOT(b,a)

        // Case 1: CNOT(0,1)
        let mut circuit1 = Circuit::new(2);
        circuit1.x(0); // Set control to |1⟩ → state is |10⟩
        circuit1.cnot(0, 1); // control=0 is |1⟩, so flip target (qubit 1)

        let mut state1 = State::zero_state(2);
        circuit1.execute(&mut state1).unwrap();

        // Expected: |11⟩ (both qubits are 1)
        assert!(state1.probability(3) > 0.99); // |11⟩ = decimal 3

        // Case 2: CNOT(1,0)
        let mut circuit2 = Circuit::new(2);
        circuit2.x(0); // Set qubit 0 to |1⟩ → state is |10⟩
        circuit2.cnot(1, 0); // control=1 is |0⟩, so do NOT flip target (qubit 0)

        let mut state2 = State::zero_state(2);
        circuit2.execute(&mut state2).unwrap();

        // Expected: |10⟩ (target NOT flipped because control=0)
        assert!(state2.probability(2) > 0.99); // |10⟩ = decimal 2
    }

    #[test]
    fn test_cnot_distant_qubits() {
        // Create a 5-qubit circuit to test CNOT with distant qubits
        let mut circuit = Circuit::new(5);

        // Prepare control qubit 0
        circuit.x(0);

        // Apply CNOT between qubits 0 and 4 (the furthest apart)
        circuit.cnot(0, 4);

        let mut state = State::zero_state(5);
        circuit.execute(&mut state).unwrap();

        // Expected result: |10001⟩
        let expected_state = 0b10001; // Binary 10001 = decimal 17
        assert!(state.probability(expected_state) > 0.99);

        // Verify all other states have negligible probability
        for i in 0..32 {
            // 2^5 = 32 possible states
            if i != expected_state {
                assert!(state.probability(i) < 1e-10);
            }
        }
    }

    #[test]
    fn test_cnot_chain() {
        // Test a chain of CNOT gates
        let mut circuit = Circuit::new(4);

        // Put first qubit in superposition
        circuit.h(0);

        // Create a chain: qubit i controls qubit i+1
        circuit.cnot(0, 1).cnot(1, 2).cnot(2, 3);

        let mut state = State::zero_state(4);
        circuit.execute(&mut state).unwrap();

        // Expected: |0000⟩ + |1111⟩ / √2
        assert!((state.probability(0) - 0.5).abs() < 1e-10); // |0000⟩
        assert!((state.probability(15) - 0.5).abs() < 1e-10); // |1111⟩

        // All other states should have zero probability
        for i in 1..15 {
            assert!(
                state.probability(i) < 1e-10,
                "Unexpected probability for state {}: {}",
                i,
                state.probability(i)
            );
        }
    }

    // ========================================================================
    // SWAP GATE TESTS
    // ========================================================================

    #[test]
    fn test_swap_gate_functionality() {
        // Create a 3-qubit circuit
        let mut circuit = Circuit::new(3);
        // Prepare initial state |100⟩ (first qubit is 1)
        circuit.x(0);

        // Add a SWAP between qubits 0 and 2
        circuit.swap(0, 2);

        // Execute on initial zero state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected result: |001⟩ (third qubit is now 1)
        assert!(state.probability(0) < 1e-10); // |000⟩
        assert!(state.probability(1) > 0.99); // |001⟩ (binary 001 = decimal 1)
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!(state.probability(5) < 1e-10); // |101⟩
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!(state.probability(7) < 1e-10); // |111⟩

        // Test that swap is its own inverse
        let mut inverse_circuit = Circuit::new(3);
        inverse_circuit.x(0);
        inverse_circuit.swap(0, 2);
        inverse_circuit.swap(0, 2); // Apply swap again

        let mut state2 = State::zero_state(3);
        inverse_circuit.execute(&mut state2).unwrap();

        // Expected result: |100⟩ (back to original state)
        assert!(state2.probability(4) > 0.99); // |100⟩ (binary 100 = decimal 4)
    }

    #[test]
    fn test_multiple_swap_operations() {
        // Create a 4-qubit circuit for more complex swap operations
        let mut circuit = Circuit::new(4);

        // Prepare initial state |1010⟩ (qubits 0 and 2 are 1)
        circuit.x(0).x(2);

        // Series of SWAP operations
        circuit
            .swap(0, 1) // |1010⟩ -> |0110⟩
            .swap(2, 3) // |0110⟩ -> |0101⟩
            .swap(1, 2); // |0101⟩ -> |0011⟩

        // Execute circuit
        let mut state = State::zero_state(4);
        circuit.execute(&mut state).unwrap();

        // Expected result: |0011⟩ (binary 0011 = decimal 3)
        assert!(state.probability(3) > 0.99);

        // Verify all other states have ~0 probability
        for i in 0..16 {
            if i != 3 {
                assert!(
                    state.probability(i) < 1e-10,
                    "Unexpected probability for state {}: {}",
                    i,
                    state.probability(i)
                );
            }
        }
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
        circuit.execute(&mut state).unwrap();

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

    // ========================================================================
    // TOFFOLI GATE TESTS
    // ========================================================================

    #[test]
    fn test_toffoli_basic_functionality() {
        // Create a 3-qubit circuit
        let mut circuit = Circuit::new(3);

        // Prepare initial state |110⟩ (both control qubits are 1)
        circuit.x(0).x(1);

        // Add Toffoli with controls=0,1 and target=2
        circuit.toffoli(0, 1, 2);

        // Execute on initial zero state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected result: |111⟩ (all qubits are 1)
        assert!(state.probability(0) < 1e-10); // |000⟩
        assert!(state.probability(1) < 1e-10); // |001⟩
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!(state.probability(5) < 1e-10); // |101⟩
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!(state.probability(7) > 0.99); // |111⟩ (binary 111 = decimal 7)

        // Test with only one control qubit set to |1⟩ (should NOT flip target)
        let mut circuit2 = Circuit::new(3);
        circuit2.x(0); // Only first control is 1
        circuit2.toffoli(0, 1, 2);

        let mut state2 = State::zero_state(3);
        circuit2.execute(&mut state2).unwrap();

        // Expected result: |100⟩ (only first qubit is 1)
        assert!(state2.probability(4) > 0.99); // |100⟩ (binary 100 = decimal 4)
    }

    #[test]
    fn test_toffoli_control_order() {
        // Test that order of control qubits doesn't matter

        // Case 1: Toffoli(0,1,2)
        let mut circuit1 = Circuit::new(3);
        circuit1.x(0).x(1); // Set both controls to |1⟩
        circuit1.toffoli(0, 1, 2);

        let mut state1 = State::zero_state(3);
        circuit1.execute(&mut state1).unwrap();

        // Expected: |111⟩ (target flipped)
        assert!(state1.probability(7) > 0.99);

        // Case 2: Toffoli(1,0,2) - reversed control order
        let mut circuit2 = Circuit::new(3);
        circuit2.x(0).x(1); // Set both controls to |1⟩
        circuit2.toffoli(1, 0, 2);

        let mut state2 = State::zero_state(3);
        circuit2.execute(&mut state2).unwrap();

        // Expected: |111⟩ (target should still be flipped)
        assert!(state2.probability(7) > 0.99);
    }

    #[test]
    fn test_toffoli_distant_qubits() {
        // Create a 5-qubit circuit to test Toffoli with distant qubits
        let mut circuit = Circuit::new(5);

        // Prepare control qubits 0 and 4 (the furthest apart)
        circuit.x(0).x(4);

        // Apply Toffoli between qubits 0,4 (controls) and 2 (target in the middle)
        circuit.toffoli(0, 4, 2);

        let mut state = State::zero_state(5);
        circuit.execute(&mut state).unwrap();

        // Expected result: |10101⟩ (qubits 0, 2, and 4 are 1)
        // Binary: qubit 0=1, qubit 1=0, qubit 2=1, qubit 3=0, qubit 4=1
        // As decimal: 16 + 4 + 1 = 21
        let expected_state = 21; // Binary 10101 = decimal 21
        assert!(
            state.probability(expected_state) > 0.99,
            "Expected state 21 (10101), got probabilities: {}",
            (0..32)
                .map(|i| format!("{}: {:.4}", i, state.probability(i)))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Verify all other states have negligible probability
        for i in 0..32 {
            // 2^5 = 32 possible states
            if i != expected_state {
                assert!(
                    state.probability(i) < 1e-10,
                    "Unexpected probability for state {}: {}",
                    i,
                    state.probability(i)
                );
            }
        }
    }

    #[test]
    fn test_toffoli_with_superposition() {
        // Test Toffoli with a control qubit in superposition
        let mut circuit = Circuit::new(3);

        // Prepare |+10⟩ state (first qubit in superposition, second qubit is 1)
        circuit.h(0).x(1);

        // Apply Toffoli
        circuit.toffoli(0, 1, 2);

        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected: 1/√2 (|010⟩ + |111⟩)
        // The target flips only when first control is |1⟩
        assert!((state.probability(2) - 0.5).abs() < 1e-10); // |010⟩
        assert!((state.probability(7) - 0.5).abs() < 1e-10); // |111⟩

        // All other states should have zero probability
        for i in 0..8 {
            if i != 2 && i != 7 {
                assert!(
                    state.probability(i) < 1e-10,
                    "Unexpected probability for state {}: {}",
                    i,
                    state.probability(i)
                );
            }
        }
    }

    #[test]
    fn test_toffoli_chain() {
        // Test a chain of Toffoli gates to create a multi-controlled operation
        let mut circuit = Circuit::new(5);

        // Set qubits 0, 1, 2 to |1⟩
        circuit.x(0).x(1).x(2);

        // Chain of Toffoli gates to implement a 3-controlled NOT on qubit 4
        // First Toffoli: controls=0,1, target=3 (temporary ancilla)
        circuit.toffoli(0, 1, 3);
        // Second Toffoli: controls=2,3, target=4
        circuit.toffoli(2, 3, 4);

        let mut state = State::zero_state(5);
        circuit.execute(&mut state).unwrap();

        // After first Toffoli: |11110⟩ (qubit 3 flipped because 0,1 are both 1)
        // After second Toffoli: |11111⟩ (qubit 4 flipped because 2,3 are both 1)
        let expected_state = 0b11111; // Binary 11111 = decimal 31
        assert!(
            state.probability(expected_state) > 0.99,
            "Expected state 31 (11111), got probabilities: {}",
            (0..32)
                .map(|i| format!("{}: {:.4}", i, state.probability(i)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn test_toffoli_self_inverse() {
        // Test that applying Toffoli twice restores the original state
        let mut circuit = Circuit::new(3);

        // Prepare |110⟩
        circuit.x(0).x(1);

        // Apply Toffoli twice
        circuit.toffoli(0, 1, 2).toffoli(0, 1, 2);

        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected result: |110⟩ (back to the original state)
        assert!(state.probability(6) > 0.99); // |110⟩ = decimal 6
    }

    // ========================================================================
    // ENTANGLED STATES (Bell, GHZ)
    // ========================================================================

    #[test]
    fn test_bell_state_circuit() {
        // Circuit to create Bell state |00⟩ + |11⟩ / sqrt(2)
        let mut circuit = Circuit::new(2);
        circuit.h(0).cnot(0, 1);

        // Execute on |00⟩ state
        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

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
        // Create |10⟩ state first
        circuit.x(0);
        // Then create Bell state
        circuit.h(0).cnot(0, 1);

        // Execute on |00⟩ state
        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should be Bell state
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
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

        circuit
            .add_matrix_gate(cnot01_full, vec![0, 1, 2], "CNOT_01")
            .unwrap();

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

        circuit
            .add_matrix_gate(cnot12_full, vec![0, 1, 2], "CNOT_12")
            .unwrap();

        // Execute on |000⟩ state
        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

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
    fn test_ghz_state_simple() {
        // Create GHZ state using simple circuit operations
        let mut circuit = Circuit::new(3);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit.cnot(0, 2); // Control=0, target=2 (non-adjacent)

        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected: |000⟩ + |111⟩ / √2
        assert!((state.probability(0) - 0.5).abs() < 1e-10); // |000⟩
        assert!(state.probability(1) < 1e-10); // |001⟩
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!(state.probability(5) < 1e-10); // |101⟩
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!((state.probability(7) - 0.5).abs() < 1e-10); // |111⟩
    }

    #[test]
    fn test_cnot_with_preexisting_entanglement() {
        // Create a Bell state on qubits 0 and 1
        let mut circuit = Circuit::new(3);
        circuit.h(0).cnot(0, 1);

        // Add CNOT from entangled qubit 1 to qubit 2
        circuit.cnot(1, 2);

        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected: |000⟩ + |111⟩ / √2
        // (Qubit 2 is flipped only when qubit 1 is |1⟩, which happens when qubit 0 is |1⟩)
        assert!((state.probability(0) - 0.5).abs() < 1e-10); // |000⟩
        assert!(state.probability(1) < 1e-10); // |001⟩
        assert!(state.probability(2) < 1e-10); // |010⟩
        assert!(state.probability(3) < 1e-10); // |011⟩
        assert!(state.probability(4) < 1e-10); // |100⟩
        assert!(state.probability(5) < 1e-10); // |101⟩
        assert!(state.probability(6) < 1e-10); // |110⟩
        assert!((state.probability(7) - 0.5).abs() < 1e-10); // |111⟩
    }

    // ========================================================================
    // EXPECTATION VALUES AND MEASUREMENT
    // ========================================================================

    #[test]
    fn test_expectation_value() {
        // Create a circuit that prepares |+⟩ state
        let mut circuit = Circuit::new(1);
        circuit.h(0);

        // Measure expectation of X operator
        let x = x_gate();
        let expectation = circuit.expectation(&x).unwrap();

        // <+|X|+> = 1
        assert!((expectation - 1.0).abs() < 1e-10);

        // Measure expectation of Z operator
        let z = z_gate();
        let expectation = circuit.expectation(&z).unwrap();

        // <+|Z|+> = 0
        assert!(expectation.abs() < 1e-10);
    }

    #[test]
    fn test_execute_and_measure() {
        // Create a deterministic circuit that always gives |1⟩
        let mut circuit = Circuit::new(1);
        circuit.x(0);

        // Execute and measure
        let results = circuit.execute_and_measure().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1);

        // Create a circuit that creates an equal superposition
        let mut circuit = Circuit::new(1);
        circuit.h(0);

        // Execute and measure multiple times to check probabilistic behavior
        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            let results = circuit.execute_and_measure().unwrap();
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
}
