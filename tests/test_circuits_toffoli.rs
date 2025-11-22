use logosq::circuits::Circuit;
use logosq::states::State;
use logosq::vis::circuit::text_diagram;

mod tests {
    use super::*;

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
    fn test_toffoli_quantum_addition() {
        // Test Toffoli in a quantum half-adder circuit
        // Proper half-adder needs to compute carry BEFORE modifying inputs

        let mut half_adder = Circuit::new(4); // Use 4 qubits to preserve inputs

        // Prepare input |11⟩ on first two qubits (a=1, b=1)
        half_adder.x(0).x(1);

        // Compute carry first: Toffoli(0,1,3) -> qubit 3 gets carry
        half_adder.toffoli(0, 1, 3);

        // Compute sum: CNOT(0,1) -> qubit 1 gets sum
        half_adder.cnot(0, 1);

        let mut state = State::zero_state(4);
        half_adder.execute(&mut state).unwrap();

        // For inputs a=1, b=1:
        // Step by step:
        // |0000⟩ -> X(0) -> |1000⟩ -> X(1) -> |1100⟩
        // |1100⟩ -> Toffoli(0,1,3) -> |1101⟩ (both controls are 1, flip target)
        // |1101⟩ -> CNOT(0,1) -> |1001⟩ (control 0 is 1, flip target 1)
        //
        // Result: |1001⟩
        // qubit 0 (a) = 1
        // qubit 1 (sum) = 0 (was 1, flipped by CNOT)
        // qubit 2 (unused) = 0
        // qubit 3 (carry) = 1
        let expected_state = 0b1001; // Binary 1001 = decimal 9
        assert!(
            state.probability(expected_state) > 0.99,
            "Expected state 9 (1001), got probabilities: {}",
            (0..16)
                .map(|i| format!("{}: {:.4}", i, state.probability(i)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn test_toffoli_as_universal_gate() {
        // Test Toffoli as a universal gate by constructing NOT and CNOT

        // NOT gate using Toffoli with controls set to |1⟩
        let mut not_circuit = Circuit::new(3);
        not_circuit.x(0).x(1); // Set first two qubits to |1⟩
        not_circuit.toffoli(0, 1, 2); // Apply Toffoli (acts as NOT on qubit 2)

        let mut not_state = State::zero_state(3);
        not_circuit.execute(&mut not_state).unwrap();

        // Expected: |111⟩ (target qubit flipped)
        assert!(not_state.probability(7) > 0.99);

        // CNOT gate using Toffoli with one control always set to |1⟩
        let mut cnot_circuit = Circuit::new(3);
        cnot_circuit.x(0); // Set first qubit to |1⟩
        cnot_circuit.x(1); // Set second qubit (which acts as control) to |1⟩
        cnot_circuit.toffoli(0, 1, 2); // Apply Toffoli (acts as CNOT from qubit 1 to 2)

        let mut cnot_state = State::zero_state(3);
        cnot_circuit.execute(&mut cnot_state).unwrap();

        // Expected: |111⟩ (target qubit flipped because both controls were |1⟩)
        assert!(
            cnot_state.probability(7) > 0.99,
            "Expected state 7 (111), got probabilities: {}",
            (0..8)
                .map(|i| format!("{}: {:.4}", i, cnot_state.probability(i)))
                .collect::<Vec<_>>()
                .join(", ")
        );
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
    fn test_toffoli_with_entanglement() {
        // Test Toffoli with one control being part of a Bell state
        let mut circuit = Circuit::new(4);

        // Create Bell state on qubits 0,1: (|00⟩ + |11⟩)/√2
        circuit.h(0).cnot(0, 1);

        // Set second control to |1⟩
        circuit.x(2);

        // Apply Toffoli with controls=1,2 and target=3
        circuit.toffoli(1, 2, 3);

        let mut state = State::zero_state(4);
        circuit.execute(&mut state).unwrap();

        // After Bell state: (|0000⟩ + |1100⟩)/√2
        // After X(2): (|0010⟩ + |1110⟩)/√2
        // After Toffoli(1,2,3):
        //   - |0010⟩: controls are 0,1 -> no flip -> |0010⟩
        //   - |1110⟩: controls are 1,1 -> flip -> |1111⟩
        // Result: (|0010⟩ + |1111⟩)/√2

        assert!(
            (state.probability(2) - 0.5).abs() < 1e-10, // |0010⟩ = decimal 2
            "State 2 probability: {}",
            state.probability(2)
        );
        assert!(
            (state.probability(15) - 0.5).abs() < 1e-10, // |1111⟩ = decimal 15
            "State 15 probability: {}",
            state.probability(15)
        );

        // All other states should have zero probability
        for i in 0..16 {
            if i != 2 && i != 15 {
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

    #[test]
    fn test_toffoli_incremental_control() {
        // Test behavior when control qubits change incrementally
        let mut circuit = Circuit::new(3);

        // Start with |000⟩ -> Toffoli -> |000⟩ (no flip, controls are 0,0)
        circuit.toffoli(0, 1, 2);

        // X(0): |000⟩ -> |100⟩
        // Toffoli: |100⟩ -> |100⟩ (no flip, controls are 1,0)
        circuit.x(0);
        circuit.toffoli(0, 1, 2);

        // X(1): |100⟩ -> |110⟩
        // Toffoli: |110⟩ -> |111⟩ (flip, controls are 1,1)
        circuit.x(1);
        circuit.toffoli(0, 1, 2);

        // X(0): |111⟩ -> |011⟩
        // Toffoli: |011⟩ -> |011⟩ (no flip, controls are 0,1)
        circuit.x(0);
        circuit.toffoli(0, 1, 2);

        let mut state = State::zero_state(3);
        circuit.execute(&mut state).unwrap();

        // Expected result: |011⟩ (first qubit is 0, second and third are 1)
        assert!(
            state.probability(3) > 0.99, // |011⟩ = decimal 3
            "Expected state 3 (011), got probabilities: {}",
            (0..8)
                .map(|i| format!("{}: {:.4}", i, state.probability(i)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn test_toffoli_circuit_diagram() {
        // Verify the circuit diagram properly shows the Toffoli gate
        let mut circuit = Circuit::new(3);
        circuit.x(0).x(1);
        circuit.toffoli(0, 1, 2);

        let diagram = text_diagram(&circuit);
        println!("Toffoli circuit diagram:\n{}", diagram);

        // Just print the diagram for visual inspection, no assertions
    }
}
