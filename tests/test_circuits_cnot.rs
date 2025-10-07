use logosq::circuits::Circuit;
use logosq::states::State;

mod tests {
    use super::*;
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
        circuit.execute(&mut state);

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
        circuit2.execute(&mut state2);

        // Expected result: |001⟩ (only third qubit is 1, flipped once)
        assert!(state2.probability(1) > 0.99); // |001⟩
    }

    #[test]
    fn test_cnot_control_target_order() {
        // Test that CNOT(a,b) != CNOT(b,a)

        // Case 1: CNOT(0,1)
        let mut circuit1 = Circuit::new(2);
        circuit1.x(0); // Set control to |1⟩
        circuit1.cnot(0, 1);

        let mut state1 = State::zero_state(2);
        circuit1.execute(&mut state1);

        // Expected: |11⟩
        assert!(state1.probability(3) > 0.99);

        println!(
            "State1 probabilities: |00⟩: {}, |01⟩: {}, |10⟩: {}, |11⟩: {}",
            state1.probability(0),
            state1.probability(1),
            state1.probability(2),
            state1.probability(3)
        );

        // Case 2: CNOT(1,0)
        let mut circuit2 = Circuit::new(2);
        circuit2.x(0); // Set what will be the target to |1⟩
        circuit2.cnot(1, 0);

        let mut state2 = State::zero_state(2);
        circuit2.execute(&mut state2);

        println!(
            "State2 probabilities: |00⟩: {}, |01⟩: {}, |10⟩: {}, |11⟩: {}",
            state2.probability(0),
            state2.probability(1),
            state2.probability(2),
            state2.probability(3)
        );

        // Expected: |00⟩ (target flipped from |1⟩ to |0⟩ because control=0)
        assert!(state2.probability(1) > 0.99); // This checks for |10⟩, not |00⟩
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
        circuit.execute(&mut state);

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
    fn test_cnot_entanglement_creation() {
        // Test CNOT's ability to create entanglement

        // Create a circuit to generate a Bell state
        let mut circuit = Circuit::new(2);
        circuit.h(0); // Put qubit 0 in superposition
        circuit.cnot(0, 1); // Entangle qubits 0 and 1

        let mut state = State::zero_state(2);
        circuit.execute(&mut state);

        // Expected: |00⟩ + |11⟩ / √2
        assert!((state.probability(0) - 0.5).abs() < 1e-10); // |00⟩
        assert!(state.probability(1) < 1e-10); // |01⟩
        assert!(state.probability(2) < 1e-10); // |10⟩
        assert!((state.probability(3) - 0.5).abs() < 1e-10); // |11⟩

        // Now test with 3 qubits to create a GHZ state
        let mut ghz_circuit = Circuit::new(3);
        ghz_circuit.h(0);
        ghz_circuit.cnot(0, 1);
        ghz_circuit.cnot(0, 2); // Control=0, target=2 (non-adjacent)

        let mut ghz_state = State::zero_state(3);
        ghz_circuit.execute(&mut ghz_state);

        // Expected: |000⟩ + |111⟩ / √2
        assert!((ghz_state.probability(0) - 0.5).abs() < 1e-10); // |000⟩
        assert!(ghz_state.probability(1) < 1e-10); // |001⟩
        assert!(ghz_state.probability(2) < 1e-10); // |010⟩
        assert!(ghz_state.probability(3) < 1e-10); // |011⟩
        assert!(ghz_state.probability(4) < 1e-10); // |100⟩
        assert!(ghz_state.probability(5) < 1e-10); // |101⟩
        assert!(ghz_state.probability(6) < 1e-10); // |110⟩
        assert!((ghz_state.probability(7) - 0.5).abs() < 1e-10); // |111⟩
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
        circuit.execute(&mut state);

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

    #[test]
    fn test_cnot_with_preexisting_entanglement() {
        // Create a Bell state on qubits 0 and 1
        let mut circuit = Circuit::new(3);
        circuit.h(0).cnot(0, 1);

        // Add CNOT from entangled qubit 1 to qubit 2
        circuit.cnot(1, 2);

        let mut state = State::zero_state(3);
        circuit.execute(&mut state);

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
}
