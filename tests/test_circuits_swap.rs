use logosq::circuits::Circuit;
use logosq::states::State;

use logosq::vis::circuit::text_diagram;

mod tests {
    use super::*;
    // Adding test for swap gate functionality
    // for multiple qubits and ensuring it behaves as expected
    #[test]
    fn test_swap_gate_functionality() {
        // Create a 3-qubit circuit
        let mut circuit = Circuit::new(3);
        // Prepare initial state |100⟩ (first qubit is 1)
        circuit.x(0);

        println!("{}", text_diagram(&circuit));

        // Add a SWAP between qubits 0 and 2
        circuit.swap(0, 2);

        println!("{}", text_diagram(&circuit));

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
}
