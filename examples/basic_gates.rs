// This file demonstrates the usage of the quantum computing library to create and manipulate basic quantum gates.

use logosq::circuits::Circuit;
use logosq::gates::{cnot_gate, h_gate, x_gate, Gate};
use logosq::states::State;

fn main() {
    // Create single-qubit gates
    println!("Creating basic quantum gates:");
    let x = x_gate();
    let h = h_gate();
    println!("Created basic gates: X, H, Z");

    // Demonstrate gates on quantum states
    println!("\nDemonstrating gate operations:");

    // X gate flips |0⟩ to |1⟩
    let mut state = State::zero_state(1);
    println!("Initial state: |0⟩");
    println!("Probability of |0⟩: {:.4}", state.probability(0));
    println!("Probability of |1⟩: {:.4}", state.probability(1));

    x.apply(&mut state);
    println!("\nAfter X gate:");
    println!("Probability of |0⟩: {:.4}", state.probability(0));
    println!("Probability of |1⟩: {:.4}", state.probability(1));

    // Hadamard creates superposition
    let mut state = State::zero_state(1);
    h.apply(&mut state);
    println!("\nAfter H gate on |0⟩:");
    println!("Probability of |0⟩: {:.4}", state.probability(0));
    println!("Probability of |1⟩: {:.4}", state.probability(1));

    // Building a Bell state circuit
    println!("\nCreating a Bell state:");
    let mut circuit = Circuit::new(2);

    // Add Hadamard to first qubit
    circuit.add_single_qubit_gate(h.matrix.clone(), 0, "H");

    // Add CNOT between qubits 0 and 1
    let cnot = cnot_gate();
    circuit.add_matrix_gate(cnot.matrix.clone(), vec![0, 1], "CNOT");

    // Execute circuit on |00⟩ state
    let mut state = State::zero_state(2);
    circuit.execute(&mut state);

    // Bell state should have equal probabilities for |00⟩ and |11⟩
    println!("Bell state probabilities:");
    println!("Probability of |00⟩: {:.4}", state.probability(0));
    println!("Probability of |01⟩: {:.4}", state.probability(1));
    println!("Probability of |10⟩: {:.4}", state.probability(2));
    println!("Probability of |11⟩: {:.4}", state.probability(3));

    // Demonstrate measurement
    println!("\nMeasuring Bell state 100 times:");
    let mut zeros_zeros = 0;
    let mut ones_ones = 0;

    for _ in 0..100 {
        let test_state = state.clone();
        let result = test_state.measure();
        if result == 0 {
            zeros_zeros += 1;
        } else if result == 3 {
            ones_ones += 1;
        }
    }

    println!("Measured |00⟩: {} times", zeros_zeros);
    println!("Measured |11⟩: {} times", ones_ones);
}
