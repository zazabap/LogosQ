use logosq::prelude::*;
use logosq::vis::{circuit_text, save_circuit_svg};
use std::f64::consts::PI;

fn main() {
    println!("Quantum Teleportation Protocol");
    println!("------------------------------");

    // Create a circuit with 3 qubits
    let mut circuit = Circuit::new(3).with_name("Quantum Teleportation");

    // Step 1: Create a state to teleport
    // |ψ⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩
    circuit.ry(0, PI / 4.0);
    println!("Preparing state to teleport: |ψ⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩");

    // Step 2: Create an entangled pair (Bell state) between qubits 1 and 2
    circuit.h(1);
    circuit.cnot(1, 2);
    println!("Created entangled Bell state between Alice and Bob");

    // Step 3: Alice performs the teleportation protocol
    circuit.cnot(0, 1);
    circuit.h(0);
    println!("Alice performs entanglement operation");

    // Create initial state - properly initialized zero state
    let initial_state = State::zero_state(3);

    // Execute the circuit up to this point
    let mut teleport_state_state = initial_state.clone();
    circuit.execute(&mut teleport_state_state);
    let teleport_state = teleport_state_state;

    // Simulate measurements on qubits 0 and 1
    // We'll manually calculate the probabilities from the state vector
    let m0 = manual_measure_qubit(&teleport_state, 0);
    let m1 = manual_measure_qubit(&teleport_state, 1);

    println!("Alice's measurements (simulated): m0={}, m1={}", m0, m1);

    // Create a new circuit that applies conditional operations based on measurements
    let mut bob_circuit = Circuit::new(3);

    // Apply the same initial operations to get to the same state
    bob_circuit.ry(0, PI / 4.0);
    bob_circuit.h(1);
    bob_circuit.cnot(1, 2);
    bob_circuit.cnot(0, 1);
    bob_circuit.h(0);

    // Apply Bob's corrections based on Alice's measurement results
    if m1 == 1 {
        bob_circuit.x(2);
        println!("Bob applies X gate based on m1");
    }

    if m0 == 1 {
        bob_circuit.z(2);
        println!("Bob applies Z gate based on m0");
    }

    // Execute Bob's full circuit
    let mut bob_state = initial_state.clone();
    bob_circuit.execute(&mut bob_state);
    let final_state = bob_state;

    // Calculate the probability of measuring |1⟩ on Bob's qubit
    let prob_one = calculate_prob_one(&final_state, 2);
    println!("Probability of Bob measuring |1⟩: {:.4}", prob_one);

    // Visualize the circuit
    println!("\nCircuit diagram:");
    println!("{}", circuit_text(&bob_circuit));

    // Save the circuit diagram
    save_circuit_svg(&bob_circuit, "quantum_teleportation.svg").unwrap_or_else(|e| {
        println!("Failed to save circuit diagram: {}", e);
    });

    println!("\nAnalysis:");
    println!("The initial state |ψ⟩ has been teleported from Alice to Bob.");
    println!("Bob's qubit should now be in the state |ψ⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩");
    println!(
        "Theoretical probability of measuring |1⟩: {:.4}",
        (PI / 8.0).sin().powi(2)
    );
}

// Helper function to manually calculate measurement probability and result
fn manual_measure_qubit(state: &State, qubit_idx: usize) -> usize {
    let mut prob_one = 0.0;

    // Calculate probability by summing up amplitudes where the qubit is 1
    for i in 0..state.vector.len() {
        if (i & (1 << qubit_idx)) != 0 {
            prob_one += state.vector[i].norm_sqr();
        }
    }

    // Deterministic result based on probability (for reproducibility)
    if prob_one > 0.5 {
        1
    } else {
        0
    }
}

// Helper function to calculate probability of measuring |1⟩
fn calculate_prob_one(state: &State, qubit_idx: usize) -> f64 {
    let mut prob_one = 0.0;

    for i in 0..state.vector.len() {
        if (i & (1 << qubit_idx)) != 0 {
            prob_one += state.vector[i].norm_sqr();
        }
    }

    prob_one
}
