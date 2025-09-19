// This file defines the algorithms module for the quantum computing library.
// It exports functions for implementing various quantum algorithms.

// Import necessary dependencies
use crate::circuits::Circuit;

// Declare and re-export each algorithm module
pub mod qft;
pub mod rqc;
pub mod xyz_heisenberg;

// Optional: Re-export commonly used functions at the module level
// for easier access (without having to use the submodule path)
pub use qft::{create_circuit as create_qft_circuit, apply as apply_qft};
pub use rqc::generate as generate_random_circuit;
pub use xyz_heisenberg::simulate as simulate_heisenberg;

// Example usage functions can stay in mod.rs as they demonstrate
// how to combine algorithms
pub fn quantum_fourier_transform_example(num_qubits: usize) -> Vec<usize> {
    // Create a circuit that demonstrates QFT
    let mut circuit = Circuit::new(num_qubits);

    // Put qubits into superposition
    for i in 0..num_qubits {
        circuit.h(i);
    }

    // Apply QFT
    let qft_circuit = qft::create_circuit(num_qubits);
    circuit.compose(&qft_circuit);

    // Measure all qubits
    circuit.execute_and_measure()
}

// Other example functions can be added here
