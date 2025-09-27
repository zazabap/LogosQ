// Quantum Fourier Transform implementation

use crate::circuits::Circuit;
use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Creates a Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Quantum Fourier Transform");

    // QFT implementation
    for i in 0..num_qubits {
        // Hadamard gate on the current qubit
        circuit.h(i);

        // Controlled phase rotations
        for j in (i + 1)..num_qubits {
            let angle = PI / (1 << (j - i)) as f64;
            controlled_phase(&mut circuit, i, j, angle);
        }
    }

    // Swap qubits to match classical FFT ordering
    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    circuit
}

/// Creates an inverse Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_inverse_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Inverse Quantum Fourier Transform");

    // Swap qubits first for inverse QFT
    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    // Apply inverse QFT
    for i in (0..num_qubits).rev() {
        // Controlled rotations in reverse with negated angles
        for j in (i + 1)..num_qubits {
            let angle = -PI / ((1 << (j - i)) as f64);
            controlled_phase(&mut circuit, i, j, angle);
        }

        // Hadamard on the current qubit
        circuit.h(i);
    }

    circuit
}

/// Applies the QFT to a quantum state
pub fn apply(state: &mut State) {
    println!("Applying QFT to state with {} qubits", state.num_qubits);
    let num_qubits = state.num_qubits;
    let circuit = create_circuit(num_qubits);
    circuit.execute(state);
}

/// Applies the inverse QFT to a quantum state
pub fn apply_inverse(state: &mut State) {
    let num_qubits = state.num_qubits;
    let circuit = create_inverse_circuit(num_qubits);
    circuit.execute(state);
}

/// Helper function to apply a controlled phase rotation.
/// Implements a controlled phase gate with rotation angle.
pub fn controlled_phase(circuit: &mut Circuit, control: usize, target: usize, angle: f64) {
    // Direct implementation of controlled phase gate
    let full_dim = 1 << circuit.num_qubits;
    let mut matrix = Array2::zeros((full_dim, full_dim));

    // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
    let control_bit = circuit.num_qubits - 1 - control;
    let target_bit = circuit.num_qubits - 1 - target;

    // Fill the matrix
    for i in 0..full_dim {
        // Extract control and target bits
        let control_val = (i >> control_bit) & 1;
        let target_val = (i >> target_bit) & 1;

        // Apply phase only when both control and target are 1
        let phase = if control_val == 1 && target_val == 1 {
            Complex64::from_polar(1.0, angle)
        } else {
            Complex64::new(1.0, 0.0)
        };

        matrix[[i, i]] = phase;
    }

    circuit.add_matrix_gate(
        matrix,
        (0..circuit.num_qubits).collect(),
        &format!("CP({:.4})", angle),
    );
}
