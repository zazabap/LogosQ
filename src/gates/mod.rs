// This file defines the gates module for the quantum computing library.
// It exports types and functions for creating and manipulating various quantum gates.

// pub mod pauli;
// pub mod hadamard;
// pub mod cnot;

// pub use pauli::PauliGate;
// pub use hadamard::HadamardGate;
// pub use cnot::CNOTGate;

// pub trait Gate {
//     fn apply(&self, state: &mut [f64]);
// }

// pub fn create_pauli_x() -> PauliGate {
//     PauliGate::new('X')
// }

// pub fn create_hadamard() -> HadamardGate {
//     HadamardGate::new()
// }

// pub fn create_cnot(control: usize, target: usize) -> CNOTGate {
//     CNOTGate::new(control, target)
// }

use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait for quantum gates.
pub trait Gate {
    fn apply(&self, state: &mut State);
}

/// Implementation of a generic matrix gate.
pub struct MatrixGate {
    pub matrix: Array2<Complex64>,
}

impl Gate for MatrixGate {
    fn apply(&self, state: &mut State) {
        // Check matrix dimensions
        assert_eq!(
            self.matrix.shape()[1],
            state.vector.len(),
            "Matrix columns must match state dimension"
        );

        state.vector = self.matrix.dot(&state.vector);
        state.normalize();
    }
}
