//! Quantum gate implementations
//!
//! This module provides various quantum gate implementations:
//! - Matrix-based gates (for small systems and reference)
//! - Optimized gates (direct state vector manipulation)
//! - Parameterized gates (for variational quantum circuits)

use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait for quantum gates.
///
/// All gate implementations must implement this trait to be applied to quantum states.
pub trait Gate: Send + Sync {
    /// Apply the gate to a quantum state
    fn apply(&self, state: &mut State);
}

/// Generic matrix-based gate implementation
///
/// This is useful for small systems or as a reference implementation.
/// For larger systems, use the optimized gate implementations.
#[derive(Clone)]
pub struct MatrixGate {
    pub matrix: Array2<Complex64>,
}

impl Gate for MatrixGate {
    fn apply(&self, state: &mut State) {
        assert_eq!(
            self.matrix.shape()[1],
            state.vector.len(),
            "Matrix columns must match state dimension"
        );

        state.vector = self.matrix.dot(&state.vector);
        state.normalize();
    }
}

// Module declarations
pub mod matrix;           // Matrix-based implementations
pub mod single_qubit;    // Optimized single-qubit gates
pub mod two_qubit;       // Optimized two-qubit gates
pub mod three_qubit;     // Optimized three-qubit gates
pub mod parameterized;   // Parameterized gates for VQC
pub mod utils;           // Helper functions

// Re-exports for convenience
pub use single_qubit::*;
pub use two_qubit::*;
pub use three_qubit::*;
pub use parameterized::*;

// Matrix-based gates (prefixed to avoid conflicts)
pub use matrix::{
    x_gate as matrix_x,
    y_gate as matrix_y,
    z_gate as matrix_z,
    h_gate as matrix_h,
    s_gate as matrix_s,
    t_gate as matrix_t,
    rx_gate as matrix_rx,
    ry_gate as matrix_ry,
    rz_gate as matrix_rz,
    cnot_gate as matrix_cnot,
    swap_gate as matrix_swap,
    cz_gate as matrix_cz,
    toffoli_gate as matrix_toffoli,
};
