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
/// Gates must be thread-safe (Send + Sync) to support parallel execution.
///
/// # Example
///
/// ```rust
/// use logosq::prelude::*;
///
/// let mut state = State::zero_state(1);
/// let gate = x(0, 1);
/// gate.apply(&mut state);
/// ```
pub trait Gate: Send + Sync {
    /// Apply the gate to a quantum state.
    ///
    /// This method modifies the state vector in place.
    /// The state will be automatically normalized after gate application.
    ///
    /// # Arguments
    /// * `state` - The quantum state to apply the gate to
    fn apply(&self, state: &mut State);
}

/// Generic matrix-based gate implementation.
///
/// This is useful for small systems or as a reference implementation.
/// For larger systems, use the optimized gate implementations which directly
/// manipulate the state vector for better performance.
///
/// # Example
///
/// ```rust
/// use logosq::prelude::*;
/// use ndarray::Array2;
///
/// let matrix = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// let gate = MatrixGate { matrix };
/// ```
#[derive(Clone, Debug)]
pub struct MatrixGate {
    /// The gate matrix (must be unitary)
    pub matrix: Array2<Complex64>,
}

impl MatrixGate {
    /// Creates a new matrix gate.
    ///
    /// # Arguments
    /// * `matrix` - The gate matrix (should be unitary)
    pub fn new(matrix: Array2<Complex64>) -> Self {
        Self { matrix }
    }

    /// Returns a reference to the gate matrix.
    pub fn matrix(&self) -> &Array2<Complex64> {
        &self.matrix
    }
}

impl Gate for MatrixGate {
    fn apply(&self, state: &mut State) {
        let state_vec = state.vector();
        let state_dim = state_vec.len();

        if self.matrix.shape()[1] != state_dim {
            panic!(
                "Matrix columns ({}) must match state dimension ({})",
                self.matrix.shape()[1],
                state_dim
            );
        }

        let new_vector = self.matrix.dot(state_vec);
        *state.vector_mut() = new_vector;
        state.normalize();
    }
}

// Module declarations
pub mod matrix; // Matrix-based implementations
pub mod parameterized; // Parameterized gates for VQC
pub mod single_qubit; // Optimized single-qubit gates
pub mod three_qubit; // Optimized three-qubit gates
pub mod two_qubit; // Optimized two-qubit gates
pub mod utils; // Helper functions

// Re-exports for convenience
pub use parameterized::*;
pub use single_qubit::*;
pub use three_qubit::*;
pub use two_qubit::*;

// Matrix-based gates (prefixed to avoid conflicts)
pub use matrix::{
    cnot_gate as matrix_cnot, cz_gate as matrix_cz, h_gate as matrix_h, rx_gate as matrix_rx,
    ry_gate as matrix_ry, rz_gate as matrix_rz, s_gate as matrix_s, swap_gate as matrix_swap,
    t_gate as matrix_t, toffoli_gate as matrix_toffoli, x_gate as matrix_x, y_gate as matrix_y,
    z_gate as matrix_z,
};
