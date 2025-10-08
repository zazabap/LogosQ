// This file defines the gates module for the quantum computing library.
// It exports types and functions for creating and manipulating various quantum gates.

use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

/// Trait for quantum gates.
pub trait Gate: Send + Sync {
    fn apply(&self, state: &mut State);
}

/// Implementation of a generic matrix gate.
#[derive(Clone)]
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

// Optimized gate implementations
pub mod optimized {
    use super::*;

    /// Optimized CNOT gate that directly manipulates state vector
    pub struct CNOTGate {
        pub control: usize,
        pub target: usize,
        pub num_qubits: usize,
    }

    impl Gate for CNOTGate {
        fn apply(&self, state: &mut State) {
            let n = state.num_qubits;
            let control_bit = n - 1 - self.control;
            let target_bit = n - 1 - self.target;

            let control_mask = 1 << control_bit;
            let target_mask = 1 << target_bit;

            let full_dim = 1 << n;
            let vector_slice = state.vector.as_slice_mut().unwrap();

            // For each basis state where control=1 and target=0, swap with target=1
            for i in 0..full_dim {
                if (i & control_mask) != 0 && (i & target_mask) == 0 {
                    let j = i ^ target_mask; // Flip target bit
                    vector_slice.swap(i, j);
                }
            }
        }
    }

    /// Optimized SWAP gate
    pub struct SWAPGate {
        pub qubit1: usize,
        pub qubit2: usize,
        pub num_qubits: usize,
    }

    impl Gate for SWAPGate {
        fn apply(&self, state: &mut State) {
            let n = state.num_qubits;
            let bit1 = n - 1 - self.qubit1;
            let bit2 = n - 1 - self.qubit2;

            let mask1 = 1 << bit1;
            let mask2 = 1 << bit2;

            let full_dim = 1 << n;
            let vector_slice = state.vector.as_slice_mut().unwrap();

            // For each basis state where bits differ, swap with the state where bits are swapped
            for i in 0..full_dim {
                let b1 = (i & mask1) != 0;
                let b2 = (i & mask2) != 0;

                // Only process if bits differ and we haven't processed this pair yet
                if b1 && !b2 {
                    let j = i ^ mask1 ^ mask2; // Flip both bits
                    vector_slice.swap(i, j);
                }
            }
        }
    }

    /// Optimized CZ gate
    pub struct CZGate {
        pub control: usize,
        pub target: usize,
        pub num_qubits: usize,
    }

    impl Gate for CZGate {
        fn apply(&self, state: &mut State) {
            let n = state.num_qubits;
            let control_bit = n - 1 - self.control;
            let target_bit = n - 1 - self.target;

            let control_mask = 1 << control_bit;
            let target_mask = 1 << target_bit;
            let both_mask = control_mask | target_mask;

            let full_dim = 1 << n;
            let vector_slice = state.vector.as_slice_mut().unwrap();

            // Apply -1 phase where both control and target are 1
            for i in 0..full_dim {
                if (i & both_mask) == both_mask {
                    vector_slice[i] = -vector_slice[i];
                }
            }
        }
    }

    /// Optimized Toffoli gate
    pub struct ToffoliGate {
        pub control1: usize,
        pub control2: usize,
        pub target: usize,
        pub num_qubits: usize,
    }

    impl Gate for ToffoliGate {
        fn apply(&self, state: &mut State) {
            let n = state.num_qubits;
            let c1_bit = n - 1 - self.control1;
            let c2_bit = n - 1 - self.control2;
            let target_bit = n - 1 - self.target;

            let c1_mask = 1 << c1_bit;
            let c2_mask = 1 << c2_bit;
            let target_mask = 1 << target_bit;
            let controls_mask = c1_mask | c2_mask;

            let full_dim = 1 << n;
            let vector_slice = state.vector.as_slice_mut().unwrap();

            // For each basis state where both controls=1 and target=0, swap with target=1
            for i in 0..full_dim {
                if (i & controls_mask) == controls_mask && (i & target_mask) == 0 {
                    let j = i ^ target_mask; // Flip target bit
                    vector_slice.swap(i, j);
                }
            }
        }
    }
}

// Common single-qubit gates
pub mod single_qubit {
    use super::*;

    /// Creates a Pauli-X (NOT) gate
    pub fn x_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a Pauli-Y gate
    pub fn y_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a Pauli-Z gate
    pub fn z_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a Hadamard gate
    pub fn h_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / SQRT_2, 0.0),
                Complex64::new(1.0 / SQRT_2, 0.0),
                Complex64::new(1.0 / SQRT_2, 0.0),
                Complex64::new(-1.0 / SQRT_2, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates an S gate (phase gate)
    pub fn s_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a T gate (π/8 gate)
    pub fn t_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0 / SQRT_2, 1.0 / SQRT_2),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a rotation gate around the X axis
    pub fn rx_gate(theta: f64) -> MatrixGate {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(cos, 0.0),
                Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin),
                Complex64::new(cos, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a rotation gate around the Y axis
    pub fn ry_gate(theta: f64) -> MatrixGate {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(cos, 0.0),
                Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0),
                Complex64::new(cos, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a rotation gate around the Z axis
    pub fn rz_gate(theta: f64) -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new((theta / 2.0).cos(), -(theta / 2.0).sin()),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new((theta / 2.0).cos(), (theta / 2.0).sin()),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }
}

// Multi-qubit gates (kept for compatibility, but not used internally)
pub mod multi_qubit {
    use super::*;

    /// Creates a CNOT (Controlled-NOT) gate matrix
    pub fn cnot_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a SWAP gate matrix
    pub fn swap_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a controlled-Z gate matrix
    pub fn cz_gate() -> MatrixGate {
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

        MatrixGate { matrix }
    }

    /// Creates a Toffoli (CCNOT) gate matrix
    pub fn toffoli_gate() -> MatrixGate {
        let mut matrix = Array2::from_elem((8, 8), Complex64::new(0.0, 0.0));

        // Diagonal elements
        for i in 0..6 {
            matrix[[i, i]] = Complex64::new(1.0, 0.0);
        }

        // Off-diagonal elements for the controlled-not part
        matrix[[6, 7]] = Complex64::new(1.0, 0.0);
        matrix[[7, 6]] = Complex64::new(1.0, 0.0);

        MatrixGate { matrix }
    }
}

// Convenience re-exports
pub use multi_qubit::*;
pub use optimized::*;
pub use single_qubit::*;
