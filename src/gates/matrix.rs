//! Matrix-based gate implementations
//!
//! These implementations use full matrix representations and are useful for:
//! - Small quantum systems
//! - Reference implementations
//! - Testing and validation
//!
//! For larger systems, use the optimized implementations in other modules.

use super::*;
use std::f64::consts::SQRT_2;

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