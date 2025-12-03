//! Quantum state backend abstraction
//!
//! This module provides a trait for abstracting over different quantum state
//! representations (dense vectors, tensor networks, etc.)

use crate::error::Result;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait for quantum state backends that can execute quantum circuits.
///
/// This trait allows circuits to execute on different state representations
/// (dense vectors, MPS, etc.) without knowing the specific implementation.
pub trait QuantumStateBackend: Send + Sync {
    /// Returns the number of qubits in the state.
    fn num_qubits(&self) -> usize;

    /// Applies a single-qubit gate matrix to the specified qubit.
    ///
    /// # Arguments
    /// * `qubit` - The qubit index (0-indexed)
    /// * `matrix` - The 2x2 unitary gate matrix
    fn apply_single_qubit_matrix(&mut self, qubit: usize, matrix: &Array2<Complex64>)
        -> Result<()>;

    /// Applies a two-qubit gate matrix to the specified qubits.
    ///
    /// # Arguments
    /// * `control` - The control qubit index (0-indexed)
    /// * `target` - The target qubit index (0-indexed)
    /// * `matrix` - The 4x4 unitary gate matrix
    fn apply_two_qubit_matrix(
        &mut self,
        control: usize,
        target: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()>;

    /// Applies a three-qubit gate matrix to the specified qubits.
    ///
    /// # Arguments
    /// * `q1` - First qubit index (0-indexed)
    /// * `q2` - Second qubit index (0-indexed)
    /// * `q3` - Third qubit index (0-indexed)
    /// * `matrix` - The 8x8 unitary gate matrix
    fn apply_three_qubit_matrix(
        &mut self,
        q1: usize,
        q2: usize,
        q3: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()>;

    /// Applies a general gate matrix to the full state.
    ///
    /// This is a fallback for gates that don't fit the single/two/three-qubit patterns.
    /// # Arguments
    /// * `matrix` - The full unitary gate matrix (2^n x 2^n for n qubits)
    fn apply_full_matrix(&mut self, matrix: &Array2<Complex64>) -> Result<()>;
}
