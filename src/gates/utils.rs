//! Utility functions for gate operations

/// Helper to compute qubit bit position
#[inline]
pub fn qubit_bit_position(qubit: usize, num_qubits: usize) -> usize {
    num_qubits - 1 - qubit
}

/// Helper to compute qubit mask
#[inline]
pub fn qubit_mask(qubit: usize, num_qubits: usize) -> usize {
    1 << qubit_bit_position(qubit, num_qubits)
}

/// Check if a qubit is in state |1⟩ for a given basis state index
#[inline]
pub fn is_qubit_set(basis_index: usize, qubit: usize, num_qubits: usize) -> bool {
    (basis_index & qubit_mask(qubit, num_qubits)) != 0
}

/// Flip a qubit in a basis state index
#[inline]
pub fn flip_qubit(basis_index: usize, qubit: usize, num_qubits: usize) -> usize {
    basis_index ^ qubit_mask(qubit, num_qubits)
}
