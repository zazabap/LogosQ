//! Optimized three-qubit gate implementations

use super::*;

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

        for i in 0..full_dim {
            if (i & controls_mask) == controls_mask && (i & target_mask) == 0 {
                let j = i ^ target_mask;
                vector_slice.swap(i, j);
            }
        }
    }
}

/// Convenience function
pub fn toffoli(control1: usize, control2: usize, target: usize, num_qubits: usize) -> ToffoliGate {
    ToffoliGate {
        control1,
        control2,
        target,
        num_qubits,
    }
}
