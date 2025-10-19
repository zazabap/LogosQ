//! Optimized single-qubit gate implementations
//!
//! These gates directly manipulate the state vector for maximum performance.

use super::*;
use std::f64::consts::SQRT_2;

/// Optimized Pauli-X gate
pub struct XGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for XGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;
                vector_slice.swap(idx0, idx1);
            }
        }
    }
}

/// Optimized Pauli-Y gate
pub struct YGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for YGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;

                let temp = vector_slice[idx0];
                vector_slice[idx0] = Complex64::new(0.0, 1.0) * vector_slice[idx1];
                vector_slice[idx1] = Complex64::new(0.0, -1.0) * temp;
            }
        }
    }
}

/// Optimized Pauli-Z gate
pub struct ZGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for ZGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let mask = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & mask) != 0 {
                vector_slice[i] = -vector_slice[i];
            }
        }
    }
}

/// Optimized Hadamard gate
pub struct HGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for HGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let sqrt_2_inv = Complex64::new(1.0 / SQRT_2, 0.0);
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;

                let amp0 = vector_slice[idx0];
                let amp1 = vector_slice[idx1];

                vector_slice[idx0] = sqrt_2_inv * (amp0 + amp1);
                vector_slice[idx1] = sqrt_2_inv * (amp0 - amp1);
            }
        }
    }
}

/// Optimized S gate (phase gate)
pub struct SGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for SGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let mask = 1 << qubit_bit;
        let full_dim = 1 << n;

        let phase = Complex64::new(0.0, 1.0);
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & mask) != 0 {
                vector_slice[i] *= phase;
            }
        }
    }
}

/// Optimized T gate (π/8 gate)
pub struct TGate {
    pub qubit: usize,
    pub num_qubits: usize,
}

impl Gate for TGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let qubit_bit = n - 1 - self.qubit;
        let mask = 1 << qubit_bit;
        let full_dim = 1 << n;

        let phase = Complex64::from_polar(1.0, std::f64::consts::PI / 4.0);
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & mask) != 0 {
                vector_slice[i] *= phase;
            }
        }
    }
}

/// Convenience function to create X gate
pub fn x(qubit: usize, num_qubits: usize) -> XGate {
    XGate { qubit, num_qubits }
}

/// Convenience function to create Y gate
pub fn y(qubit: usize, num_qubits: usize) -> YGate {
    YGate { qubit, num_qubits }
}

/// Convenience function to create Z gate
pub fn z(qubit: usize, num_qubits: usize) -> ZGate {
    ZGate { qubit, num_qubits }
}

/// Convenience function to create H gate
pub fn h(qubit: usize, num_qubits: usize) -> HGate {
    HGate { qubit, num_qubits }
}

/// Convenience function to create S gate
pub fn s(qubit: usize, num_qubits: usize) -> SGate {
    SGate { qubit, num_qubits }
}

/// Convenience function to create T gate
pub fn t(qubit: usize, num_qubits: usize) -> TGate {
    TGate { qubit, num_qubits }
}
