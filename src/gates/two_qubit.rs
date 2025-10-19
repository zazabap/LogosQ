//! Optimized two-qubit gate implementations

use super::*;

/// Optimized CNOT gate
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

        for i in 0..full_dim {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i ^ target_mask;
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

        for i in 0..full_dim {
            let b1 = (i & mask1) != 0;
            let b2 = (i & mask2) != 0;

            if b1 && !b2 {
                let j = i ^ mask1 ^ mask2;
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

        for i in 0..full_dim {
            if (i & both_mask) == both_mask {
                vector_slice[i] = -vector_slice[i];
            }
        }
    }
}

/// Optimized Controlled-Phase gate
pub struct CPhaseGate {
    pub control: usize,
    pub target: usize,
    pub angle: f64,
    pub num_qubits: usize,
}

impl Gate for CPhaseGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        let control_bit = n - 1 - self.control;
        let target_bit = n - 1 - self.target;

        let control_mask = 1 << control_bit;
        let target_mask = 1 << target_bit;
        let both_mask = control_mask | target_mask;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();
        let phase = Complex64::from_polar(1.0, self.angle);

        #[cfg(feature = "parallel")]
        {
            if full_dim > 1024 {
                use rayon::prelude::*;

                vector_slice
                    .par_iter_mut()
                    .enumerate()
                    .filter(|(i, _)| (*i & both_mask) == both_mask)
                    .for_each(|(_, val)| {
                        *val = *val * phase;
                    });
                return;
            }
        }

        for i in 0..full_dim {
            if (i & both_mask) == both_mask {
                vector_slice[i] *= phase;
            }
        }
    }
}

/// Convenience functions
pub fn cnot(control: usize, target: usize, num_qubits: usize) -> CNOTGate {
    CNOTGate {
        control,
        target,
        num_qubits,
    }
}

pub fn swap(qubit1: usize, qubit2: usize, num_qubits: usize) -> SWAPGate {
    SWAPGate {
        qubit1,
        qubit2,
        num_qubits,
    }
}

pub fn cz(control: usize, target: usize, num_qubits: usize) -> CZGate {
    CZGate {
        control,
        target,
        num_qubits,
    }
}

pub fn cphase(control: usize, target: usize, angle: f64, num_qubits: usize) -> CPhaseGate {
    CPhaseGate {
        control,
        target,
        angle,
        num_qubits,
    }
}
