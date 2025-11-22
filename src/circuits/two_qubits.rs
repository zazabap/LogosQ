use super::Circuit;
use crate::gates::two_qubit::{CNOTGate, CZGate, SWAPGate};
use ndarray::Array2;
use num_complex::Complex64;

// Import rayon conditionally
#[cfg(feature = "parallel")]

impl Circuit {
    /// Adds a two-qubit gate to the circuit
    pub fn add_two_qubit_gate(
        &mut self,
        gate_matrix: Array2<Complex64>,
        qubit1: usize,
        qubit2: usize,
        name: &str,
    ) {
        assert_eq!(
            gate_matrix.shape(),
            &[4, 4],
            "Gate must be a 4x4 matrix for two-qubit operation"
        );
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        // Sort qubits for consistency
        let (q1, q2) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        // For a 2-qubit system, add the matrix directly
        if self.num_qubits == 2 {
            self.add_matrix_gate(gate_matrix, vec![q1, q2], name)
                .expect("Failed to add two-qubit gate");
            return;
        }

        // For larger systems, expand the matrix to act on the full Hilbert space
        let full_dim = 1 << self.num_qubits; // 2^n
        let mut full_matrix = Array2::zeros((full_dim, full_dim));

        // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
        let bit_pos1 = self.num_qubits - 1 - q1;
        let bit_pos2 = self.num_qubits - 1 - q2;

        // Iterate through all basis states
        for i in 0..full_dim {
            // Extract bits for the target qubits
            let bit1 = (i >> bit_pos1) & 1;
            let bit2 = (i >> bit_pos2) & 1;

            // Calculate the 2-qubit subspace index (0-3)
            let subidx = (bit1 << 1) | bit2;

            for j in 0..4 {
                // Skip if matrix element is zero
                if gate_matrix[[j, subidx]] == Complex64::new(0.0, 0.0) {
                    continue;
                }

                // Calculate new state after gate application
                let new_bit1 = (j >> 1) & 1;
                let new_bit2 = j & 1;

                // Create the new state by modifying the bits at the correct bit positions
                let mut new_state = i;
                new_state = (new_state & !(1 << bit_pos1)) | (new_bit1 << bit_pos1);
                new_state = (new_state & !(1 << bit_pos2)) | (new_bit2 << bit_pos2);

                // Set the matrix element
                full_matrix[[new_state, i]] = gate_matrix[[j, subidx]];
            }
        }

        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), name)
            .expect("Failed to add two-qubit gate");
    }

    /// Adds a CNOT gate to the circuit - OPTIMIZED
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CNOTGate {
            control,
            target,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CNOT");
        self
    }

    /// Adds a SWAP gate to the circuit - OPTIMIZED
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        let gate = SWAPGate {
            qubit1,
            qubit2,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit1, qubit2], "SWAP");
        self
    }

    /// Adds a CZ gate to the circuit - OPTIMIZED
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CZGate {
            control,
            target,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CZ");
        self
    }
}
