use super::Circuit;
use crate::gates::{cnot_gate, cz_gate, swap_gate, toffoli_gate};
use ndarray::Array2;
use num_complex::Complex64;

// Import rayon conditionally
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
            self.add_matrix_gate(gate_matrix, vec![q1, q2], name);
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

        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), name);
    }

    /// Adds a CNOT gate to the circuit
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        // For a 2-qubit system, use the standard CNOT matrix
        if self.num_qubits == 2 {
            let cnot = cnot_gate();
            // IMPORTANT: If control > target, we need to adjust the matrix
            // Standard CNOT matrix assumes control=0, target=1
            if control > target {
                // Create a modified matrix for CNOT with swapped control/target
                let mut swapped_matrix = Array2::zeros((4, 4));
                swapped_matrix[[0, 0]] = Complex64::new(1.0, 0.0);
                swapped_matrix[[1, 2]] = Complex64::new(1.0, 0.0);
                swapped_matrix[[2, 1]] = Complex64::new(1.0, 0.0);
                swapped_matrix[[3, 3]] = Complex64::new(1.0, 0.0);
                self.add_matrix_gate(swapped_matrix, vec![control, target], "CNOT");
            } else {
                self.add_matrix_gate(cnot.matrix, vec![control, target], "CNOT");
            }
            return self;
        }

        // For larger systems, create a custom matrix
        let full_dim = 1 << self.num_qubits; // 2^n
        let mut full_matrix = Array2::zeros((full_dim, full_dim));

        // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
        let control_bit = self.num_qubits - 1 - control;
        let target_bit = self.num_qubits - 1 - target;

        // Parallelize for large matrices
        #[cfg(feature = "parallel")]
        if full_dim > 1024 {
            // Rayon cannot mutate full_matrix in parallel directly.
            // Instead, build a vector of (row, col, value) tuples in parallel, then apply them sequentially.
            let updates: Vec<(usize, usize, Complex64)> = (0..full_dim)
                .into_par_iter()
                .map(|i| {
                    let control_val = (i >> control_bit) & 1;
                    if control_val == 0 {
                        (i, i, Complex64::new(1.0, 0.0))
                    } else {
                        let j = i ^ (1 << target_bit);
                        (j, i, Complex64::new(1.0, 0.0))
                    }
                })
                .collect();
            for (row, col, val) in updates {
                full_matrix[[row, col]] = val;
            }
            self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), "CNOT");
            return self;
        }

        // Sequential implementation for smaller matrices or when parallel feature is disabled
        for i in 0..full_dim {
            // Extract control bit
            let control_val = (i >> control_bit) & 1;

            if control_val == 0 {
                // If control is 0, leave state unchanged
                full_matrix[[i, i]] = Complex64::new(1.0, 0.0);
            } else {
                // If control is 1, flip the target bit
                let j = i ^ (1 << target_bit);
                full_matrix[[j, i]] = Complex64::new(1.0, 0.0);
            }
        }

        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), "CNOT");
        self
    }

    /// Adds a SWAP gate to the circuit
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        // For a 2-qubit system, use the standard SWAP matrix
        if self.num_qubits == 2 {
            let swap = swap_gate();
            self.add_matrix_gate(swap.matrix, vec![qubit1, qubit2], "SWAP");
            return self;
        }

        // For larger systems, create a custom matrix
        let full_dim = 1 << self.num_qubits; // 2^n
        let mut full_matrix = Array2::zeros((full_dim, full_dim));

        // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
        let bit_pos1 = self.num_qubits - 1 - qubit1;
        let bit_pos2 = self.num_qubits - 1 - qubit2;

        // Iterate through all basis states
        for i in 0..full_dim {
            // Extract bits for the qubits
            let bit1 = (i >> bit_pos1) & 1;
            let bit2 = (i >> bit_pos2) & 1;

            if bit1 == bit2 {
                // If bits are the same, no change needed
                full_matrix[[i, i]] = Complex64::new(1.0, 0.0);
            } else {
                // If bits are different, swap them
                let j = i ^ (1 << bit_pos1) ^ (1 << bit_pos2);
                full_matrix[[j, i]] = Complex64::new(1.0, 0.0);
            }
        }

        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), "SWAP");
        self
    }

    /// Adds a CZ gate to the circuit
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        // For a 2-qubit system, use the standard CZ matrix
        if self.num_qubits == 2 {
            let cz = cz_gate();
            self.add_matrix_gate(cz.matrix, vec![control, target], "CZ");
            return self;
        }

        // For larger systems, create a custom matrix
        let full_dim = 1 << self.num_qubits; // 2^n
        let mut full_matrix = Array2::zeros((full_dim, full_dim));

        // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
        let control_bit = self.num_qubits - 1 - control;
        let target_bit = self.num_qubits - 1 - target;

        // Iterate through all basis states
        for i in 0..full_dim {
            // Extract control and target bits
            let control_val = (i >> control_bit) & 1;
            let target_val = (i >> target_bit) & 1;

            if control_val == 1 && target_val == 1 {
                // If both are 1, apply phase -1
                full_matrix[[i, i]] = Complex64::new(-1.0, 0.0);
            } else {
                // Otherwise, leave state unchanged
                full_matrix[[i, i]] = Complex64::new(1.0, 0.0);
            }
        }

        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), "CZ");
        self
    }

    /// Adds a Toffoli gate to the circuit
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        let toffoli = toffoli_gate();
        // For now, we'll just add the matrix directly
        self.add_matrix_gate(toffoli.matrix, vec![control1, control2, target], "Toffoli");
        self
    }
}
