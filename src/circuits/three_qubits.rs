use super::Circuit;
use ndarray::Array2;
use num_complex::Complex64;

impl Circuit {
    /// Adds a Toffoli gate to the circuit
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        assert!(
            control1 < self.num_qubits && control2 < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(
            control1 != control2 && control1 != target && control2 != target,
            "Control and target qubits must be different"
        );

        // Build the full Toffoli matrix for the entire system
        let full_dim = 1 << self.num_qubits; // 2^n
        let mut full_matrix = Array2::eye(full_dim); // Start with identity

        // Convert qubit indices to bit positions (MSB = qubit 0, LSB = qubit n-1)
        let control1_bit = self.num_qubits - 1 - control1;
        let control2_bit = self.num_qubits - 1 - control2;
        let target_bit = self.num_qubits - 1 - target;

        let control1_mask = 1 << control1_bit;
        let control2_mask = 1 << control2_bit;
        let target_mask = 1 << target_bit;

        // Modify the identity matrix to implement Toffoli
        // For each basis state where both controls are 1, swap with state where target is flipped
        for i in 0..full_dim {
            let control1_val = (i & control1_mask) != 0;
            let control2_val = (i & control2_mask) != 0;
            let target_val = (i & target_mask) != 0;

            if control1_val && control2_val {
                // Both controls are 1, so we need to flip the target
                let j = i ^ target_mask; // Flip the target bit

                // Only process each pair once (when target is 0 in state i)
                if !target_val {
                    // Clear the diagonal entries for i and j
                    full_matrix[[i, i]] = Complex64::new(0.0, 0.0);
                    full_matrix[[j, j]] = Complex64::new(0.0, 0.0);
                    
                    // Set the swap entries
                    full_matrix[[i, j]] = Complex64::new(1.0, 0.0);
                    full_matrix[[j, i]] = Complex64::new(1.0, 0.0);
                }
            }
        }

        // Apply to all qubits (the matrix already encodes which qubits are affected)
        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), "Toffoli");
        
        self
    }
}