use super::utils::tensor_product;
use super::Circuit;
use crate::gates::matrix::{
    h_gate, rx_gate, ry_gate, rz_gate, s_gate, t_gate, x_gate, y_gate, z_gate,
};
use ndarray::Array2;
use num_complex::Complex64;

impl Circuit {
    /// Applies a single-qubit gate to a specific qubit in the circuit.
    /// This properly handles the expansion to the full system dimension.
    pub fn add_single_qubit_gate(
        &mut self,
        gate_matrix: Array2<Complex64>,
        target_qubit: usize,
        name: &str,
    ) {
        assert_eq!(
            gate_matrix.shape(),
            &[2, 2],
            "Gate must be a 2x2 matrix for single-qubit operation"
        );
        assert!(
            target_qubit < self.num_qubits,
            "Target qubit index out of range"
        );

        // Identity matrix
        let identity = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();

        // Start with the appropriate matrix based on the target qubit
        let mut full_matrix: Array2<Complex64>;

        if target_qubit == 0 {
            // For qubit 0, start with the gate matrix
            full_matrix = gate_matrix.clone();
        } else {
            // For other qubits, start with identity
            full_matrix = identity.clone();
        }

        // Tensor product with identity or gate for each remaining qubit
        for i in 1..self.num_qubits {
            if i == target_qubit {
                full_matrix = tensor_product(&full_matrix, &gate_matrix);
            } else {
                full_matrix = tensor_product(&full_matrix, &identity);
            }
        }

        // Add the expanded gate to the circuit
        self.add_matrix_gate(full_matrix, (0..self.num_qubits).collect(), name);
    }

    /// Adds a Pauli-X (NOT) gate to the circuit
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        let x = x_gate();
        self.add_single_qubit_gate(x.matrix, qubit, "X");
        self
    }

    /// Adds a Pauli-Y gate to the circuit
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        let y = y_gate();
        self.add_single_qubit_gate(y.matrix, qubit, "Y");
        self
    }

    /// Adds a Pauli-Z gate to the circuit
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        let z = z_gate();
        self.add_single_qubit_gate(z.matrix, qubit, "Z");
        self
    }

    /// Adds a Hadamard gate to the circuit
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        let h = h_gate();
        self.add_single_qubit_gate(h.matrix, qubit, "H");
        self
    }

    /// Adds an S gate to the circuit
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        let s = s_gate();
        self.add_single_qubit_gate(s.matrix, qubit, "S");
        self
    }

    /// Adds a T gate to the circuit
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        let t = t_gate();
        self.add_single_qubit_gate(t.matrix, qubit, "T");
        self
    }

    /// Adds an Rx gate to the circuit
    pub fn rx_matrix(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let rx = rx_gate(theta);
        self.add_single_qubit_gate(rx.matrix, qubit, &format!("RX({})", theta));
        self
    }

    /// Adds an Ry gate to the circuit
    pub fn ry_matrix(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let ry = ry_gate(theta);
        self.add_single_qubit_gate(ry.matrix, qubit, &format!("RY({})", theta));
        self
    }

    /// Adds an Rz gate to the circuit
    pub fn rz_matrix(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let rz = rz_gate(theta);
        self.add_single_qubit_gate(rz.matrix, qubit, &format!("RZ({})", theta));
        self
    }
}
