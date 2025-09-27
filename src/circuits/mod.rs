use crate::gates::{cnot_gate, cz_gate, swap_gate, toffoli_gate};
use crate::gates::{h_gate, rx_gate, ry_gate, rz_gate, s_gate, t_gate, x_gate, y_gate, z_gate};
use crate::gates::{Gate, MatrixGate};
use crate::noise::NoiseModel;
use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::rc::Rc;

/// A quantum operation with an associated gate and target qubits
pub struct Operation {
    /// The gate to apply
    pub gate: Rc<dyn Gate>,
    /// The indices of qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Optional name for the operation (useful for debugging)
    pub name: String,
}

impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(qubits: {:?})", self.name, self.qubits)
    }
}

/// Represents a quantum circuit with operations and measurement
pub struct Circuit {
    /// Operations to apply in sequence
    pub operations: Vec<Operation>,
    /// Total number of qubits in the circuit
    pub num_qubits: usize,
    /// Optional name for the circuit
    pub name: Option<String>,
    /// Add this field for noise models
    pub noise_models: Vec<Rc<dyn NoiseModel>>,
}

impl Circuit {
    /// Creates a new quantum circuit with specified number of qubits
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            operations: Vec::new(),
            num_qubits,
            name: None,
            noise_models: Vec::new(),
        }
    }

    /// Sets a name for the circuit
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Adds a gate operation to the circuit
    pub fn add_operation<G: Gate + 'static>(&mut self, gate: G, qubits: Vec<usize>, name: &str) {
        // Validate qubit indices
        for &qubit in &qubits {
            assert!(
                qubit < self.num_qubits,
                "Qubit index {} out of range",
                qubit
            );
        }

        self.operations.push(Operation {
            gate: Rc::new(gate),
            qubits,
            name: name.to_string(),
        });
    }

    /// Adds a matrix gate to the circuit
    pub fn add_matrix_gate(&mut self, matrix: Array2<Complex64>, qubits: Vec<usize>, name: &str) {
        let gate = MatrixGate { matrix };
        self.add_operation(gate, qubits, name);
    }

    /// Executes the circuit on a given initial state (without noise)
    pub fn execute_without_noise<'a>(&self, initial_state: &'a mut State) -> &'a mut State {
        assert_eq!(
            initial_state.num_qubits, self.num_qubits,
            "Initial state must have the same number of qubits as the circuit"
        );

        for operation in &self.operations {
            // Apply the gate to the state
            operation.gate.apply(initial_state);
        }

        initial_state
    }

    /// Creates and executes the circuit on a new zero state
    pub fn execute_and_measure(&self) -> Vec<usize> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state);

        // Measure each qubit
        let mut results = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            results.push(state.measure_qubit(i));
        }

        results
    }

    /// Calculates the expectation value of an observable
    pub fn expectation(&self, observable: &dyn Gate) -> f64 {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state);

        // Create a copy of the state to apply the observable
        let mut obs_state = state.clone();
        observable.apply(&mut obs_state);

        // Calculate expectation value as <ψ|O|ψ>
        let mut expectation = Complex64::new(0.0, 0.0);
        for i in 0..state.vector.len() {
            expectation += state.vector[i].conj() * obs_state.vector[i];
        }

        expectation.re // Return the real part
    }

    /// Composes this circuit with another circuit
    pub fn compose(&mut self, other: &Circuit) {
        assert_eq!(
            self.num_qubits, other.num_qubits,
            "Cannot compose circuits with different numbers of qubits"
        );

        // Add all operations from the other circuit
        for op in &other.operations {
            let op_clone = Operation {
                gate: op.gate.clone(),
                qubits: op.qubits.clone(),
                name: op.name.clone(),
            };
            self.operations.push(op_clone);
        }
    }

    /// Creates a reversed version of this circuit
    pub fn reversed(&self) -> Self {
        let mut reversed = Circuit::new(self.num_qubits);

        // Add operations in reverse order
        for op in self.operations.iter().rev() {
            reversed.operations.push(Operation {
                gate: op.gate.clone(),
                qubits: op.qubits.clone(),
                name: format!("reversed_{}", op.name),
            });
        }

        if let Some(name) = &self.name {
            reversed.name = Some(format!("{}_reversed", name));
        }

        reversed
    }

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

    /// Adds a noise model to the circuit
    pub fn add_noise<N: NoiseModel + 'static>(&mut self, noise_model: N) -> &mut Self {
        self.noise_models.push(Rc::new(noise_model));
        self
    }

    /// Executes the circuit on a given initial state, applying noise if models are present
    pub fn execute<'a>(&self, initial_state: &'a mut State) -> &'a mut State {
        assert_eq!(
            initial_state.num_qubits, self.num_qubits,
            "Initial state must have the same number of qubits as the circuit"
        );

        let apply_noise = !self.noise_models.is_empty();

        for operation in &self.operations {
            // Apply the gate to the state
            operation.gate.apply(initial_state);

            // Apply noise if enabled
            if apply_noise {
                for noise_model in &self.noise_models {
                    noise_model.apply(initial_state);
                }
            }
        }

        initial_state
    }

    /// Creates and executes the circuit on a new zero state, with noise
    pub fn execute_and_measure_with_noise(&self) -> Vec<usize> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state);

        // Measure each qubit
        let mut results = Vec::with_capacity(self.num_qubits);
        let mut state_copy = state.clone();

        for i in 0..self.num_qubits {
            results.push(state_copy.measure_qubit(i));
        }

        results
    }

    // Convenience methods for common gates

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
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let rx = rx_gate(theta);
        self.add_single_qubit_gate(rx.matrix, qubit, &format!("RX({})", theta));
        self
    }

    /// Adds an Ry gate to the circuit
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let ry = ry_gate(theta);
        self.add_single_qubit_gate(ry.matrix, qubit, &format!("RY({})", theta));
        self
    }

    /// Adds an Rz gate to the circuit
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        let rz = rz_gate(theta);
        self.add_single_qubit_gate(rz.matrix, qubit, &format!("RZ({})", theta));
        self
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

        // Iterate through all basis states
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

/// Helper function for tensor product of matrices
fn tensor_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let result_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    let mut result = Array2::zeros(result_shape);

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    result[[i * b_shape[0] + k, j * b_shape[1] + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

impl fmt::Debug for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let circuit_name = self.name.as_deref().unwrap_or("Unnamed Circuit");
        writeln!(f, "Circuit: {} ({} qubits)", circuit_name, self.num_qubits)?;

        for (i, op) in self.operations.iter().enumerate() {
            writeln!(f, "{}: {:?}", i, op)?;
        }

        Ok(())
    }
}

// Implementation for the Clone trait for Circuit
impl Clone for Circuit {
    fn clone(&self) -> Self {
        let mut cloned = Circuit::new(self.num_qubits);
        cloned.name = self.name.clone();

        for op in &self.operations {
            cloned.operations.push(Operation {
                gate: op.gate.clone(),
                qubits: op.qubits.clone(),
                name: op.name.clone(),
            });
        }

        cloned
    }
}
