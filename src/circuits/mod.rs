use crate::gates::{Gate, MatrixGate};
use crate::noise::NoiseModel;
use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::rc::Rc;

mod single_qubit;
mod two_qubits;
mod utils;

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
}

/// Helper function for tensor product of matrices

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
