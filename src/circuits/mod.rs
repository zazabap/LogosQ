//! Quantum circuit representation and execution
//!
//! This module provides the `Circuit` struct for building and executing
//! quantum circuits composed of gates and noise models.

use crate::error::{LogosQError, Result};
use crate::gates::{Gate, MatrixGate};
use crate::noise::NoiseModel;
use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod parameterized;
mod single_qubit;
mod three_qubits;
mod two_qubits;
mod utils;

// Circuit builder methods are implemented as impl blocks in submodules
// They are accessible directly on Circuit instances

/// A quantum operation with an associated gate and target qubits
#[derive(Clone)]
pub struct Operation {
    gate: Arc<dyn Gate>,
    qubits: Vec<usize>,
    name: String,
}

impl Operation {
    /// Creates a new operation
    pub fn new<G: Gate + 'static>(gate: G, qubits: Vec<usize>, name: String) -> Self {
        Self {
            gate: Arc::new(gate),
            qubits,
            name,
        }
    }

    /// Returns a reference to the gate
    pub fn gate(&self) -> &Arc<dyn Gate> {
        &self.gate
    }

    /// Returns the qubit indices
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// Returns the operation name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(qubits: {:?})", self.name, self.qubits)
    }
}

/// Represents a quantum circuit with operations and measurement
///
/// # Example
///
/// ```rust
/// use logosq::prelude::*;
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(0).cnot(0, 1);
///
/// let mut state = State::zero_state(2);
/// circuit.execute(&mut state).unwrap();
/// ```
#[derive(Clone)]
pub struct Circuit {
    operations: Vec<Operation>,
    num_qubits: usize,
    name: Option<String>,
    noise_models: Vec<Arc<dyn NoiseModel>>,
}

impl Circuit {
    /// Creates a new quantum circuit with specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - The number of qubits in the circuit
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

    /// Returns the number of qubits in the circuit
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the number of operations in the circuit
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Returns a reference to the operations
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Returns the circuit name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Adds a gate operation to the circuit
    ///
    /// # Arguments
    /// * `gate` - The gate to add
    /// * `qubits` - The qubit indices the gate acts on
    /// * `name` - A name for the operation
    ///
    /// # Errors
    /// Returns an error if any qubit index is out of range
    pub fn add_operation<G: Gate + 'static>(
        &mut self,
        gate: G,
        qubits: Vec<usize>,
        name: &str,
    ) -> Result<()> {
        // Validate qubit indices
        for &qubit in &qubits {
            if qubit >= self.num_qubits {
                return Err(LogosQError::InvalidQubitIndex {
                    index: qubit,
                    num_qubits: self.num_qubits,
                });
            }
        }

        self.operations.push(Operation::new(gate, qubits, name.to_string()));
        Ok(())
    }

    /// Internal helper for builder pattern methods
    /// Panics on error (appropriate for builder APIs)
    pub(crate) fn add_operation_unchecked<G: Gate + 'static>(
        &mut self,
        gate: G,
        qubits: Vec<usize>,
        name: &str,
    ) {
        self.add_operation(gate, qubits, name)
            .expect("Invalid qubit index in circuit builder");
    }

    /// Adds a matrix gate to the circuit
    ///
    /// # Arguments
    /// * `matrix` - The gate matrix
    /// * `qubits` - The qubit indices the gate acts on
    /// * `name` - A name for the operation
    ///
    /// # Errors
    /// Returns an error if any qubit index is out of range
    pub fn add_matrix_gate(
        &mut self,
        matrix: Array2<Complex64>,
        qubits: Vec<usize>,
        name: &str,
    ) -> Result<()> {
        let gate = MatrixGate { matrix };
        self.add_operation(gate, qubits, name)
    }

    /// Executes the circuit on a given initial state (without noise)
    ///
    /// # Arguments
    /// * `initial_state` - The initial quantum state
    ///
    /// # Errors
    /// Returns an error if the state has a different number of qubits than the circuit
    pub fn execute_without_noise(&self, initial_state: &mut State) -> Result<()> {
        if initial_state.num_qubits() != self.num_qubits {
            return Err(LogosQError::CircuitQubitMismatch {
                circuit_qubits: self.num_qubits,
                state_qubits: initial_state.num_qubits(),
            });
        }

        for operation in &self.operations {
            operation.gate.apply(initial_state);
        }

        Ok(())
    }

    /// Creates and executes the circuit on a new zero state, then measures all qubits
    ///
    /// # Returns
    /// A vector of measurement results (0 or 1) for each qubit
    #[cfg(feature = "parallel")]
    pub fn execute_and_measure(&self) -> Result<Vec<usize>> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state)?;

        // Make a copy of the state for parallel measurements
        let state_copy = state.clone();

        // Measure each qubit in parallel
        let results: Result<Vec<usize>> = (0..self.num_qubits)
            .into_par_iter()
            .map(|i| {
                let mut local_state = state_copy.clone();
                local_state.measure_qubit(i)
            })
            .collect();

        results
    }

    /// Creates and executes the circuit on a new zero state, then measures all qubits (sequential)
    #[cfg(not(feature = "parallel"))]
    pub fn execute_and_measure(&self) -> Result<Vec<usize>> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state)?;

        let mut results = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            results.push(state.measure_qubit(i)?);
        }

        Ok(results)
    }

    /// Calculates the expectation value of an observable
    ///
    /// # Arguments
    /// * `observable` - The observable (must implement Gate trait)
    ///
    /// # Returns
    /// The expectation value ⟨ψ|O|ψ⟩
    pub fn expectation(&self, observable: &dyn Gate) -> Result<f64> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state)?;

        // Create a copy of the state to apply the observable
        let mut obs_state = state.clone();
        observable.apply(&mut obs_state);

        #[cfg(feature = "parallel")]
        {
            let state_vec = state.vector();
            let obs_vec = obs_state.vector();
            let result = (0..state_vec.len())
                .into_par_iter()
                .map(|i| state_vec[i].conj() * obs_vec[i])
                .reduce(|| Complex64::new(0.0, 0.0), |a, b| a + b)
                .re;
            Ok(result)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let state_vec = state.vector();
            let obs_vec = obs_state.vector();
            let result: Complex64 = state_vec
                .iter()
                .zip(obs_vec.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            Ok(result.re)
        }
    }

    /// Composes this circuit with another circuit
    ///
    /// Appends all operations from `other` to this circuit.
    ///
    /// # Arguments
    /// * `other` - The circuit to compose with
    ///
    /// # Errors
    /// Returns an error if the circuits have different numbers of qubits
    pub fn compose(&mut self, other: &Circuit) -> Result<()> {
        if self.num_qubits != other.num_qubits {
            return Err(LogosQError::CircuitQubitMismatch {
                circuit_qubits: self.num_qubits,
                state_qubits: other.num_qubits,
            });
        }

        // Add all operations from the other circuit
        for op in &other.operations {
            self.operations.push(op.clone());
        }

        Ok(())
    }

    /// Creates a reversed version of this circuit
    ///
    /// Returns a new circuit with operations in reverse order.
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
    ///
    /// # Arguments
    /// * `noise_model` - The noise model to add
    pub fn add_noise<N: NoiseModel + 'static>(&mut self, noise_model: N) -> &mut Self {
        self.noise_models.push(Arc::new(noise_model));
        self
    }

    /// Executes the circuit on a given initial state, applying noise if models are present
    ///
    /// # Arguments
    /// * `initial_state` - The initial quantum state
    ///
    /// # Errors
    /// Returns an error if the state has a different number of qubits than the circuit
    pub fn execute(&self, initial_state: &mut State) -> Result<()> {
        if initial_state.num_qubits() != self.num_qubits {
            return Err(LogosQError::CircuitQubitMismatch {
                circuit_qubits: self.num_qubits,
                state_qubits: initial_state.num_qubits(),
            });
        }

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

        Ok(())
    }

    /// Creates and executes the circuit on a new zero state, with noise, then measures
    ///
    /// # Returns
    /// A vector of measurement results (0 or 1) for each qubit
    #[cfg(feature = "parallel")]
    pub fn execute_and_measure_with_noise(&self) -> Result<Vec<usize>> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state)?;

        // Make a copy of the state for parallel measurements
        let state_copy = state.clone();

        // Measure each qubit in parallel
        let results: Result<Vec<usize>> = (0..self.num_qubits)
            .into_par_iter()
            .map(|i| {
                let mut local_state = state_copy.clone();
                local_state.measure_qubit(i)
            })
            .collect();

        results
    }

    /// Creates and executes the circuit on a new zero state, with noise, then measures (sequential)
    #[cfg(not(feature = "parallel"))]
    pub fn execute_and_measure_with_noise(&self) -> Result<Vec<usize>> {
        let mut state = State::zero_state(self.num_qubits);
        self.execute(&mut state)?;

        let mut results = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            results.push(state.measure_qubit(i)?);
        }

        Ok(results)
    }
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
