//! Variational Quantum Circuit (VQC) for quantum machine learning
//!
//! This module provides flexible VQC implementations that can be used
//! as quantum layers in hybrid quantum-classical models like QLSTM.

use crate::circuits::Circuit;
use crate::optimization::ansatz::Ansatz;
use crate::states::State;
use std::f64::consts::PI;

/// Type of VQC architecture
#[derive(Clone, Debug, PartialEq)]
pub enum VQCType {
    /// Simple VQC with single-qubit rotations only
    Simple,
    /// Strongly entangling layers (similar to PennyLane's StronglyEntanglingLayers)
    StronglyEntangling,
    /// Basic entangling layers with alternating entanglement
    BasicEntangling,
    /// Custom VQC
    Custom,
}

/// Builder for Variational Quantum Circuits
#[derive(Clone, Debug)]
pub struct VQCBuilder {
    num_qubits: usize,
    num_layers: usize,
    vqc_type: VQCType,
    include_input_encoding: bool,
}

impl VQCBuilder {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            num_layers: 1,
            vqc_type: VQCType::BasicEntangling,
            include_input_encoding: true,
        }
    }

    pub fn with_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn with_type(mut self, vqc_type: VQCType) -> Self {
        self.vqc_type = vqc_type;
        self
    }

    pub fn with_input_encoding(mut self, include: bool) -> Self {
        self.include_input_encoding = include;
        self
    }

    pub fn build(self) -> VariationalQuantumCircuit {
        VariationalQuantumCircuit {
            num_qubits: self.num_qubits,
            num_layers: self.num_layers,
            vqc_type: self.vqc_type,
            include_input_encoding: self.include_input_encoding,
        }
    }
}

/// A Variational Quantum Circuit for use in quantum neural networks
///
/// This circuit structure follows the pattern from the QLSTM paper:
/// - Input encoding layer (optional)
/// - Parameterized rotation layers
/// - Entangling layers
#[derive(Clone, Debug)]
pub struct VariationalQuantumCircuit {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub vqc_type: VQCType,
    pub include_input_encoding: bool,
}

impl VariationalQuantumCircuit {
    /// Create a new VQC with default settings
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        Self {
            num_qubits,
            num_layers,
            vqc_type: VQCType::BasicEntangling,
            include_input_encoding: true,
        }
    }

    /// Get the number of parameters for input encoding
    pub fn num_input_params(&self) -> usize {
        if self.include_input_encoding {
            self.num_qubits
        } else {
            0
        }
    }

    /// Get the number of variational parameters (excluding inputs)
    pub fn num_variational_params(&self) -> usize {
        match self.vqc_type {
            VQCType::Simple => self.num_qubits * self.num_layers,
            VQCType::StronglyEntangling => 3 * self.num_qubits * self.num_layers,
            VQCType::BasicEntangling => 2 * self.num_qubits * self.num_layers,
            VQCType::Custom => 2 * self.num_qubits * self.num_layers,
        }
    }

    /// Build the circuit with given inputs and parameters
    ///
    /// # Arguments
    /// * `inputs` - Input data (for data encoding)
    /// * `params` - Variational parameters
    pub fn build_circuit(&self, inputs: &[f64], params: &[f64]) -> Circuit {
        let mut circuit = Circuit::new(self.num_qubits);

        // Input encoding layer
        if self.include_input_encoding {
            for (i, &input) in inputs.iter().enumerate().take(self.num_qubits) {
                circuit.ry(i, PI * input);
            }
        }

        // Variational layers
        let mut param_idx = 0;
        for layer in 0..self.num_layers {
            match self.vqc_type {
                VQCType::Simple => {
                    // Simple: only RY rotations
                    for qubit in 0..self.num_qubits {
                        circuit.ry(qubit, params[param_idx]);
                        param_idx += 1;
                    }
                }
                VQCType::StronglyEntangling => {
                    // Strongly entangling: RZ-RY-RZ on each qubit, then CNOT chain
                    for qubit in 0..self.num_qubits {
                        circuit.rz(qubit, params[param_idx]);
                        param_idx += 1;
                        circuit.ry(qubit, params[param_idx]);
                        param_idx += 1;
                        circuit.rz(qubit, params[param_idx]);
                        param_idx += 1;
                    }
                    // Entangling: CNOT chain with different patterns per layer
                    let offset = layer % self.num_qubits;
                    for i in 0..self.num_qubits.saturating_sub(1) {
                        let control = (i + offset) % self.num_qubits;
                        let target = (i + 1 + offset) % self.num_qubits;
                        if control != target {
                            circuit.cnot(control, target);
                        }
                    }
                }
                VQCType::BasicEntangling | VQCType::Custom => {
                    // Basic entangling: RY-RZ on each qubit, then CNOT ladder
                    for qubit in 0..self.num_qubits {
                        circuit.ry(qubit, params[param_idx]);
                        param_idx += 1;
                        circuit.rz(qubit, params[param_idx]);
                        param_idx += 1;
                    }
                    // Entangling layer
                    for i in 0..self.num_qubits.saturating_sub(1) {
                        circuit.cnot(i, i + 1);
                    }
                }
            }
        }

        circuit
    }

    /// Execute the VQC and return measurement expectation
    ///
    /// # Arguments
    /// * `inputs` - Input data for encoding
    /// * `params` - Variational parameters
    ///
    /// # Returns
    /// Vector of expectation values (one per qubit)
    pub fn forward(&self, inputs: &[f64], params: &[f64]) -> Vec<f64> {
        let circuit = self.build_circuit(inputs, params);
        let mut state = State::zero_state(self.num_qubits);

        circuit
            .execute(&mut state)
            .expect("Circuit execution failed");

        // Compute Z expectation for each qubit
        self.measure_expectations(&state)
    }

    /// Measure expectation values of Z for each qubit
    fn measure_expectations(&self, state: &State) -> Vec<f64> {
        let mut expectations = vec![0.0; self.num_qubits];

        let probs = state.probabilities();

        for (basis_idx, prob) in probs.iter().enumerate() {
            for qubit in 0..self.num_qubits {
                // Check if this qubit is |1⟩ in this basis state
                let bit = (basis_idx >> qubit) & 1;
                // Z eigenvalue: +1 for |0⟩, -1 for |1⟩
                let eigenvalue = if bit == 0 { 1.0 } else { -1.0 };
                expectations[qubit] += eigenvalue * prob;
            }
        }

        expectations
    }

    /// Compute the output with a final activation (tanh-like)
    ///
    /// Maps the Z expectations from [-1, 1] to a desired range
    pub fn forward_activated(&self, inputs: &[f64], params: &[f64]) -> Vec<f64> {
        self.forward(inputs, params)
    }
}

impl Ansatz for VariationalQuantumCircuit {
    fn build_circuit(&self, parameters: &[f64]) -> Circuit {
        // For Ansatz trait, we use empty inputs (parameters only)
        self.build_circuit(&[], parameters)
    }

    fn num_parameters(&self) -> usize {
        self.num_variational_params()
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// A dressed quantum circuit (VQC with input scaling and bias)
///
/// Output = W * VQC(inputs, params) + b
///
/// This provides additional classical expressivity around the quantum circuit.
#[derive(Clone, Debug)]
pub struct DressedVQC {
    pub vqc: VariationalQuantumCircuit,
    /// Output weights (one per output)
    pub output_weights: Vec<f64>,
    /// Output biases (one per output)
    pub output_biases: Vec<f64>,
}

impl DressedVQC {
    pub fn new(vqc: VariationalQuantumCircuit, output_size: usize) -> Self {
        Self {
            output_weights: vec![1.0; output_size],
            output_biases: vec![0.0; output_size],
            vqc,
        }
    }

    /// Forward pass with classical post-processing
    pub fn forward(&self, inputs: &[f64], params: &[f64]) -> Vec<f64> {
        let vqc_output = self.vqc.forward(inputs, params);

        vqc_output
            .iter()
            .zip(self.output_weights.iter())
            .zip(self.output_biases.iter())
            .map(|((v, w), b)| v * w + b)
            .collect()
    }

    /// Get total number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.vqc.num_variational_params() + self.output_weights.len() + self.output_biases.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqc_creation() {
        let vqc = VariationalQuantumCircuit::new(4, 2);
        assert_eq!(vqc.num_qubits, 4);
        assert_eq!(vqc.num_layers, 2);
    }

    #[test]
    fn test_vqc_builder() {
        let vqc = VQCBuilder::new(3)
            .with_layers(2)
            .with_type(VQCType::StronglyEntangling)
            .build();

        assert_eq!(vqc.num_qubits, 3);
        assert_eq!(vqc.num_layers, 2);
        assert_eq!(vqc.vqc_type, VQCType::StronglyEntangling);
    }

    #[test]
    fn test_vqc_forward() {
        let vqc = VariationalQuantumCircuit::new(2, 1);
        let inputs = vec![0.5, 0.3];
        let params = vec![0.1, 0.2, 0.3, 0.4]; // 2 * 2 params for BasicEntangling

        let output = vqc.forward(&inputs, &params);

        assert_eq!(output.len(), 2);
        // Outputs should be in [-1, 1] for Z expectations
        for &o in &output {
            assert!(o >= -1.0 && o <= 1.0);
        }
    }

    #[test]
    fn test_num_params() {
        let simple = VariationalQuantumCircuit {
            num_qubits: 3,
            num_layers: 2,
            vqc_type: VQCType::Simple,
            include_input_encoding: true,
        };
        assert_eq!(simple.num_variational_params(), 6); // 3 * 2

        let strong = VariationalQuantumCircuit {
            num_qubits: 3,
            num_layers: 2,
            vqc_type: VQCType::StronglyEntangling,
            include_input_encoding: true,
        };
        assert_eq!(strong.num_variational_params(), 18); // 3 * 3 * 2

        let basic = VariationalQuantumCircuit {
            num_qubits: 3,
            num_layers: 2,
            vqc_type: VQCType::BasicEntangling,
            include_input_encoding: true,
        };
        assert_eq!(basic.num_variational_params(), 12); // 2 * 3 * 2
    }
}
