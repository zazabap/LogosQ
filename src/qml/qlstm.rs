//! Quantum Long Short-Term Memory (QLSTM)
//!
//! Implementation based on Figure 5 from:
//! - "Quantum Long Short-Term Memory" (arXiv:2009.01783) by Chen et al.
//! - PennyLane's Learning to Learn with Quantum Neural Networks
//!
//! The QLSTM replaces classical LSTM gates with variational quantum circuits,
//! creating a hybrid quantum-classical recurrent neural network for sequence modeling.
//!
//! # Architecture (6 VQCs as per Figure 5)
//!
//! The QLSTM cell contains 6 VQCs:
//!
//! ## Gate VQCs (VQC₁-VQC₄)
//! - VQC₁ (Forget): f_t = σ(VQC₁([h_{t-1}, x_t]; θ₁))
//! - VQC₂ (Input):  i_t = σ(VQC₂([h_{t-1}, x_t]; θ₂))
//! - VQC₃ (Cell):   c̃_t = tanh(VQC₃([h_{t-1}, x_t]; θ₃))
//! - VQC₄ (Output): o_t = σ(VQC₄([h_{t-1}, x_t]; θ₄))
//!
//! ## Cell state update (classical)
//! - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
//!
//! ## Hidden state processing (VQC₅)
//! - h_t = VQC₅(o_t ⊙ tanh(c_t); θ₅)
//!
//! ## Output processing (VQC₆)
//! - y_t = VQC₆(h_t; θ₆)

use super::vqc::{VQCType, VariationalQuantumCircuit};
use std::f64::consts::PI;

/// Configuration for QLSTM
#[derive(Clone, Debug)]
pub struct QLSTMConfig {
    /// Number of qubits used for each VQC gate
    pub num_qubits: usize,
    /// Number of VQC layers per gate
    pub num_layers: usize,
    /// Input feature dimension
    pub input_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Type of VQC to use
    pub vqc_type: VQCType,
    /// Whether to use classical pre-processing (data re-uploading)
    pub use_classical_preprocessing: bool,
}

impl Default for QLSTMConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            num_layers: 2,
            input_size: 1,
            hidden_size: 4,
            vqc_type: VQCType::BasicEntangling,
            use_classical_preprocessing: true,
        }
    }
}

impl QLSTMConfig {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // Number of qubits should accommodate input + hidden state
        let num_qubits = (input_size + hidden_size).max(2);
        Self {
            num_qubits,
            num_layers: 2,
            input_size,
            hidden_size,
            vqc_type: VQCType::BasicEntangling,
            use_classical_preprocessing: true,
        }
    }

    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn with_num_qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = num_qubits;
        self
    }

    pub fn with_vqc_type(mut self, vqc_type: VQCType) -> Self {
        self.vqc_type = vqc_type;
        self
    }
}

/// Output of a QLSTM cell
#[derive(Clone, Debug)]
pub struct QLSTMOutput {
    /// Hidden state h_t
    pub hidden_state: Vec<f64>,
    /// Cell state c_t
    pub cell_state: Vec<f64>,
    /// Raw output before any post-processing
    pub output: Vec<f64>,
}

/// A single QLSTM cell
///
/// The cell contains six VQCs as shown in Figure 5 of arXiv:2009.01783:
/// - VQC₁: Forget gate VQC
/// - VQC₂: Input gate VQC
/// - VQC₃: Cell gate VQC
/// - VQC₄: Output gate VQC
/// - VQC₅: Hidden state processing VQC (processes o_t ⊙ tanh(c_t) → h_t)
/// - VQC₆: Output processing VQC (processes h_t → y_t)
#[derive(Clone, Debug)]
pub struct QLSTMCell {
    pub config: QLSTMConfig,
    /// VQC₁: Forget gate
    vqc_forget: VariationalQuantumCircuit,
    /// VQC₂: Input gate
    vqc_input: VariationalQuantumCircuit,
    /// VQC₃: Cell gate
    vqc_cell: VariationalQuantumCircuit,
    /// VQC₄: Output gate
    vqc_output: VariationalQuantumCircuit,
    /// VQC₅: Hidden state processing
    vqc_hidden: VariationalQuantumCircuit,
    /// VQC₆: Output processing (for y_t)
    vqc_output_proc: VariationalQuantumCircuit,
}

impl QLSTMCell {
    /// Create a new QLSTM cell with 6 VQCs as per Figure 5 of arXiv:2009.01783
    pub fn new(config: QLSTMConfig) -> Self {
        // VQC₁: Forget gate
        let vqc_forget = VariationalQuantumCircuit {
            num_qubits: config.num_qubits,
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };
        // VQC₂: Input gate
        let vqc_input = VariationalQuantumCircuit {
            num_qubits: config.num_qubits,
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };
        // VQC₃: Cell gate
        let vqc_cell = VariationalQuantumCircuit {
            num_qubits: config.num_qubits,
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };
        // VQC₄: Output gate
        let vqc_output = VariationalQuantumCircuit {
            num_qubits: config.num_qubits,
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };
        // VQC₅: Hidden state processing (uses hidden_size qubits)
        let vqc_hidden = VariationalQuantumCircuit {
            num_qubits: config.hidden_size.max(2),
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };
        // VQC₆: Output processing (uses hidden_size qubits)
        let vqc_output_proc = VariationalQuantumCircuit {
            num_qubits: config.hidden_size.max(2),
            num_layers: config.num_layers,
            vqc_type: config.vqc_type.clone(),
            include_input_encoding: true,
        };

        Self {
            config,
            vqc_forget,
            vqc_input,
            vqc_cell,
            vqc_output,
            vqc_hidden,
            vqc_output_proc,
        }
    }

    /// Get the number of parameters for a single gate VQC (VQC₁-VQC₄)
    pub fn params_per_gate(&self) -> usize {
        self.vqc_forget.num_variational_params()
    }

    /// Get the number of parameters for VQC₅ (hidden state processing)
    pub fn params_vqc_hidden(&self) -> usize {
        self.vqc_hidden.num_variational_params()
    }

    /// Get the number of parameters for VQC₆ (output processing)
    pub fn params_vqc_output_proc(&self) -> usize {
        self.vqc_output_proc.num_variational_params()
    }

    /// Get total number of parameters (all six VQCs)
    pub fn num_parameters(&self) -> usize {
        4 * self.params_per_gate() + self.params_vqc_hidden() + self.params_vqc_output_proc()
    }

    /// Split parameters into six VQC parameter sets
    #[allow(clippy::type_complexity)]
    fn split_params<'a>(
        &self,
        params: &'a [f64],
    ) -> (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
    ) {
        let n_gate = self.params_per_gate();
        let n_hidden = self.params_vqc_hidden();
        let n_output_proc = self.params_vqc_output_proc();

        let params_f = &params[0..n_gate];
        let params_i = &params[n_gate..2 * n_gate];
        let params_c = &params[2 * n_gate..3 * n_gate];
        let params_o = &params[3 * n_gate..4 * n_gate];
        let params_h = &params[4 * n_gate..4 * n_gate + n_hidden];
        let params_y = &params[4 * n_gate + n_hidden..4 * n_gate + n_hidden + n_output_proc];

        (params_f, params_i, params_c, params_o, params_h, params_y)
    }

    /// Prepare input for VQC: concatenate input and hidden state
    fn prepare_vqc_input(&self, input: &[f64], hidden: &[f64]) -> Vec<f64> {
        let mut combined = Vec::with_capacity(self.config.num_qubits);

        // Add input features
        for &x in input.iter().take(self.config.input_size) {
            combined.push(x);
        }

        // Add hidden state
        for &h in hidden.iter().take(self.config.hidden_size) {
            combined.push(h);
        }

        // Pad with zeros if needed
        while combined.len() < self.config.num_qubits {
            combined.push(0.0);
        }

        // Truncate if too long
        combined.truncate(self.config.num_qubits);

        combined
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply sigmoid to a vector
    fn sigmoid_vec(v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| Self::sigmoid(x)).collect()
    }

    /// Apply tanh to a vector
    fn tanh_vec(v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| x.tanh()).collect()
    }

    /// Element-wise multiplication
    fn hadamard_product(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Element-wise addition
    fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Forward pass through the QLSTM cell
    ///
    /// Implements the full 6-VQC architecture from Figure 5 of arXiv:2009.01783:
    /// - VQC₁-VQC₄: Compute gate values (forget, input, cell, output)
    /// - VQC₅: Process o_t ⊙ tanh(c_t) → h_t (hidden state)
    /// - VQC₆: Process h_t → y_t (output)
    ///
    /// # Arguments
    /// * `input` - Input vector x_t
    /// * `hidden_state` - Previous hidden state h_{t-1}
    /// * `cell_state` - Previous cell state c_{t-1}
    /// * `params` - All VQC parameters (concatenated: VQC₁, VQC₂, VQC₃, VQC₄, VQC₅, VQC₆)
    ///
    /// # Returns
    /// New hidden state h_t, cell state c_t, and output y_t
    pub fn forward(
        &self,
        input: &[f64],
        hidden_state: &[f64],
        cell_state: &[f64],
        params: &[f64],
    ) -> QLSTMOutput {
        assert_eq!(
            params.len(),
            self.num_parameters(),
            "Parameter count mismatch: expected {}, got {}",
            self.num_parameters(),
            params.len()
        );

        // Split parameters for all 6 VQCs
        let (params_f, params_i, params_c, params_o, params_h, params_y) = self.split_params(params);

        // Prepare input for VQC₁-VQC₄: concatenate [x_t, h_{t-1}]
        let vqc_input_data = self.prepare_vqc_input(input, hidden_state);

        // ============ VQC₁-VQC₄: Compute gate values ============
        // The VQC outputs are in [-1, 1] (Z expectation values)
        let f_raw = self.vqc_forget.forward(&vqc_input_data, params_f);
        let i_raw = self.vqc_input.forward(&vqc_input_data, params_i);
        let c_raw = self.vqc_cell.forward(&vqc_input_data, params_c);
        let o_raw = self.vqc_output.forward(&vqc_input_data, params_o);

        // Apply activations
        // f_t = σ(VQC₁)
        let f_t = Self::sigmoid_vec(&f_raw);
        // i_t = σ(VQC₂)
        let i_t = Self::sigmoid_vec(&i_raw);
        // c̃_t = tanh(VQC₃)
        let c_tilde = Self::tanh_vec(&c_raw);
        // o_t = σ(VQC₄)
        let o_t = Self::sigmoid_vec(&o_raw);

        // Ensure vectors have correct size (hidden_size)
        let f_t: Vec<f64> = f_t.into_iter().take(self.config.hidden_size).collect();
        let i_t: Vec<f64> = i_t.into_iter().take(self.config.hidden_size).collect();
        let c_tilde: Vec<f64> = c_tilde.into_iter().take(self.config.hidden_size).collect();
        let o_t: Vec<f64> = o_t.into_iter().take(self.config.hidden_size).collect();

        // Ensure cell_state has correct size
        let cell_state: Vec<f64> = cell_state
            .iter()
            .take(self.config.hidden_size)
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(self.config.hidden_size)
            .collect();

        // ============ Cell state update (classical) ============
        // c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        let forget_contribution = Self::hadamard_product(&f_t, &cell_state);
        let input_contribution = Self::hadamard_product(&i_t, &c_tilde);
        let c_t = Self::add_vec(&forget_contribution, &input_contribution);

        // ============ Intermediate: o_t ⊙ tanh(c_t) ============
        let tanh_c_t = Self::tanh_vec(&c_t);
        let pre_hidden = Self::hadamard_product(&o_t, &tanh_c_t);

        // ============ VQC₅: Hidden state processing ============
        // h_t = VQC₅(o_t ⊙ tanh(c_t))
        let h_t_raw = self.vqc_hidden.forward(&pre_hidden, params_h);
        let h_t: Vec<f64> = h_t_raw.into_iter().take(self.config.hidden_size).collect();

        // ============ VQC₆: Output processing ============
        // y_t = VQC₆(h_t)
        let y_t_raw = self.vqc_output_proc.forward(&h_t, params_y);
        let y_t: Vec<f64> = y_t_raw.into_iter().take(self.config.hidden_size).collect();

        QLSTMOutput {
            hidden_state: h_t,
            cell_state: c_t,
            output: y_t,
        }
    }
}

/// Full QLSTM layer for sequence processing
///
/// Processes a sequence of inputs through one or more QLSTM cells.
#[derive(Clone, Debug)]
pub struct QLSTM {
    /// The QLSTM cell
    cell: QLSTMCell,
    /// Whether to return sequences or just the final output
    return_sequences: bool,
    /// Number of LSTM layers (stacked)
    num_layers: usize,
}

impl QLSTM {
    /// Create a new QLSTM layer
    pub fn new(config: QLSTMConfig) -> Self {
        Self {
            cell: QLSTMCell::new(config),
            return_sequences: false,
            num_layers: 1,
        }
    }

    /// Create QLSTM with specified return behavior
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Create stacked QLSTM
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &QLSTMConfig {
        &self.cell.config
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.cell.num_parameters() * self.num_layers
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.cell.config.hidden_size
    }

    /// Initialize hidden and cell states
    pub fn init_states(&self) -> (Vec<f64>, Vec<f64>) {
        let hidden = vec![0.0; self.cell.config.hidden_size];
        let cell = vec![0.0; self.cell.config.hidden_size];
        (hidden, cell)
    }

    /// Process a single time step through the first layer only
    ///
    /// Note: For stacked QLSTM (num_layers > 1), this method only processes
    /// through the first layer using the first layer's parameters. Use `forward()`
    /// for full stacked processing.
    pub fn step(
        &self,
        input: &[f64],
        hidden_state: &[f64],
        cell_state: &[f64],
        params: &[f64],
    ) -> QLSTMOutput {
        let params_per_layer = self.cell.num_parameters();
        // Use only the first layer's parameters
        let layer_params = if params.len() >= params_per_layer {
            &params[..params_per_layer]
        } else {
            params
        };
        self.cell.forward(input, hidden_state, cell_state, layer_params)
    }

    /// Process an entire sequence
    ///
    /// For stacked QLSTM (num_layers > 1), each layer processes the entire sequence,
    /// and its output becomes the input to the next layer.
    ///
    /// # Arguments
    /// * `sequence` - Vector of input vectors, one per time step
    /// * `params` - VQC parameters (concatenated for all layers)
    /// * `initial_hidden` - Optional initial hidden state (for first layer only)
    /// * `initial_cell` - Optional initial cell state (for first layer only)
    ///
    /// # Returns
    /// If return_sequences is true: outputs (y_t from VQC₆) at each time step
    /// If return_sequences is false: output (y_t from VQC₆) at the final time step only
    ///
    /// Note: In both cases, returns the processed output (y_t), not the hidden state (h_t).
    /// Use `forward_with_state()` if you need access to the final hidden/cell states.
    pub fn forward(
        &self,
        sequence: &[Vec<f64>],
        params: &[f64],
        initial_hidden: Option<&[f64]>,
        initial_cell: Option<&[f64]>,
    ) -> Vec<Vec<f64>> {
        let (outputs, _, _) = self.forward_with_state(sequence, params, initial_hidden, initial_cell);
        outputs
    }

    /// Process an entire sequence and return final states
    ///
    /// This is like `forward()` but also returns the final hidden and cell states,
    /// which is useful for sequence-to-sequence models or when you need to continue
    /// processing from where you left off.
    ///
    /// # Arguments
    /// * `sequence` - Vector of input vectors, one per time step
    /// * `params` - VQC parameters (concatenated for all layers)
    /// * `initial_hidden` - Optional initial hidden state (for first layer only)
    /// * `initial_cell` - Optional initial cell state (for first layer only)
    ///
    /// # Returns
    /// Tuple of (outputs, final_hidden_state, final_cell_state)
    /// - outputs: If return_sequences is true, all y_t; otherwise just the final y_t
    /// - final_hidden_state: The h_t from the final layer's last timestep
    /// - final_cell_state: The c_t from the final layer's last timestep
    pub fn forward_with_state(
        &self,
        sequence: &[Vec<f64>],
        params: &[f64],
        initial_hidden: Option<&[f64]>,
        initial_cell: Option<&[f64]>,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let params_per_layer = self.cell.num_parameters();

        // Current sequence to process (starts as input sequence)
        let mut current_sequence: Vec<Vec<f64>> = sequence.to_vec();

        // Final states (will be updated by each layer)
        let mut final_h_t = vec![0.0; self.cell.config.hidden_size];
        let mut final_c_t = vec![0.0; self.cell.config.hidden_size];

        for layer_idx in 0..self.num_layers {
            // Get parameters for this layer
            let layer_params =
                &params[layer_idx * params_per_layer..(layer_idx + 1) * params_per_layer];

            // Initialize states for this layer
            let (h0, c0) = self.init_states();
            let mut h_t = if layer_idx == 0 {
                initial_hidden.unwrap_or(&h0).to_vec()
            } else {
                h0
            };
            let mut c_t = if layer_idx == 0 {
                initial_cell.unwrap_or(&c0).to_vec()
            } else {
                c0
            };

            let mut layer_outputs = Vec::new();

            // Process each time step through this layer
            for input in &current_sequence {
                let output = self.cell.forward(input, &h_t, &c_t, layer_params);
                h_t = output.hidden_state;
                c_t = output.cell_state;
                layer_outputs.push(output.output);
            }

            // Store final states from this layer
            final_h_t = h_t;
            final_c_t = c_t;

            // Next layer takes this layer's outputs as its inputs
            current_sequence = layer_outputs;
        }

        let outputs = if self.return_sequences {
            current_sequence
        } else {
            // Return the final output (y_t), not the hidden state (h_t)
            // This ensures consistent semantics: forward() always returns outputs
            vec![current_sequence.pop().unwrap_or_else(|| vec![0.0; self.cell.config.hidden_size])]
        };

        (outputs, final_h_t, final_c_t)
    }

    /// Compute gradient using parameter-shift rule
    ///
    /// This computes the gradient of a loss function with respect to VQC parameters.
    ///
    /// # Arguments
    /// * `sequence` - Input sequence
    /// * `target` - Target output
    /// * `params` - Current parameters
    /// * `loss_fn` - Loss function that takes (output, target) and returns a scalar
    pub fn compute_gradient<F>(
        &self,
        sequence: &[Vec<f64>],
        target: &[f64],
        params: &[f64],
        loss_fn: F,
    ) -> Vec<f64>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let num_params = params.len();
        let mut gradient = vec![0.0; num_params];
        let shift = PI / 2.0;

        for i in 0..num_params {
            // Forward shift
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;
            let output_plus = self.forward(sequence, &params_plus, None, None);
            let loss_plus = loss_fn(&output_plus[0], target);

            // Backward shift
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;
            let output_minus = self.forward(sequence, &params_minus, None, None);
            let loss_minus = loss_fn(&output_minus[0], target);

            // Parameter-shift rule
            gradient[i] = (loss_plus - loss_minus) / 2.0;
        }

        gradient
    }
}

/// Mean squared error loss
pub fn mse_loss(output: &[f64], target: &[f64]) -> f64 {
    let n = output.len().min(target.len());
    if n == 0 {
        return 0.0;
    }

    output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| (o - t).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Cross-entropy loss (for classification)
pub fn cross_entropy_loss(output: &[f64], target: &[f64]) -> f64 {
    let epsilon = 1e-10;
    let n = output.len().min(target.len());
    if n == 0 {
        return 0.0;
    }

    -output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| {
            let o_clamped = o.clamp(epsilon, 1.0 - epsilon);
            t * o_clamped.ln() + (1.0 - t) * (1.0 - o_clamped).ln()
        })
        .sum::<f64>()
        / n as f64
}

/// A simple training helper for QLSTM
pub struct QLSTMTrainer {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub verbose: bool,
}

impl Default for QLSTMTrainer {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
        }
    }
}

impl QLSTMTrainer {
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance: 1e-6,
            verbose: true,
        }
    }

    /// Train the QLSTM on a single sequence
    ///
    /// # Arguments
    /// * `qlstm` - The QLSTM model
    /// * `sequence` - Training sequence
    /// * `target` - Target output
    /// * `initial_params` - Initial parameters
    ///
    /// # Returns
    /// Tuple of (best parameters, best loss) - the parameters that achieved the lowest loss
    pub fn train(
        &self,
        qlstm: &QLSTM,
        sequence: &[Vec<f64>],
        target: &[f64],
        initial_params: &[f64],
    ) -> (Vec<f64>, f64) {
        let mut params = initial_params.to_vec();
        let mut best_loss = f64::INFINITY;
        let mut best_params = params.clone();

        for iteration in 0..self.max_iterations {
            // Forward pass
            let output = qlstm.forward(sequence, &params, None, None);
            let loss = mse_loss(&output[0], target);

            if loss < best_loss {
                best_loss = loss;
                best_params = params.clone();
            }

            // Check convergence
            if loss < self.tolerance {
                if self.verbose {
                    println!("Converged at iteration {}: loss = {:.6}", iteration, loss);
                }
                break;
            }

            // Compute gradient
            let gradient = qlstm.compute_gradient(sequence, target, &params, mse_loss);

            // Update parameters (gradient descent)
            for (p, g) in params.iter_mut().zip(gradient.iter()) {
                *p -= self.learning_rate * g;
            }

            if self.verbose && iteration % 10 == 0 {
                println!("Iteration {}: loss = {:.6}", iteration, loss);
            }
        }

        (best_params, best_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlstm_cell_creation() {
        let config = QLSTMConfig::new(2, 4);
        let cell = QLSTMCell::new(config);

        assert_eq!(cell.config.input_size, 2);
        assert_eq!(cell.config.hidden_size, 4);
    }

    #[test]
    fn test_qlstm_cell_forward() {
        let config = QLSTMConfig::new(2, 4).with_num_layers(1);
        let cell = QLSTMCell::new(config);

        let num_params = cell.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

        let input = vec![0.5, 0.3];
        let hidden = vec![0.0; 4];
        let cell_state = vec![0.0; 4];

        let output = cell.forward(&input, &hidden, &cell_state, &params);

        assert_eq!(output.hidden_state.len(), 4);
        assert_eq!(output.cell_state.len(), 4);
        assert_eq!(output.output.len(), 4);
    }

    #[test]
    fn test_qlstm_sequence() {
        let config = QLSTMConfig::new(1, 2).with_num_layers(1);
        let qlstm = QLSTM::new(config);

        let num_params = qlstm.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

        let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

        let output = qlstm.forward(&sequence, &params, None, None);

        // Without return_sequences, we get only the final state
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2);
    }

    #[test]
    fn test_qlstm_return_sequences() {
        let config = QLSTMConfig::new(1, 2).with_num_layers(1);
        let qlstm = QLSTM::new(config).with_return_sequences(true);

        let num_params = qlstm.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

        let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

        let output = qlstm.forward(&sequence, &params, None, None);

        // With return_sequences, we get output at each time step
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_mse_loss() {
        let output = vec![0.5, 0.5];
        let target = vec![1.0, 0.0];

        let loss = mse_loss(&output, &target);
        // MSE = ((0.5-1)^2 + (0.5-0)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        assert!((loss - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_qlstm_gradient() {
        let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(3);
        let qlstm = QLSTM::new(config);

        let num_params = qlstm.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.1).collect();

        let sequence = vec![vec![0.5]];
        let target = vec![0.3, 0.7];

        let gradient = qlstm.compute_gradient(&sequence, &target, &params, mse_loss);

        assert_eq!(gradient.len(), num_params);
        // Gradients should be finite
        for g in &gradient {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_stacked_qlstm() {
        // Test QLSTM with num_layers > 1 (stacked layers)
        let config = QLSTMConfig::new(1, 2).with_num_layers(1);
        let qlstm = QLSTM::new(config).with_num_layers(2); // 2 stacked layers

        let num_params = qlstm.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

        let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];

        // This should work without panicking
        let output = qlstm.forward(&sequence, &params, None, None);

        // Should return final hidden state
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2); // hidden_size = 2
    }

    #[test]
    fn test_stacked_qlstm_return_sequences() {
        // Test stacked QLSTM with return_sequences = true
        let config = QLSTMConfig::new(1, 2).with_num_layers(1);
        let qlstm = QLSTM::new(config)
            .with_num_layers(2)
            .with_return_sequences(true);

        let num_params = qlstm.num_parameters();
        let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

        let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

        let output = qlstm.forward(&sequence, &params, None, None);

        // With return_sequences, we get output at each time step
        assert_eq!(output.len(), 4);
    }
}
