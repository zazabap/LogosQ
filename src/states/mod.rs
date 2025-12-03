//! Quantum state representation and operations
//!
//! This module provides the `State` struct for representing quantum state vectors
//! and operations for manipulating and measuring quantum states.

mod backend;

use crate::error::{LogosQError, Result};
use crate::gates::Gate;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::f64::consts::SQRT_2;

pub use backend::QuantumStateBackend;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Represents a quantum state vector using complex amplitudes.
///
/// For n qubits, the state vector has 2^n complex amplitudes.
/// The state is automatically normalized when created.
///
/// # Example
///
/// ```rust
/// use logosq::prelude::*;
///
/// // Create a 2-qubit zero state
/// let state = State::zero_state(2);
///
/// // Get the number of qubits
/// assert_eq!(state.num_qubits(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct State {
    vector: Array1<Complex64>,
    num_qubits: usize,
}

impl State {
    /// Creates a new quantum state from a complex vector (auto-normalized).
    ///
    /// # Arguments
    /// * `vector` - The state vector (must have length 2^n for n qubits)
    /// * `num_qubits` - Optional number of qubits (auto-calculated if None)
    ///
    /// # Returns
    /// A normalized quantum state
    ///
    /// # Errors
    /// Returns an error if the vector length is not a power of 2
    pub fn new(vector: Array1<Complex64>, num_qubits: Option<usize>) -> Result<Self> {
        let actual_qubits = num_qubits.unwrap_or_else(|| {
            let len = vector.len();
            (len as f64).log2() as usize
        });

        // Validate that vector length is a power of 2
        let expected_len = 1 << actual_qubits;
        if vector.len() != expected_len {
            return Err(LogosQError::InvalidStateDimension {
                dimension: vector.len(),
            });
        }

        let mut state = State {
            vector,
            num_qubits: actual_qubits,
        };
        state.normalize();
        Ok(state)
    }

    /// Creates a new quantum state from a complex vector without validation.
    ///
    /// # Safety
    /// This function does not validate that the vector length is a power of 2.
    /// Use with caution.
    pub fn new_unchecked(vector: Array1<Complex64>, num_qubits: usize) -> Self {
        let mut state = State { vector, num_qubits };
        state.normalize();
        state
    }

    /// Normalizes the state vector.
    ///
    /// Ensures that the sum of squared amplitudes equals 1.
    pub fn normalize(&mut self) {
        #[cfg(feature = "parallel")]
        {
            let norm = self
                .vector
                .par_iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt();

            if norm > 1e-10 {
                self.vector.par_mapv_inplace(|c| c / norm);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            let norm = self.vector.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

            if norm > 1e-10 {
                self.vector.mapv_inplace(|c| c / norm);
            }
        }
    }

    /// Returns the number of qubits in this state.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns a reference to the state vector.
    pub fn vector(&self) -> &Array1<Complex64> {
        &self.vector
    }

    /// Returns a mutable reference to the state vector.
    ///
    /// # Warning
    /// Modifying the vector directly may break normalization.
    /// Consider using `normalize()` after modifications.
    pub fn vector_mut(&mut self) -> &mut Array1<Complex64> {
        &mut self.vector
    }

    /// Creates standard basis state |0...0⟩
    pub fn zero_state(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut vector = Array1::zeros(size);
        vector[0] = Complex64::new(1.0, 0.0);
        Self { vector, num_qubits }
    }

    /// Creates standard basis state |1...1⟩
    pub fn one_state(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut vector = Array1::zeros(size);
        vector[size - 1] = Complex64::new(1.0, 0.0);
        Self { vector, num_qubits }
    }

    /// Creates |+⟩ state (equal superposition)
    pub fn plus_state(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let norm = 1.0 / (size as f64).sqrt();
        let vector = Array1::from_elem(size, Complex64::new(norm, 0.0));
        Self { vector, num_qubits }
    }

    /// Creates Bell state (maximally entangled two-qubit state)
    ///
    /// The Bell state is |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    pub fn bell_state() -> Self {
        let mut vector = Array1::zeros(4);
        vector[0] = Complex64::new(1.0 / SQRT_2, 0.0);
        vector[3] = Complex64::new(1.0 / SQRT_2, 0.0);
        Self {
            vector,
            num_qubits: 2,
        }
    }

    /// Applies a quantum gate to the state.
    pub fn apply_gate<G: Gate>(&mut self, gate: &G) {
        gate.apply(self);
    }

    /// Measures the entire state, returning the observed basis state index.
    ///
    /// This is a probabilistic measurement that collapses the state.
    pub fn measure(&self) -> usize {
        let probs: Vec<f64> = self.vector.iter().map(|x| x.norm_sqr()).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }

    /// Measures a specific qubit, collapsing the state.
    ///
    /// # Arguments
    /// * `qubit_idx` - Index of the qubit to measure (0-indexed)
    ///
    /// # Returns
    /// The measurement result (0 or 1)
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range
    pub fn measure_qubit(&mut self, qubit_idx: usize) -> Result<usize> {
        if qubit_idx >= self.num_qubits {
            return Err(LogosQError::InvalidQubitIndex {
                index: qubit_idx,
                num_qubits: self.num_qubits,
            });
        }

        let mask = 1 << qubit_idx;
        let mut prob_one = 0.0;

        // Calculate probability of measuring |1⟩
        for (i, amp) in self.vector.iter().enumerate() {
            if (i & mask) != 0 {
                prob_one += amp.norm_sqr();
            }
        }

        // Determine measurement outcome
        let result = if rand::random::<f64>() < prob_one {
            1
        } else {
            0
        };

        // Project state according to measurement
        let projection_value = if result == 1 { mask } else { 0 };
        let mut new_vector = Array1::zeros(self.vector.len());

        for (i, amp) in self.vector.iter().enumerate() {
            if (i & mask) == projection_value {
                new_vector[i] = *amp;
            }
        }

        self.vector = new_vector;
        self.normalize();
        Ok(result)
    }

    /// Parallel measurement for multiple shots.
    ///
    /// Returns a HashMap mapping basis state indices to measurement counts.
    ///
    /// # Arguments
    /// * `n_shots` - Number of measurement shots
    #[cfg(feature = "parallel")]
    pub fn measure_shots_parallel(
        &self,
        n_shots: usize,
    ) -> std::collections::HashMap<usize, usize> {
        let results: Vec<usize> = (0..n_shots)
            .into_par_iter()
            .map(|_| self.measure())
            .collect();

        let mut counts = std::collections::HashMap::new();
        for result in results {
            *counts.entry(result).or_insert(0) += 1;
        }
        counts
    }

    /// Sequential measurement for multiple shots (fallback when parallel is disabled).
    ///
    /// Returns a HashMap mapping basis state indices to measurement counts.
    ///
    /// # Arguments
    /// * `n_shots` - Number of measurement shots
    #[cfg(not(feature = "parallel"))]
    pub fn measure_shots(&self, n_shots: usize) -> std::collections::HashMap<usize, usize> {
        let mut counts = std::collections::HashMap::new();
        for _ in 0..n_shots {
            let result = self.measure();
            *counts.entry(result).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the probability distribution over all basis states.
    pub fn probabilities(&self) -> Vec<f64> {
        self.vector.mapv(|c| c.norm_sqr()).to_vec()
    }

    /// Calculates probability of measuring a particular basis state.
    ///
    /// # Arguments
    /// * `basis_state` - The basis state index
    ///
    /// # Returns
    /// The probability (0.0 if basis_state is out of range)
    pub fn probability(&self, basis_state: usize) -> f64 {
        if basis_state >= self.vector.len() {
            return 0.0;
        }
        self.vector[basis_state].norm_sqr()
    }

    /// Tensor product with another state.
    ///
    /// Creates a new state representing |self⟩ ⊗ |other⟩
    pub fn tensor_product(&self, other: &State) -> State {
        let n1 = self.vector.len();
        let n2 = other.vector.len();
        let mut result = Array1::zeros(n1 * n2);

        for i in 0..n1 {
            for j in 0..n2 {
                result[i * n2 + j] = self.vector[i] * other.vector[j];
            }
        }

        State {
            vector: result,
            num_qubits: self.num_qubits + other.num_qubits,
        }
    }

    /// Returns a string representation of the quantum state.
    ///
    /// Shows all basis states with non-negligible probability.
    pub fn print(&self) -> String {
        let mut output = format!(
            "State: {} qubit{}\n",
            self.num_qubits,
            if self.num_qubits != 1 { "s" } else { "" }
        );

        // Show states with non-negligible probability
        for (i, amplitude) in self.vector.iter().enumerate() {
            let probability = amplitude.norm_sqr();
            if probability > 1e-10 {
                let basis = format!("|{:0width$b}⟩", i, width = self.num_qubits);
                output.push_str(&format!(
                    "{} : {:.4}{:+.4}i (p={:.4})\n",
                    basis, amplitude.re, amplitude.im, probability
                ));
            }
        }

        output
    }

    /// Computes the inner product with another state.
    ///
    /// Returns ⟨self|other⟩ = Σᵢ self.vector[i]* · other.vector[i]
    ///
    /// # Arguments
    /// * `other` - The other state (must have same dimension)
    ///
    /// # Errors
    /// Returns an error if the states have different dimensions
    #[cfg(feature = "parallel")]
    pub fn inner_product_parallel(&self, other: &State) -> Result<Complex64> {
        use ndarray::Zip;

        if self.vector.len() != other.vector.len() {
            return Err(LogosQError::DimensionMismatch {
                expected: self.vector.len(),
                actual: other.vector.len(),
            });
        }

        Ok(Zip::from(&self.vector)
            .and(&other.vector)
            .par_map_collect(|a, b| a.conj() * b)
            .into_iter()
            .sum())
    }

    /// Computes the inner product with another state (sequential version).
    ///
    /// Returns ⟨self|other⟩ = Σᵢ self.vector[i]* · other.vector[i]
    ///
    /// # Arguments
    /// * `other` - The other state (must have same dimension)
    ///
    /// # Errors
    /// Returns an error if the states have different dimensions
    #[cfg(not(feature = "parallel"))]
    pub fn inner_product(&self, other: &State) -> Result<Complex64> {
        if self.vector.len() != other.vector.len() {
            return Err(LogosQError::DimensionMismatch {
                expected: self.vector.len(),
                actual: other.vector.len(),
            });
        }

        Ok(self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a.conj() * b)
            .sum())
    }
}

impl QuantumStateBackend for State {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply_single_qubit_matrix(
        &mut self,
        qubit: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(LogosQError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }

        // Expand single-qubit gate to full system dimension
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

        let mut full_matrix = if qubit == 0 {
            matrix.clone()
        } else {
            identity.clone()
        };

        // Tensor product with identity or gate for each remaining qubit
        for i in 1..self.num_qubits {
            if i == qubit {
                full_matrix = tensor_product_2d(&full_matrix, matrix);
            } else {
                full_matrix = tensor_product_2d(&full_matrix, &identity);
            }
        }

        self.apply_full_matrix(&full_matrix)
    }

    fn apply_two_qubit_matrix(
        &mut self,
        control: usize,
        target: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(LogosQError::InvalidQubitIndex {
                index: control.max(target),
                num_qubits: self.num_qubits,
            });
        }

        // Expand two-qubit gate to full system dimension
        let full_matrix = expand_two_qubit_gate(matrix, control, target, self.num_qubits)?;
        self.apply_full_matrix(&full_matrix)
    }

    fn apply_three_qubit_matrix(
        &mut self,
        _q1: usize,
        _q2: usize,
        _q3: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()> {
        // For three-qubit gates, use full matrix expansion
        // This is a simplified implementation - could be optimized
        self.apply_full_matrix(matrix)
    }

    fn apply_full_matrix(&mut self, matrix: &Array2<Complex64>) -> Result<()> {
        let state_vec = self.vector();
        let state_dim = state_vec.len();

        if matrix.shape()[1] != state_dim {
            return Err(LogosQError::DimensionMismatch {
                expected: state_dim,
                actual: matrix.shape()[1],
            });
        }

        let new_vector = matrix.dot(state_vec);
        *self.vector_mut() = new_vector;
        self.normalize();
        Ok(())
    }
}

// Helper function for tensor product of 2D matrices
fn tensor_product_2d(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows_a, cols_a) = a.dim();
    let (rows_b, cols_b) = b.dim();
    let mut result = Array2::zeros((rows_a * rows_b, cols_a * cols_b));

    for i in 0..rows_a {
        for j in 0..cols_a {
            for k in 0..rows_b {
                for l in 0..cols_b {
                    result[[i * rows_b + k, j * cols_b + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

// Helper to expand two-qubit gate to full system
fn expand_two_qubit_gate(
    gate: &Array2<Complex64>,
    control: usize,
    target: usize,
    num_qubits: usize,
) -> Result<Array2<Complex64>> {
    let dim = 1 << num_qubits;
    let mut full_matrix = Array2::zeros((dim, dim));

    // Build the full matrix by placing the gate at the correct positions
    for i in 0..dim {
        for j in 0..dim {
            // Extract the bits for control and target qubits
            let control_bit = (i >> (num_qubits - 1 - control)) & 1;
            let target_bit = (i >> (num_qubits - 1 - target)) & 1;
            let control_bit_j = (j >> (num_qubits - 1 - control)) & 1;
            let target_bit_j = (j >> (num_qubits - 1 - target)) & 1;

            // Check if this is in the subspace where the gate acts
            let gate_row = (control_bit << 1) | target_bit;
            let gate_col = (control_bit_j << 1) | target_bit_j;

            // Check if other qubits match
            let other_qubits_match = (0..num_qubits)
                .filter(|&q| q != control && q != target)
                .all(|q| {
                    let bit_i = (i >> (num_qubits - 1 - q)) & 1;
                    let bit_j = (j >> (num_qubits - 1 - q)) & 1;
                    bit_i == bit_j
                });

            if other_qubits_match {
                full_matrix[[i, j]] = gate[[gate_row, gate_col]];
            } else {
                full_matrix[[i, j]] = Complex64::new(0.0, 0.0);
            }
        }
    }

    Ok(full_matrix)
}
