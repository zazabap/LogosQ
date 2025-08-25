use crate::gates::Gate;
use ndarray::Array1;
use num_complex::Complex64;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::f64::consts::SQRT_2;

/// Represents a quantum state vector using complex amplitudes.
pub struct State {
    pub vector: Array1<Complex64>,
    pub num_qubits: usize,
}

impl State {
    /// Creates a new quantum state from a complex vector (auto-normalized).
    pub fn new(vector: Array1<Complex64>, num_qubits: Option<usize>) -> Self {
        // Calculate num_qubits before moving vector
        let actual_qubits = num_qubits.unwrap_or_else(|| (vector.len() as f64).log2() as usize);

        let mut state = State {
            vector,
            num_qubits: actual_qubits,
        };
        state.normalize();
        state
    }

    /// Normalizes the state vector.
    pub fn normalize(&mut self) {
        let norm = self.vector.mapv(|x| x.norm_sqr()).sum().sqrt();
        if norm > 1e-10 {
            self.vector.mapv_inplace(|x| x / norm);
        }
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
    pub fn measure(&self) -> usize {
        let probs: Vec<f64> = self.vector.iter().map(|x| x.norm_sqr()).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }

    /// Measures a specific qubit, collapsing the state.
    /// Returns the measurement result (0 or 1).
    pub fn measure_qubit(&mut self, qubit_idx: usize) -> usize {
        assert!(qubit_idx < self.num_qubits, "Qubit index out of range");

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
        result
    }

    /// Calculates probability of measuring a particular basis state.
    pub fn probability(&self, basis_state: usize) -> f64 {
        if basis_state >= self.vector.len() {
            return 0.0;
        }
        self.vector[basis_state].norm_sqr()
    }

    /// Tensor product with another state.
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
}

impl Clone for State {
    fn clone(&self) -> Self {
        State {
            vector: self.vector.clone(),
            num_qubits: self.num_qubits,
        }
    }
}
