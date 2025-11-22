// This file defines the noise module for simulating noise in quantum systems.
// It exports types and functions for adding noise to quantum operations.

use crate::states::State;
use ndarray::Array1;
use num_complex::Complex64;
use rand::Rng;

/// Trait for quantum noise models
pub trait NoiseModel {
    /// Apply noise to a quantum state
    fn apply(&self, state: &mut State);

    /// Return a descriptive name for this noise model
    fn name(&self) -> String;
}

/// Represents the common parameters for noise models
pub struct NoiseParams {
    /// The probability of an error occurring (0.0 to 1.0)
    pub error_probability: f64,

    /// Which qubits this noise applies to (None means all qubits)
    pub target_qubits: Option<Vec<usize>>,
}

impl NoiseParams {
    pub fn new(error_probability: f64) -> Self {
        Self {
            error_probability,
            target_qubits: None,
        }
    }

    pub fn with_target_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.target_qubits = Some(qubits);
        self
    }
}

/// Depolarizing noise model - replaces the state with a completely mixed state with probability p
pub struct DepolarizingNoise {
    params: NoiseParams,
}

impl DepolarizingNoise {
    pub fn new(error_probability: f64) -> Self {
        Self {
            params: NoiseParams::new(error_probability),
        }
    }

    pub fn with_target_qubits(self, qubits: Vec<usize>) -> Self {
        Self {
            params: self.params.with_target_qubits(qubits),
        }
    }
}

impl NoiseModel for DepolarizingNoise {
    fn apply(&self, state: &mut State) {
        let mut rng = rand::thread_rng();

        // Apply noise to specified qubits or all qubits
        let target_qubits = match &self.params.target_qubits {
            Some(qubits) => qubits.clone(),
            None => (0..state.num_qubits()).collect(),
        };

        for &qubit in &target_qubits {
            if rng.gen::<f64>() < self.params.error_probability {
                // Apply a random Pauli error (X, Y, or Z)
                let error_type = rng.gen_range(0..3);
                match error_type {
                    0 => apply_x_error(state, qubit), // X error
                    1 => apply_y_error(state, qubit), // Y error
                    2 => apply_z_error(state, qubit), // Z error
                    _ => unreachable!(),
                }
            }
        }
    }

    fn name(&self) -> String {
        format!("Depolarizing(p={})", self.params.error_probability)
    }
}

/// Amplitude damping noise - models energy dissipation (e.g., |1⟩ → |0⟩ transitions)
pub struct AmplitudeDampingNoise {
    params: NoiseParams,
}

impl AmplitudeDampingNoise {
    pub fn new(error_probability: f64) -> Self {
        Self {
            params: NoiseParams::new(error_probability),
        }
    }

    pub fn with_target_qubits(self, qubits: Vec<usize>) -> Self {
        Self {
            params: self.params.with_target_qubits(qubits),
        }
    }
}

impl NoiseModel for AmplitudeDampingNoise {
    fn apply(&self, state: &mut State) {
        let dimension = state.vector().len();

        // Apply noise to specified qubits or all qubits
        let target_qubits = match &self.params.target_qubits {
            Some(qubits) => qubits.clone(),
            None => (0..state.num_qubits()).collect(),
        };

        // Create a new vector to store the updated state
        let mut new_vector = Array1::<Complex64>::zeros(dimension);
        let state_vec = state.vector();

        for &qubit in &target_qubits {
            let gamma = self.params.error_probability;
            let sqrt_gamma = gamma.sqrt();

            // Apply the Kraus operators for amplitude damping
            // K0 = |0⟩⟨0| + sqrt(1-γ)|1⟩⟨1|
            // K1 = sqrt(γ)|0⟩⟨1|

            for i in 0..dimension {
                if (i & (1 << qubit)) == 0 {
                    // If qubit is |0⟩, apply K0 part |0⟩⟨0|
                    new_vector[i] += state_vec[i];
                } else {
                    // If qubit is |1⟩, apply K0 part sqrt(1-γ)|1⟩⟨1|
                    new_vector[i] += state_vec[i] * Complex64::new((1.0 - gamma).sqrt(), 0.0);

                    // Apply K1 = sqrt(γ)|0⟩⟨1|
                    let j = i & !(1 << qubit); // Flip qubit from |1⟩ to |0⟩
                    new_vector[j] += state_vec[i] * Complex64::new(sqrt_gamma, 0.0);
                }
            }
        }

        *state.vector_mut() = new_vector;
        state.normalize();
    }

    fn name(&self) -> String {
        format!("AmplitudeDamping(p={})", self.params.error_probability)
    }
}

/// Phase damping noise - causes phase decoherence without energy dissipation
pub struct PhaseDampingNoise {
    params: NoiseParams,
}

impl PhaseDampingNoise {
    pub fn new(error_probability: f64) -> Self {
        Self {
            params: NoiseParams::new(error_probability),
        }
    }

    pub fn with_target_qubits(self, qubits: Vec<usize>) -> Self {
        Self {
            params: self.params.with_target_qubits(qubits),
        }
    }
}

impl NoiseModel for PhaseDampingNoise {
    fn apply(&self, state: &mut State) {
        let dimension = state.vector().len();

        // Apply noise to specified qubits or all qubits
        let target_qubits = match &self.params.target_qubits {
            Some(qubits) => qubits.clone(),
            None => (0..state.num_qubits()).collect(),
        };

        // Create a new vector to store the updated state
        let mut new_vector = ndarray::Array1::zeros(dimension);
        let state_vec = state.vector();

        for &qubit in &target_qubits {
            // Phase damping Kraus operators:
            // K0 = |0⟩⟨0| + sqrt(1-λ)|1⟩⟨1|
            // K1 = sqrt(λ)|1⟩⟨1|

            let lambda = self.params.error_probability;
            let sqrt_1_minus_lambda = (1.0 - lambda).sqrt();

            for i in 0..dimension {
                // Copy original state
                new_vector[i] = state_vec[i];

                // If qubit is in state |1⟩, apply the phase damping
                if (i & (1 << qubit)) != 0 {
                    // The amplitude is reduced by sqrt(1-λ) but phase is preserved
                    new_vector[i] *= Complex64::new(sqrt_1_minus_lambda, 0.0);
                }
            }
        }

        // Update the state vector and normalize
        *state.vector_mut() = new_vector;
        state.normalize();
    }

    fn name(&self) -> String {
        format!("PhaseDamping(p={})", self.params.error_probability)
    }
}

/// Thermal relaxation noise model - combines amplitude and phase damping
pub struct ThermalRelaxationNoise {
    /// T1 relaxation time (amplitude damping)
    t1: f64,

    /// T2 relaxation time (phase damping)
    t2: f64,

    /// Gate time (how long the gate takes to execute)
    gate_time: f64,

    /// Target qubits
    target_qubits: Option<Vec<usize>>,
}

impl ThermalRelaxationNoise {
    pub fn new(t1: f64, t2: f64, gate_time: f64) -> Self {
        // Ensure t2 <= 2*t1 (physical constraint)
        let t2 = t2.min(2.0 * t1);

        Self {
            t1,
            t2,
            gate_time,
            target_qubits: None,
        }
    }

    pub fn with_target_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.target_qubits = Some(qubits);
        self
    }
}

impl NoiseModel for ThermalRelaxationNoise {
    fn apply(&self, state: &mut State) {
        // Calculate probabilities from relaxation times
        // For amplitude damping, p_a = 1 - exp(-gate_time/T1)
        let p_amplitude = 1.0 - (-self.gate_time / self.t1).exp();

        // For pure dephasing (not including relaxation-induced dephasing)
        // Pure dephasing rate is calculated as 1/T_phi = 1/T2 - 1/(2*T1)
        let pure_dephasing_rate = if self.t2 <= 2.0 * self.t1 {
            1.0 / self.t2 - 1.0 / (2.0 * self.t1)
        } else {
            // If T2 > 2*T1, which is physically impossible, default to 0
            0.0
        };

        // Phase damping probability p_p = 1 - exp(-gate_time/T_phi)
        let p_phase = if pure_dephasing_rate > 0.0 {
            1.0 - (-self.gate_time * pure_dephasing_rate).exp()
        } else {
            0.0
        };

        // Target qubits for noise application
        let target_qubits = self
            .target_qubits
            .clone()
            .unwrap_or_else(|| (0..state.num_qubits()).collect());

        // Apply amplitude damping first
        if p_amplitude > 0.0 {
            let amplitude_noise =
                AmplitudeDampingNoise::new(p_amplitude).with_target_qubits(target_qubits.clone());
            amplitude_noise.apply(state);
        }

        // Then apply phase damping
        if p_phase > 0.0 {
            let phase_noise = PhaseDampingNoise::new(p_phase).with_target_qubits(target_qubits);
            phase_noise.apply(state);
        }
    }

    fn name(&self) -> String {
        format!(
            "ThermalRelaxation(T1={}, T2={}, gate_time={})",
            self.t1, self.t2, self.gate_time
        )
    }
}

/// Composite noise model - combines multiple noise models
pub struct CompositeNoise {
    noise_models: Vec<Box<dyn NoiseModel>>,
}

impl CompositeNoise {
    pub fn new() -> Self {
        Self {
            noise_models: Vec::new(),
        }
    }

    pub fn add_noise<N: NoiseModel + 'static>(&mut self, noise_model: N) -> &mut Self {
        self.noise_models.push(Box::new(noise_model));
        self
    }
}

impl NoiseModel for CompositeNoise {
    fn apply(&self, state: &mut State) {
        // Apply each noise model in sequence
        for noise_model in &self.noise_models {
            // Create a copy of the state to avoid borrow issues
            let mut temp_state = state.clone();
            noise_model.apply(&mut temp_state);
            *state = temp_state;
        }
    }

    fn name(&self) -> String {
        let names: Vec<String> = self.noise_models.iter().map(|model| model.name()).collect();
        format!("Composite({})", names.join(", "))
    }
}

// Helper functions to apply specific Pauli errors
fn apply_x_error(state: &mut State, qubit: usize) {
    let dimension = state.vector().len();
    let state_vec = state.vector();
    let mut new_vector = state_vec.clone();

    for i in 0..dimension {
        let j = i ^ (1 << qubit); // Flip the qubit bit
        new_vector[i] = state_vec[j];
    }

    *state.vector_mut() = new_vector;
}

fn apply_y_error(state: &mut State, qubit: usize) {
    let dimension = state.vector().len();
    let state_vec = state.vector();
    let mut new_vector = state_vec.clone();
    let imag_i = Complex64::new(0.0, 1.0);

    for i in 0..dimension {
        let j = i ^ (1 << qubit); // Flip the qubit bit

        // Check if the qubit is 1 in state i
        if (i & (1 << qubit)) != 0 {
            new_vector[i] = -imag_i * state_vec[j];
        } else {
            new_vector[i] = imag_i * state_vec[j];
        }
    }

    *state.vector_mut() = new_vector;
}

fn apply_z_error(state: &mut State, qubit: usize) {
    let dimension = state.vector().len();
    let vector_slice = state.vector_mut().as_slice_mut().unwrap();

    for i in 0..dimension {
        // Apply phase flip if qubit is in state |1⟩
        if (i & (1 << qubit)) != 0 {
            vector_slice[i] = -vector_slice[i];
        }
    }
}
