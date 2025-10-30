//! Gradient computation methods for variational quantum algorithms
//!
//! This module provides different methods for computing gradients of expectation values.

use super::ansatz::Ansatz;
use super::observable::Observable;
use crate::states::State;

/// Trait for gradient computation methods
pub trait GradientMethod {
    /// Compute gradient of expectation value with respect to parameters
    fn compute_gradient<A, O>(&self, ansatz: &A, observable: &O, parameters: &[f64]) -> Vec<f64>
    where
        A: Ansatz,
        O: Observable;
}

/// Parameter-shift rule for computing gradients
///
/// Uses the parameter-shift rule: ∂⟨H⟩/∂θᵢ = (⟨H⟩₊ - ⟨H⟩₋) / 2
/// where ⟨H⟩₊ is evaluated at θᵢ + π/2 and ⟨H⟩₋ at θᵢ - π/2
pub struct ParameterShift {
    pub shift: f64,
}

impl ParameterShift {
    pub fn new() -> Self {
        Self {
            shift: std::f64::consts::PI / 2.0,
        }
    }

    pub fn with_shift(shift: f64) -> Self {
        Self { shift }
    }
}

impl Default for ParameterShift {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientMethod for ParameterShift {
    fn compute_gradient<A, O>(&self, ansatz: &A, observable: &O, parameters: &[f64]) -> Vec<f64>
    where
        A: Ansatz,
        O: Observable,
    {
        let num_params = parameters.len();
        let mut gradient = vec![0.0; num_params];

        for i in 0..num_params {
            // Forward shift
            let mut params_plus = parameters.to_vec();
            params_plus[i] += self.shift;

            let mut state_plus = State::zero_state(ansatz.num_qubits());
            ansatz.apply(&mut state_plus, &params_plus);
            let expectation_plus = observable.expectation(&state_plus);

            // Backward shift
            let mut params_minus = parameters.to_vec();
            params_minus[i] -= self.shift;

            let mut state_minus = State::zero_state(ansatz.num_qubits());
            ansatz.apply(&mut state_minus, &params_minus);
            let expectation_minus = observable.expectation(&state_minus);

            // Gradient via parameter-shift rule
            // For standard rotation generators, the correct prefactor is 1/2
            // i.e., ∂⟨H⟩/∂θᵢ = (⟨H⟩(θᵢ + s) - ⟨H⟩(θᵢ - s)) / 2
            gradient[i] = (expectation_plus - expectation_minus) / 2.0;
        }

        gradient
    }
}

/// Finite difference gradient computation (for testing/debugging)
pub struct FiniteDifference {
    pub epsilon: f64,
}

impl FiniteDifference {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for FiniteDifference {
    fn default() -> Self {
        Self::new(1e-7)
    }
}

impl GradientMethod for FiniteDifference {
    fn compute_gradient<A, O>(&self, ansatz: &A, observable: &O, parameters: &[f64]) -> Vec<f64>
    where
        A: Ansatz,
        O: Observable,
    {
        let num_params = parameters.len();
        let mut gradient = vec![0.0; num_params];

        // Compute f(x)
        let mut state = State::zero_state(ansatz.num_qubits());
        ansatz.apply(&mut state, parameters);
        let f_x = observable.expectation(&state);

        for i in 0..num_params {
            // Compute f(x + εeᵢ)
            let mut params_plus = parameters.to_vec();
            params_plus[i] += self.epsilon;

            let mut state_plus = State::zero_state(ansatz.num_qubits());
            ansatz.apply(&mut state_plus, &params_plus);
            let f_x_plus = observable.expectation(&state_plus);

            // Forward difference
            gradient[i] = (f_x_plus - f_x) / self.epsilon;
        }

        gradient
    }
}
