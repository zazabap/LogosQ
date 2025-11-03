//! Classical optimizers for variational quantum algorithms
//!
//! This module provides various classical optimization algorithms.

use std::f64;

/// Result of an optimization run
#[derive(Debug, Clone)]
pub struct OptimizerResult {
    pub optimal_parameters: Vec<f64>,
    pub optimal_value: f64,
    pub num_iterations: usize,
    pub convergence_history: Vec<f64>,
    pub converged: bool,
}

/// Trait for classical optimizers
pub trait Optimizer {
    /// Minimize an objective function
    ///
    /// # Arguments
    /// * `f` - Objective function to minimize
    /// * `grad_f` - Gradient of objective function
    /// * `initial_params` - Starting parameters
    ///
    /// # Returns
    /// Optimization result with optimal parameters and value
    fn minimize<F, G>(&mut self, f: F, grad_f: G, initial_params: &[f64]) -> OptimizerResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>;
}

/// Gradient Descent optimizer
pub struct GradientDescent {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance: 1e-6,
        }
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Optimizer for GradientDescent {
    fn minimize<F, G>(&mut self, f: F, grad_f: G, initial_params: &[f64]) -> OptimizerResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut params = initial_params.to_vec();
        let mut convergence_history = Vec::new();
        let mut converged = false;

        for iteration in 0..self.max_iterations {
            let value = f(&params);
            convergence_history.push(value);

            let gradient = grad_f(&params);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Check convergence
            if grad_norm < self.tolerance {
                converged = true;
                break;
            }

            // Update parameters
            for i in 0..params.len() {
                params[i] -= self.learning_rate * gradient[i];
            }

            // Print progress every 10 iterations
            if iteration % 10 == 0 {
                println!(
                    "Iteration {}: value = {:.6}, |grad| = {:.6}",
                    iteration, value, grad_norm
                );
            }
        }

        let optimal_value = f(&params);
        // Final convergence check using gradient at the final parameters
        let final_grad = grad_f(&params);
        let final_grad_norm: f64 = final_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if final_grad_norm < self.tolerance {
            converged = true;
        }
        OptimizerResult {
            optimal_parameters: params,
            optimal_value,
            num_iterations: convergence_history.len(),
            convergence_history,
            converged,
        }
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Adam {
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iterations,
            tolerance: 1e-6,
        }
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Optimizer for Adam {
    fn minimize<F, G>(&mut self, f: F, grad_f: G, initial_params: &[f64]) -> OptimizerResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut params = initial_params.to_vec();
        let n = params.len();

        let mut m = vec![0.0; n]; // First moment
        let mut v = vec![0.0; n]; // Second moment
        let mut convergence_history = Vec::new();
        let mut converged = false;

        let mut prev_value: Option<f64> = None;
        for iteration in 1..=self.max_iterations {
            let value = f(&params);
            convergence_history.push(value);
            if let Some(pv) = prev_value {
                if (pv - value).abs() < self.tolerance {
                    converged = true;
                    break;
                }
            }
            prev_value = Some(value);

            let gradient = grad_f(&params);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Check convergence
            if grad_norm < self.tolerance {
                converged = true;
                break;
            }

            // Update biased first and second moment estimates
            for i in 0..n {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * gradient[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];
            }

            // Bias correction
            let m_hat_coef = 1.0 / (1.0 - self.beta1.powi(iteration as i32));
            let v_hat_coef = 1.0 / (1.0 - self.beta2.powi(iteration as i32));

            // Compute parameter step and update
            let mut step_norm_sq = 0.0;
            for i in 0..n {
                let m_hat = m[i] * m_hat_coef;
                let v_hat = v[i] * v_hat_coef;
                let step = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                params[i] -= step;
                step_norm_sq += step * step;
            }

            // Early stop if parameter updates are very small
            if step_norm_sq.sqrt() < self.tolerance {
                converged = true;
                break;
            }

            // Print progress
            if iteration % 10 == 0 {
                println!(
                    "Iteration {}: value = {:.6}, |grad| = {:.6}",
                    iteration, value, grad_norm
                );
            }
        }

        let optimal_value = f(&params);
        // Final convergence check using gradient at the final parameters
        let final_grad = grad_f(&params);
        let final_grad_norm: f64 = final_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if final_grad_norm < self.tolerance {
            converged = true;
        }

        OptimizerResult {
            optimal_parameters: params,
            optimal_value,
            num_iterations: convergence_history.len(),
            convergence_history,
            converged,
        }
    }
}

/// L-BFGS-B optimizer (for constrained optimization)
pub struct LBFGSB {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub m: usize, // Memory size for L-BFGS
}

impl LBFGSB {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            tolerance: 1e-6,
            m: 10,
        }
    }
}

// Note: Full L-BFGS-B implementation is complex and often uses external libraries
// For now, we provide the interface
