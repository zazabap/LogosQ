//! Variational Quantum Eigensolver (VQE)
//!
//! This module implements the VQE algorithm for finding ground state energies
//! of quantum Hamiltonians.

use super::ansatz::Ansatz;
use super::gradient::GradientMethod;
use super::observable::Observable;
use super::optimizer::Optimizer;
use crate::states::State;

/// VQE result
#[derive(Debug, Clone)]
pub struct VQEResult {
    pub ground_state_energy: f64,
    pub optimal_parameters: Vec<f64>,
    pub num_iterations: usize,
    pub convergence_history: Vec<f64>,
    pub final_state: Option<State>,
}

/// Variational Quantum Eigensolver
pub struct VQE<A, O, G, Opt>
where
    A: Ansatz,
    O: Observable,
    G: GradientMethod,
    Opt: Optimizer,
{
    pub ansatz: A,
    pub hamiltonian: O,
    pub gradient_method: G,
    pub optimizer: Opt,
    pub verbose: bool,
}

impl<A, O, G, Opt> VQE<A, O, G, Opt>
where
    A: Ansatz,
    O: Observable,
    G: GradientMethod,
    Opt: Optimizer,
{
    pub fn new(ansatz: A, hamiltonian: O, gradient_method: G, optimizer: Opt) -> Self {
        Self {
            ansatz,
            hamiltonian,
            gradient_method,
            optimizer,
            verbose: true,
        }
    }

    /// Run VQE optimization
    pub fn run(&mut self, initial_parameters: &[f64]) -> VQEResult {
        if self.verbose {
            println!("Starting VQE optimization...");
            println!("Number of qubits: {}", self.ansatz.num_qubits());
            println!("Number of parameters: {}", self.ansatz.num_parameters());
        }

        // Define objective function (energy expectation)
        let objective = |params: &[f64]| -> f64 {
            let mut state = State::zero_state(self.ansatz.num_qubits());
            self.ansatz.apply(&mut state, params);
            self.hamiltonian.expectation(&state)
        };

        // Define gradient function
        let gradient = |params: &[f64]| -> Vec<f64> {
            self.gradient_method
                .compute_gradient(&self.ansatz, &self.hamiltonian, params)
        };

        // Run optimization
        let opt_result = self
            .optimizer
            .minimize(objective, gradient, initial_parameters);

        // Prepare final state
        let mut final_state = State::zero_state(self.ansatz.num_qubits());
        self.ansatz
            .apply(&mut final_state, &opt_result.optimal_parameters);

        if self.verbose {
            println!("\nVQE optimization complete!");
            println!("Ground state energy: {:.6}", opt_result.optimal_value);
            println!("Number of iterations: {}", opt_result.num_iterations);
            println!("Converged: {}", opt_result.converged);
        }

        VQEResult {
            ground_state_energy: opt_result.optimal_value,
            optimal_parameters: opt_result.optimal_parameters,
            num_iterations: opt_result.num_iterations,
            convergence_history: opt_result.convergence_history,
            final_state: Some(final_state),
        }
    }

    /// Run VQE with random initial parameters
    pub fn run_random(&mut self) -> VQEResult {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let initial_params: Vec<f64> = (0..self.ansatz.num_parameters())
            .map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI))
            .collect();

        self.run(&initial_params)
    }
}
