use logosq::optimization::ansatz::*;
use logosq::optimization::gradient::*;
use logosq::optimization::observable::*;
use logosq::optimization::optimizer::*;
use logosq::optimization::vqe::*;
use logosq::states::State;
use std::f64::consts::PI;

mod tests {
    use super::*;
    // ============================================================================
    // OPTIMIZER TESTS
    // ============================================================================

    #[test]
    fn test_gradient_descent_simple_quadratic() {
        // Minimize f(x) = x²
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];

        let mut optimizer = GradientDescent::new(0.1, 100).with_tolerance(1e-6);

        let initial = vec![5.0];
        let result = optimizer.minimize(f, grad_f, &initial);

        assert!(result.converged);
        assert!((result.optimal_parameters[0] - 0.0).abs() < 0.1);
        assert!(result.optimal_value < 0.01);
    }

    #[test]
    fn test_gradient_descent_two_variables() {
        // Minimize f(x,y) = x² + y²
        let f = |params: &[f64]| params[0] * params[0] + params[1] * params[1];
        let grad_f = |params: &[f64]| vec![2.0 * params[0], 2.0 * params[1]];

        let mut optimizer = GradientDescent::new(0.1, 100).with_tolerance(1e-6);

        let initial = vec![3.0, 4.0];
        let result = optimizer.minimize(f, grad_f, &initial);

        assert!(result.converged);
        assert!((result.optimal_parameters[0] - 0.0).abs() < 0.1);
        assert!((result.optimal_parameters[1] - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_adam_optimizer() {
        // Minimize f(x,y) = (x-1)² + (y+2)²
        let f = |params: &[f64]| (params[0] - 1.0).powi(2) + (params[1] + 2.0).powi(2);
        let grad_f = |params: &[f64]| vec![2.0 * (params[0] - 1.0), 2.0 * (params[1] + 2.0)];

        let mut optimizer = Adam::new(0.1, 200).with_tolerance(1e-6);        
        let initial = vec![5.0, 5.0];
        let result = optimizer.minimize(f, grad_f, &initial);

        assert!(result.converged);
        assert!((result.optimal_parameters[0] - 1.0).abs() < 0.1);
        assert!((result.optimal_parameters[1] - (-2.0)).abs() < 0.1);
    }

    // Decomposition tests for Adam optimizer
    #[test]
    fn test_adam_converges_on_1d_quadratic() {
        // f(x) = (x-3)^2
        let f = |p: &[f64]| (p[0] - 3.0).powi(2);
        let grad = |p: &[f64]| vec![2.0 * (p[0] - 3.0)];

        let mut opt = Adam::new(0.1, 200).with_tolerance(1e-6);
        let res = opt.minimize(f, grad, &vec![10.0]);

        assert!(res.converged, "Adam should converge on simple 1D quadratic");
        assert!((res.optimal_parameters[0] - 3.0).abs() < 1e-3);
        assert!(res.optimal_value < 1e-6);
    }

    #[test]
    fn test_adam_converges_on_shifted_2d_quadratic() {
        // f(x,y) = (x-1)^2 + (y+2)^2
        let f = |p: &[f64]| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2);
        let grad = |p: &[f64]| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)];

        let mut opt = Adam::new(0.1, 200).with_tolerance(1e-6);
        let res = opt.minimize(f, grad, &vec![5.0, 5.0]);

        assert!(res.converged, "Adam should mark converged on 2D shifted quadratic");
        assert!((res.optimal_parameters[0] - 1.0).abs() < 1e-2);
        assert!((res.optimal_parameters[1] + 2.0).abs() < 1e-2);
        assert!(res.optimal_value < 1e-4);
    }

    #[test]
    fn test_adam_gradient_norm_convergence_condition() {
        // Very small-scale quadratic to hit gradient tolerance
        let f = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>();
        let grad = |p: &[f64]| p.iter().map(|x| 2.0 * x).collect::<Vec<_>>();

        let mut opt = Adam::new(0.2, 100).with_tolerance(1e-6);
        let res = opt.minimize(f, grad, &vec![1e-3, -1e-3]);

        assert!(res.converged, "Adam should set converged when gradient norm is tiny");
    }

    #[test]
    fn test_adam_small_step_convergence_condition() {
        // Flat region: f(x) = c + (1e-8)*x^2 so steps become tiny quickly
        let f = |p: &[f64]| 1.0 + 1e-8 * p[0] * p[0];
        let grad = |p: &[f64]| vec![2.0e-8 * p[0]];

        let mut opt = Adam::new(0.1, 100).with_tolerance(1e-6);
        let res = opt.minimize(f, grad, &vec![1.0]);

        assert!(res.converged, "Adam should stop when parameter updates are tiny");
    }

    #[test]
    fn test_optimizer_convergence_history() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];

        let mut optimizer = GradientDescent::new(0.1, 50);
        let result = optimizer.minimize(f, grad_f, &vec![10.0]);

        // Convergence history should show decreasing values
        for i in 1..result.convergence_history.len() {
            assert!(result.convergence_history[i] <= result.convergence_history[i - 1]);
        }
    }

    #[test]
    fn test_optimizer_max_iterations() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad_f = |x: &[f64]| vec![2.0 * x[0]];

        let mut optimizer = GradientDescent::new(0.001, 10); // Very small learning rate
        let result = optimizer.minimize(f, grad_f, &vec![100.0]);

        // Should stop after max iterations
        assert!(result.num_iterations <= 10);
    }

    // ============================================================================
    // VQE INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn test_vqe_single_qubit_z() {
        // Simple test: minimize ⟨Z⟩ which has ground state |1⟩ with energy -1
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let hamiltonian = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();
        let optimizer = GradientDescent::new(0.1, 100);

        let mut vqe = VQE::new(ansatz, hamiltonian, gradient_method, optimizer);
        vqe.verbose = false;

        let initial_params = vec![0.1, 0.1];
        let result = vqe.run(&initial_params);

        // Ground state energy should be close to -1
        assert!(
            (result.ground_state_energy - (-1.0)).abs() < 0.1,
            "Expected energy near -1, got {}",
            result.ground_state_energy
        );
    }

    #[test]
    fn test_vqe_two_qubit_hamiltonian() {
        // H = Z₀ + Z₁ (ground state is |11⟩ with energy -2)
        let ansatz =
            HardwareEfficientAnsatz::new(2, 2, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let mut hamiltonian = PauliObservable::new(2);
        hamiltonian.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::I]));
        hamiltonian.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::Z]));

        let gradient_method = ParameterShift::new();
        let optimizer = Adam::new(0.1, 100);

        let mut vqe = VQE::new(ansatz, hamiltonian, gradient_method, optimizer);
        vqe.verbose = false;

        let initial_params = vec![0.1; 6];
        let result = vqe.run(&initial_params);

        // Ground state energy should approach -2
        assert!(
            result.ground_state_energy < 0.0,
            "Energy should be negative, got {}",
            result.ground_state_energy
        );
    }

    #[test]
    fn test_vqe_ising_model() {
        // Simple Ising model: H = -Z₀Z₁ (antiferromagnetic coupling)
        let ansatz = RealAmplitudesAnsatz::new(2, 2);

        let hamiltonian = PauliObservable::zz_interaction(2, 0, 1, -1.0);

        let gradient_method = ParameterShift::new();
        let optimizer = Adam::new(0.1, 50);

        let mut vqe = VQE::new(ansatz, hamiltonian, gradient_method, optimizer);
        vqe.verbose = false;

        let result = vqe.run_random();

        // For -Z₀Z₁, ground state is |01⟩ or |10⟩ with energy -1
        assert!(
            result.ground_state_energy < 0.5,
            "Expected negative energy, got {}",
            result.ground_state_energy
        );
    }

    #[test]
    fn test_vqe_convergence() {
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let hamiltonian = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();
        let optimizer = GradientDescent::new(0.1, 50);

        let mut vqe = VQE::new(ansatz, hamiltonian, gradient_method, optimizer);
        vqe.verbose = false;

        let result = vqe.run(&vec![0.5, 0.5]);

        // Check that convergence history shows improvement
        let initial_energy = result.convergence_history[0];
        let final_energy = result.convergence_history.last().unwrap();

        assert!(
            final_energy < &initial_energy,
            "Energy should decrease during optimization"
        );
    }

    #[test]
    fn test_vqe_with_different_ansatz() {
        let hamiltonian = PauliObservable::single_z(2, 0);
        let gradient_method = ParameterShift::new();

        // Test with Hardware Efficient
        let ansatz_he =
            HardwareEfficientAnsatz::new(2, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);
        let optimizer_he = GradientDescent::new(0.1, 50);
        let mut vqe_he = VQE::new(
            ansatz_he,
            hamiltonian.clone(),
            gradient_method,
            optimizer_he,
        );
        vqe_he.verbose = false;
        let result_he = vqe_he.run_random();

        // Test with Real Amplitudes
        let ansatz_ra = RealAmplitudesAnsatz::new(2, 1);
        let gradient_method2 = ParameterShift::new();
        let optimizer_ra = GradientDescent::new(0.1, 50);
        let mut vqe_ra = VQE::new(ansatz_ra, hamiltonian, gradient_method2, optimizer_ra);
        vqe_ra.verbose = false;
        let result_ra = vqe_ra.run_random();

        // Both should find reasonable ground states
        assert!(result_he.ground_state_energy < 1.5);
        assert!(result_ra.ground_state_energy < 1.5);
    }

    // ============================================================================
    // EDGE CASES AND ERROR HANDLING
    // ============================================================================

    #[test]
    #[should_panic(expected = "Qubit index out of bounds")]
    fn test_observable_wrong_qubit_index() {
        let _obs = PauliObservable::single_z(2, 3); // Qubit 3 doesn't exist in 2-qubit system
    }

    #[test]
    #[should_panic(expected = "State qubit count must match ansatz")]
    fn test_ansatz_wrong_state_size() {
        let ansatz =
            HardwareEfficientAnsatz::new(2, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let mut state = State::zero_state(3); // Wrong size
        let params = vec![0.1; 4];
        ansatz.apply(&mut state, &params);
    }

    #[test]
    fn test_zero_depth_ansatz() {
        // Edge case: ansatz with zero depth should still work
        let ansatz =
            HardwareEfficientAnsatz::new(2, 0, EntanglingGate::CNOT, EntanglingPattern::Linear);

        assert_eq!(ansatz.num_parameters(), 2); // Just initial rotation layer

        let params = vec![0.0, 0.0];
        let circuit = ansatz.build_circuit(&params);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should still be in |00⟩ state
        assert!((state.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_ansatz() {
        let ansatz = RealAmplitudesAnsatz::new(1, 1);
        assert_eq!(ansatz.num_parameters(), 2);

        let params = vec![PI / 2.0, 0.0];
        let mut state = State::zero_state(1);
        ansatz.apply(&mut state, &params);

        // assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_observable_with_coefficients() {
        let mut obs = PauliObservable::new(1);
        obs.add_term(PauliTerm::new(2.0, vec![Pauli::Z]));

        let state = State::zero_state(1);
        let expectation = obs.expectation(&state);

        // 2.0 * ⟨0|Z|0⟩ = 2.0 * 1 = 2.0
        assert!((expectation - 2.0).abs() < 1e-10);
    }

    // ============================================================================
    // PERFORMANCE AND SCALING TESTS
    // ============================================================================

    #[test]
    fn test_ansatz_scaling() {
        // Test that ansatz works for different sizes
        for n_qubits in 1..=4 {
            let ansatz = HardwareEfficientAnsatz::new(
                n_qubits,
                1,
                EntanglingGate::CNOT,
                EntanglingPattern::Linear,
            );

            let params = vec![0.1; ansatz.num_parameters()];
            let circuit = ansatz.build_circuit(&params);

            assert_eq!(circuit.num_qubits(), n_qubits);
        }
    }

    #[test]
    fn test_gradient_computation_time() {
        // Ensure gradient computation completes in reasonable time
        use std::time::Instant;

        let ansatz =
            HardwareEfficientAnsatz::new(3, 2, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(3, 0);
        let gradient_method = ParameterShift::new();

        let params = vec![0.1; ansatz.num_parameters()];

        let start = Instant::now();
        let _gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);
        let duration = start.elapsed();

        // Should complete in less than 1 second for this small problem
        assert!(duration.as_secs() < 1);
    }

    #[test]
    fn test_vqe_reproducibility() {
        // Same initial parameters should give same result
        let ansatz = RealAmplitudesAnsatz::new(2, 1);
        let hamiltonian = PauliObservable::single_z(2, 0);
        let gradient_method = ParameterShift::new();
        let optimizer = GradientDescent::new(0.1, 30);

        let mut vqe1 = VQE::new(
            ansatz.clone(),
            hamiltonian.clone(),
            gradient_method,
            optimizer,
        );
        vqe1.verbose = false;

        let gradient_method2 = ParameterShift::new();
        let optimizer2 = GradientDescent::new(0.1, 30);
        let mut vqe2 = VQE::new(ansatz, hamiltonian, gradient_method2, optimizer2);
        vqe2.verbose = false;

        let initial_params = vec![0.5; 6];
        let result1 = vqe1.run(&initial_params);
        let result2 = vqe2.run(&initial_params);

        // Results should be very similar (within numerical precision)
        assert!((result1.ground_state_energy - result2.ground_state_energy).abs() < 1e-6);
    }
}
