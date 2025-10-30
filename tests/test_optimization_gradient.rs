use logosq::optimization::ansatz::*;
use logosq::optimization::gradient::*;
use logosq::optimization::observable::*;
use std::f64::consts::PI;

mod tests {
    use super::*;

    // ============================================================================
    // GRADIENT TESTS
    // ============================================================================

    #[test]
    fn test_parameter_shift_gradient_simple() {
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();

        // Use non-zero parameters where gradient should be non-zero
        let params = vec![PI / 4.0, PI / 6.0];
        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        assert_eq!(gradient.len(), 2);
        
        // At non-trivial angles, gradient should be non-zero
        let has_nonzero = gradient.iter().any(|&g| g.abs() > 1e-10);
        assert!(
            has_nonzero,
            "Expected non-zero gradient at θ=[π/4, π/6], got {:?}",
            gradient
        );
    }

    #[test]
    fn test_parameter_shift_at_extremum() {
        // Test that gradient is zero at an extremum
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();

        // At θ = 0, we're at maximum of ⟨Z⟩ for this circuit
        let params = vec![0.0, 0.0];
        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        assert_eq!(gradient.len(), 2);
        
        // Gradient should be near zero at extremum
        for &g in &gradient {
            assert!(
                g.abs() < 1e-8,
                "Expected zero gradient at extremum, got {}",
                g
            );
        }
    }

    #[test]
    fn test_parameter_shift_gradient_known_value() {
        // Test against a known analytical gradient
        // For single RY rotation: |ψ⟩ = RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        // ⟨Z⟩ = cos²(θ/2) - sin²(θ/2) = cos(θ)
        // d⟨Z⟩/dθ = -sin(θ)
        
        let ansatz =
            HardwareEfficientAnsatz::new(1, 0, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();

        // At θ = π/4
        let theta = PI / 4.0;
        let params = vec![theta];
        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        // Analytical gradient: -sin(π/4) ≈ -0.7071
        let expected_gradient = -theta.sin();
        
        assert!(
            (gradient[0] - expected_gradient).abs() < 1e-6,
            "Expected gradient ≈ {}, got {}",
            expected_gradient,
            gradient[0]
        );
    }

    #[test]
    fn test_parameter_shift_vs_finite_difference() {
        let ansatz = RealAmplitudesAnsatz::new(2, 1);
        let obs = PauliObservable::single_z(2, 0);

        let params = vec![0.5, 1.0, 1.5, 0.3, 0.8, 0.2];

        let ps_method = ParameterShift::new();
        let fd_method = FiniteDifference::new(1e-5);

        let ps_gradient = ps_method.compute_gradient(&ansatz, &obs, &params);
        let fd_gradient = fd_method.compute_gradient(&ansatz, &obs, &params);

        // Both methods should give similar results
        for i in 0..params.len() {
            assert!(
                (ps_gradient[i] - fd_gradient[i]).abs() < 1e-4,
                "Gradient mismatch at index {}: PS={}, FD={}",
                i,
                ps_gradient[i],
                fd_gradient[i]
            );
        }
    }

    #[test]
    fn test_gradient_at_minimum() {
        // For ⟨Z⟩, minimum is at |1⟩ state (energy = -1)
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(1, 0);
        let gradient_method = ParameterShift::new();

        // RY(π) flips |0⟩ to |1⟩
        let params = vec![PI, 0.0];
        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        // Gradient should be close to zero at minimum
        for &g in &gradient {
            assert!(
                g.abs() < 0.1,
                "Expected small gradient at minimum, got {}",
                g
            );
        }
    }

    #[test]
    fn test_finite_difference_custom_epsilon() {
        let ansatz = RealAmplitudesAnsatz::new(1, 1);
        let obs = PauliObservable::single_z(1, 0);

        let params = vec![0.5, 1.0];

        let fd_large = FiniteDifference::new(1e-3);
        let fd_small = FiniteDifference::new(1e-7);

        let grad_large = fd_large.compute_gradient(&ansatz, &obs, &params);
        let grad_small = fd_small.compute_gradient(&ansatz, &obs, &params);

        // Both should give reasonable results
        assert_eq!(grad_large.len(), params.len());
        assert_eq!(grad_small.len(), params.len());
        
        // They should be approximately equal
        for i in 0..params.len() {
            assert!(
                (grad_large[i] - grad_small[i]).abs() < 1e-3,
                "Large difference between epsilon values: {} vs {}",
                grad_large[i],
                grad_small[i]
            );
        }
    }

    #[test]
    fn test_parameter_shift_custom_shift() {
        let ansatz =
            HardwareEfficientAnsatz::new(1, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::single_z(1, 0);

        let params = vec![0.5, 1.0];

        let ps_default = ParameterShift::new();
        let ps_custom = ParameterShift::with_shift(PI / 4.0);

        let grad_default = ps_default.compute_gradient(&ansatz, &obs, &params);
        let grad_custom = ps_custom.compute_gradient(&ansatz, &obs, &params);

        // Both should compute valid gradients
        assert_eq!(grad_default.len(), grad_custom.len());
        
        // They will have different values but should both be reasonable
        for i in 0..params.len() {
            assert!(grad_default[i].is_finite());
            assert!(grad_custom[i].is_finite());
        }
    }

    #[test]
    fn test_gradient_on_two_qubit_system() {
        let ansatz =
            HardwareEfficientAnsatz::new(2, 1, EntanglingGate::CNOT, EntanglingPattern::Linear);

        let obs = PauliObservable::zz_interaction(2, 0, 1, 1.0);
        let gradient_method = ParameterShift::new();

        let params = vec![0.2, 0.4, 0.6, 0.8];
        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        assert_eq!(gradient.len(), 4);
        
        // For ZZ interaction with entangling gates, expect non-zero gradients
        let total_gradient_magnitude: f64 = gradient.iter().map(|g| g.abs()).sum();
        assert!(
            total_gradient_magnitude > 1e-10,
            "Expected some non-zero gradient components"
        );
    }

    #[test]
    fn test_gradient_linearity() {
        // Test that gradient scales linearly with observable coefficient
        let ansatz = RealAmplitudesAnsatz::new(2, 1);
        let params = vec![0.5, 1.0, 1.5, 0.3, 0.8, 0.2];

        let obs1 = PauliObservable::single_z(2, 0);
        let obs2 = {
            let mut obs = PauliObservable::new(2);
            obs.add_term(PauliTerm::new(2.0, vec![Pauli::Z, Pauli::I]));
            obs
        };

        let gradient_method = ParameterShift::new();

        let grad1 = gradient_method.compute_gradient(&ansatz, &obs1, &params);
        let grad2 = gradient_method.compute_gradient(&ansatz, &obs2, &params);

        // grad2 should be approximately 2 * grad1
        for i in 0..params.len() {
            assert!(
                (grad2[i] - 2.0 * grad1[i]).abs() < 1e-10,
                "Gradient should scale linearly: 2*{} ≠ {}",
                grad1[i],
                grad2[i]
            );
        }
    }

    #[test]
    fn test_gradient_vanishes_for_identity_observable() {
        // If observable is identity, gradient should be zero
        let ansatz = HardwareEfficientAnsatz::new(
            2,
            1,
            EntanglingGate::CNOT,
            EntanglingPattern::Linear,
        );

        let mut obs = PauliObservable::new(2);
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::I]));

        let gradient_method = ParameterShift::new();
        let params = vec![0.5, 1.0, 1.5, 2.0];

        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        // Identity observable has constant expectation value → gradient = 0
        for &g in &gradient {
            assert!(
                g.abs() < 1e-10,
                "Gradient for identity observable should be zero, got {}",
                g
            );
        }
    }

    #[test]
    fn test_gradient_symmetry() {
        // For symmetric circuits, certain gradients should be equal
        let ansatz = HardwareEfficientAnsatz::new(
            2,
            1,
            EntanglingGate::CNOT,
            EntanglingPattern::Linear,
        );

        let obs = {
            let mut obs = PauliObservable::new(2);
            // Symmetric observable: Z₀ + Z₁
            obs.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::I]));
            obs.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::Z]));
            obs
        };

        let gradient_method = ParameterShift::new();
        
        // Symmetric parameters
        let theta = 0.5;
        let params = vec![theta, theta, theta, theta];

        let gradient = gradient_method.compute_gradient(&ansatz, &obs, &params);

        // Due to symmetry, some gradients should be equal
        // This is problem-dependent, but we can check consistency
        assert_eq!(gradient.len(), 4);
        
        // First layer gradients should be equal due to symmetry
        assert!(
            (gradient[0] - gradient[1]).abs() < 1e-6,
            "Expected symmetric gradients: {} vs {}",
            gradient[0],
            gradient[1]
        );
    }

    #[test]
    fn test_gradient_multiple_observables() {
        // Test gradient with sum of observables
        let ansatz = RealAmplitudesAnsatz::new(2, 1);
        let params = vec![0.3, 0.7, 1.1, 0.5, 0.9, 1.3];

        let mut hamiltonian = PauliObservable::new(2);
        hamiltonian.add_term(PauliTerm::new(0.5, vec![Pauli::Z, Pauli::I]));
        hamiltonian.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::Z]));

        let gradient_method = ParameterShift::new();
        let gradient = gradient_method.compute_gradient(&ansatz, &hamiltonian, &params);

        assert_eq!(gradient.len(), params.len());
        
        // Should have at least some non-zero components
        let has_nonzero = gradient.iter().any(|&g| g.abs() > 1e-10);
        assert!(has_nonzero, "Expected some non-zero gradient components");
    }
}
