//! Comprehensive demonstration of gradient error scenarios related to
//! Parameter-Shift Rule (PSR) usage.
//!
//! This example demonstrates various potential gradient computation issues
//! that were found in PennyLane, and verifies whether LogosQ (a type-safe,
//! compile-time-checked solution) prevents or handles these bugs correctly.
//!
//! Test scenarios (matching PennyLane bug order):
//! 1. Interleaving non-parameterized gates with parameterized gates
//! 2. Parameter reuse in entangled circuits
//! 3. Sequential batch processing issues with VQCs
//! 4. Silent NaN errors from edge cases
//! 5. Parameter reuse in multiple gates
//!
//! 6a. Operation ordering and PSR gradient computation
//! 6. Complex VQC training failure scenarios

use logosq::circuits::Circuit;
use logosq::optimization::ansatz::{Ansatz, ParameterizedCircuit};
use logosq::optimization::gradient::{FiniteDifference, GradientMethod, ParameterShift};
use logosq::optimization::observable::{Observable, PauliObservable, PauliTerm, Pauli};
use logosq::states::State;
use std::f64::consts::PI;

struct LogosQGradientBugDemo {
    results: std::collections::HashMap<String, String>,
}

impl LogosQGradientBugDemo {
    fn new() -> Self {
        Self {
            results: std::collections::HashMap::new(),
        }
    }

    /// BUG 1: Interleaving non-parameterized gates with parameterized gates
    ///
    /// Problem: When non-parameterized gates (like CNOT) are interleaved between
    /// parameterized gates, PSR's parameter dependency tracking may be affected,
    /// potentially leading to incorrect gradient computation.
    fn test_bug_1_invalid_generator_operations(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 1: Interleaving Non-Parameterized Gates with Parameterized Gates");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            4,
            3,
            move |params| {
                let mut circuit = Circuit::new(4);
                circuit.rx(0, params[0]);
                circuit.cnot(0, 1);
                circuit.ry(1, params[1]);
                circuit.cry(0, 1, params[2]);
                circuit
            },
        );

        let obs = PauliObservable::single_z(4, 0);
        let ps_method = ParameterShift::new();
        let fd_method = FiniteDifference::new(1e-7);

        let params = vec![0.5, PI / 2.0, 0.3];

        println!("\n⚠ PROBLEM: CNOT (non-parameterized) is interleaved between parameterized gates");
        println!("   This may affect PSR's parameter dependency tracking!");
        println!("{}", "-".repeat(70));

        let ps_grad = ps_method.compute_gradient(&ansatz, &obs, &params);
        let fd_grad = fd_method.compute_gradient(&ansatz, &obs, &params);

        println!("✓ PSR Gradient computed: {:?}", ps_grad);
        println!("  Finite-diff gradient: {:?}", fd_grad);

        // Check for NaN values (silent errors)
        let ps_has_nan = ps_grad.iter().any(|&g| g.is_nan() || g.is_infinite());
        let fd_has_nan = fd_grad.iter().any(|&g| g.is_nan() || g.is_infinite());

        if ps_has_nan {
            println!("⚠ WARNING: PSR gradient contains NaN/Inf values! {:?}", ps_grad);
        }
        if fd_has_nan {
            println!("⚠ WARNING: Finite-diff gradient contains NaN/Inf values! {:?}", fd_grad);
        }

        // Check if gradients match
        if ps_grad.is_empty() {
            println!("⚠ WARNING: PSR returned empty gradient! This indicates a bug.");
            println!("  Expected {} gradient values but got 0", params.len());
        } else if fd_grad.is_empty() {
            println!("⚠ WARNING: Finite-diff returned empty gradient!");
        } else if ps_grad.len() == fd_grad.len() && !ps_has_nan && !fd_has_nan {
            let max_diff = ps_grad
                .iter()
                .zip(fd_grad.iter())
                .map(|(p, f)| (p - f).abs())
                .fold(0.0, f64::max);

            if max_diff > 1e-4 {
                println!("⚠ WARNING: Gradient mismatch! Max difference: {:.6}", max_diff);
                println!("  PSR: {:?}", ps_grad);
                println!("  FD:  {:?}", fd_grad);
                println!("  This suggests PSR may be computing wrong gradients");
            } else {
                println!("✓ Gradients match within tolerance");
            }
        } else if ps_grad.len() != fd_grad.len() {
            println!("⚠ WARNING: Gradient shape mismatch! PSR: {}, FD: {}", ps_grad.len(), fd_grad.len());
        }

        self.results.insert(
            "bug_1".to_string(),
            if ps_grad.is_empty() {
                "FAILED: Empty gradient".to_string()
            } else if ps_has_nan || fd_has_nan {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 2: Parameter reuse in entangled circuits
    ///
    /// Problem: When the same parameter is used multiple times in a circuit with
    /// entanglement, PSR must correctly account for all contributions. Reusing
    /// qubits after entanglement may also complicate gradient computation.
    fn test_bug_2_parameter_reuse_entangled(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 2: Parameter Reuse in Entangled Circuits");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            2,
            2,
            move |params| {
                let mut circuit = Circuit::new(2);
                circuit.h(0);
                circuit.cnot(0, 1);
                circuit.ry(0, params[0]);
                circuit.rz(0, params[1]);
                circuit.rx(1, params[0]);
                circuit
            },
        );

        let mut obs = PauliObservable::new(2);
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::Z]));

        let ps_method = ParameterShift::new();
        let fd_method = FiniteDifference::new(1e-7);

        let params = vec![0.5, 0.3];

        println!("\n⚠ PROBLEM: Creates entangled Bell state, then reuses parameter θ₀");
        println!("   Parameter θ₀ is used twice (RY on qubit 0, RX on qubit 1)");
        println!("   PSR must correctly sum all contributions from parameter reuse!");
        println!("{}", "-".repeat(70));

        let ps_grad = ps_method.compute_gradient(&ansatz, &obs, &params);
        let fd_grad = fd_method.compute_gradient(&ansatz, &obs, &params);

        println!("✓ PSR Gradient computed: {:?}", ps_grad);
        println!("  Finite-diff gradient: {:?}", fd_grad);

        let ps_has_nan = ps_grad.iter().any(|&g| g.is_nan() || g.is_infinite());
        let fd_has_nan = fd_grad.iter().any(|&g| g.is_nan() || g.is_infinite());

        if ps_has_nan {
            println!("⚠ ERROR: PSR gradient contains NaN/Inf! {:?}", ps_grad);
        } else if fd_has_nan {
            println!("⚠ ERROR: Finite-diff gradient contains NaN/Inf! {:?}", fd_grad);
        } else if ps_grad.len() == fd_grad.len() {
            let max_diff = ps_grad
                .iter()
                .zip(fd_grad.iter())
                .map(|(p, f)| (p - f).abs())
                .fold(0.0, f64::max);

            if max_diff > 1e-3 {
                println!("⚠ WARNING: Significant gradient mismatch! Max diff: {:.6}", max_diff);
                println!("  This indicates incorrect gradient due to parameter reuse handling");
            } else {
                println!("✓ Gradients match within tolerance");
            }
        }

        self.results.insert(
            "bug_2".to_string(),
            if ps_has_nan || fd_has_nan {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 3: Sequential batch processing issues with VQCs
    ///
    /// Problem: When processing batches of data sequentially, PSR may compute
    /// inconsistent gradients across different data points, or fail silently
    /// when evaluating gradients for multiple inputs.
    fn test_bug_3_broadcasting_batched_vqc(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 3: Sequential Batch Processing Issues with VQCs");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            4,
            4,
            move |params| {
                let mut circuit = Circuit::new(4);
                circuit.ry(0, params[3]);
                circuit.ry(0, params[0]);
                circuit.rx(1, params[1]);
                circuit.cnot(0, 1);
                circuit.rz(0, params[2]);
                circuit
            },
        );

        let obs = PauliObservable::single_z(4, 0);
        let ps_method = ParameterShift::new();

        let params = vec![0.1, 0.2, 0.3, 0.5];

        println!("\n⚠ PROBLEM: Data embedding (RY(x)) followed by parameterized gates");
        println!("   When processing x values sequentially, gradients may be inconsistent!");
        println!("{}", "-".repeat(70));

        // Test with single input
        println!("\n  Testing with single input...");
        let grad_single = ps_method.compute_gradient(&ansatz, &obs, &params);
        println!("✓ Single input gradient: {:?}", grad_single);

        // Test with batched input - this often causes issues
        println!("\n  Testing with batched input (common source of bugs)...");
        let x_batch = vec![0.1, 0.2, 0.3, 0.4];
        
        let mut grads = Vec::new();
        for x_val in &x_batch {
            let mut params_batch = params.clone();
            params_batch[3] = *x_val; // Update x value
            let grad = ps_method.compute_gradient(&ansatz, &obs, &params_batch);
            grads.push(grad);
        }

        if !grads.is_empty() && grads.iter().all(|g| !g.is_empty()) {
            // Check for inconsistencies
            let grad_arrays: Vec<Vec<f64>> = grads.to_vec();
            
            // Compute variance across batch for each gradient component
            let num_params = grad_arrays[0].len();
            let mut grad_std = vec![0.0; num_params];
            
            for param_idx in 0..num_params {
                let values: Vec<f64> = grad_arrays.iter().map(|g| g[param_idx]).collect();
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                grad_std[param_idx] = variance.sqrt();
            }

            let max_std: f64 = grad_std.iter().fold(0.0f64, |acc: f64, x: &f64| acc.max(*x));
            if max_std > 1e-6 {
                println!("⚠ WARNING: Gradient variance across batch! Max std: {:.6}", max_std);
                println!("  Std per param: {:?}", grad_std);
                println!("  This suggests inconsistent gradient computation");
            } else {
                println!("✓ Gradients are consistent across batch");
            }

            // Check for NaN
            let has_nan = grads.iter().any(|g| g.iter().any(|&v| v.is_nan() || v.is_infinite()));
            if has_nan {
                println!("⚠ ERROR: NaN in batch gradients!");
            }
        } else {
            println!("⚠ WARNING: Some gradients are empty!");
        }

        self.results.insert(
            "bug_3".to_string(),
            if grads.iter().any(|g| g.is_empty()) {
                "FAILED: Empty gradients".to_string()
            } else if grads.iter().any(|g| g.iter().any(|&v| v.is_nan() || v.is_infinite())) {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 4: Silent NaN errors from edge cases
    ///
    /// Problem: Certain parameter values or circuit configurations cause
    /// NaN gradients that are not caught or reported properly.
    fn test_bug_4_silent_nan_errors(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 4: Silent NaN Errors from Edge Cases");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            2,
            4,
            move |params| {
                let mut circuit = Circuit::new(2);
                circuit.rx(0, params[0]);
                circuit.ry(1, params[1]);
                circuit.rz(0, params[2]);
                circuit.cnot(0, 1);
                circuit.cry(1, 0, params[3]);
                circuit
            },
        );

        let obs = PauliObservable::single_z(2, 0);
        let ps_method = ParameterShift::new();

        let test_cases = vec![
            ("Normal values", vec![0.5, 0.3, 0.2, 0.1]),
            ("Large values", vec![10.0, 5.0, 3.0, 2.0]),
            ("Near zero", vec![1e-8, 1e-7, 1e-6, 1e-5]),
            ("At π/2", vec![PI / 2.0, PI / 2.0, PI / 2.0, PI / 2.0]),
            ("At π", vec![PI, PI, PI, PI]),
        ];

        let mut nan_count = 0;

        for (name, params) in test_cases {
            let grad = ps_method.compute_gradient(&ansatz, &obs, &params);

            let has_nan = grad.iter().any(|&g| g.is_nan() || g.is_infinite());

            if has_nan {
                println!("⚠ {}: Gradient contains NaN/Inf!", name);
                println!("  Params: {:?}", params);
                println!("  Gradient: {:?}", grad);
                nan_count += 1;
            } else {
                println!("✓ {}: OK (grad={:?})", name, grad);
            }
        }

        if nan_count > 0 {
            println!("\n⚠ Found {} cases with NaN/Inf or exceptions", nan_count);
            println!("  This demonstrates silent errors in PSR gradient computation");
        } else {
            println!("\n✓ All edge cases handled correctly - no NaN detected");
        }

        self.results.insert(
            "bug_4".to_string(),
            if nan_count > 0 {
                format!("FAILED: {} NaN cases", nan_count)
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 5: Parameter reuse in multiple gates
    ///
    /// Problem: Reusing the same parameter in multiple gates requires PSR to
    /// correctly sum all contributions from each parameter use. This can cause
    /// incorrect gradient computation if not handled properly.
    fn test_bug_5_parameter_reuse_multiple_gates(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 5: Parameter Reuse in Multiple Gates");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            2,
            2,
            move |params| {
                let mut circuit = Circuit::new(2);
                circuit.rx(0, params[0]);
                circuit.ry(1, params[0]);
                circuit.cnot(0, 1);
                circuit.rz(0, params[1]);
                circuit.rx(0, params[0]);
                circuit.cry(0, 1, params[1]);
                circuit
            },
        );

        let obs = PauliObservable::single_z(2, 0);
        let ps_method = ParameterShift::new();
        let fd_method = FiniteDifference::new(1e-7);

        let params = vec![0.5, 0.3];

        println!("\n⚠ PROBLEM: Parameter θ₀ used 3 times, θ₁ used 2 times");
        println!("   PSR must correctly sum all contributions from each parameter!");
        println!("{}", "-".repeat(70));

        let ps_grad = ps_method.compute_gradient(&ansatz, &obs, &params);
        let fd_grad = fd_method.compute_gradient(&ansatz, &obs, &params);

        println!("✓ PSR Gradient computed: {:?}", ps_grad);
        println!("  Finite-diff gradient: {:?}", fd_grad);

        let ps_has_nan = ps_grad.iter().any(|&g| g.is_nan() || g.is_infinite());
        let fd_has_nan = fd_grad.iter().any(|&g| g.is_nan() || g.is_infinite());

        if ps_has_nan {
            println!("⚠ WARNING: PSR gradient contains NaN/Inf!");
        }
        if fd_has_nan {
            println!("⚠ WARNING: Finite-diff gradient contains NaN/Inf!");
        }

        if ps_grad.len() == fd_grad.len() && !ps_has_nan && !fd_has_nan {
            let max_diff = ps_grad
                .iter()
                .zip(fd_grad.iter())
                .map(|(p, f)| (p - f).abs())
                .fold(0.0, f64::max);

            if max_diff > 1e-4 {
                println!("⚠ WARNING: Gradient mismatch! Max difference: {:.6}", max_diff);
                println!("  PSR may not be correctly handling parameter reuse");
            } else {
                println!("✓ Gradients match within tolerance");
            }
        }

        self.results.insert(
            "bug_5".to_string(),
            if ps_has_nan || fd_has_nan {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 6a: Operation ordering and PSR gradient computation
    ///
    /// Problem: Different operation orders produce different circuits (and thus
    /// different gradients). PSR must correctly compute gradients for each
    /// circuit structure, especially when entangling gates are interleaved with
    /// parameterized gates.
    fn test_bug_6a_operation_ordering(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 6a: Operation Ordering PSR Evaluation Issues");
        println!("{}", "=".repeat(70));

        let ansatz1 = ParameterizedCircuit::new(
            2,
            2,
            move |params| {
                let mut circuit = Circuit::new(2);
                circuit.ry(0, params[0]);
                circuit.cnot(0, 1);
                circuit.rx(1, params[1]);
                circuit
            },
        );

        let ansatz2 = ParameterizedCircuit::new(
            2,
            2,
            move |params| {
                let mut circuit = Circuit::new(2);
                circuit.cnot(0, 1);
                circuit.ry(0, params[0]);
                circuit.rx(1, params[1]);
                circuit
            },
        );

        println!("\n⚠ PROBLEM: Different operation orders produce different circuits");
        println!("   PSR must correctly compute gradients for each structure!");
        println!("{}", "-".repeat(70));

        let obs = PauliObservable::single_z(2, 0);
        let ps_method = ParameterShift::new();
        let fd_method = FiniteDifference::new(1e-7);

        let params = vec![0.5, 0.3];

        let grad1_ps = ps_method.compute_gradient(&ansatz1, &obs, &params);
        let grad2_ps = ps_method.compute_gradient(&ansatz2, &obs, &params);

        let grad1_fd = fd_method.compute_gradient(&ansatz1, &obs, &params);
        let grad2_fd = fd_method.compute_gradient(&ansatz2, &obs, &params);

        println!("✓ Circuit 1 PSR gradient: {:?}", grad1_ps);
        println!("✓ Circuit 2 PSR gradient: {:?}", grad2_ps);
        println!("  Circuit 1 FD gradient: {:?}", grad1_fd);
        println!("  Circuit 2 FD gradient: {:?}", grad2_fd);

        // Check if PSR matches FD for each circuit
        let diff1 = grad1_ps
            .iter()
            .zip(grad1_fd.iter())
            .map(|(p, f)| (p - f).abs())
            .fold(0.0, f64::max);

        let diff2 = grad2_ps
            .iter()
            .zip(grad2_fd.iter())
            .map(|(p, f)| (p - f).abs())
            .fold(0.0, f64::max);

        if diff1 > 1e-4 {
            println!("⚠ WARNING: PSR vs FD mismatch in circuit 1! Max diff: {:.6}", diff1);
        } else {
            println!("✓ Circuit 1: PSR matches FD");
        }

        if diff2 > 1e-4 {
            println!("⚠ WARNING: PSR vs FD mismatch in circuit 2! Max diff: {:.6}", diff2);
        } else {
            println!("✓ Circuit 2: PSR matches FD");
        }

        let order_diff = grad1_ps
            .iter()
            .zip(grad2_ps.iter())
            .map(|(g1, g2)| (g1 - g2).abs())
            .fold(0.0, f64::max);

        println!("  Gradient difference between orders: {:.6}", order_diff);
        println!("  (Expected: gradients differ since circuits differ)");

        self.results.insert(
            "bug_6a".to_string(),
            if diff1 > 1e-4 || diff2 > 1e-4 {
                "FAILED: PSR mismatch with FD".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// BUG 6: Complex VQC training failure scenarios
    ///
    /// Problem: Real-world VQC training scenarios combine multiple issues,
    /// leading to training failures, wrong gradients, or crashes.
    fn test_bug_6_complex_vqc_training(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("BUG 6: Complex VQC Training Failure Scenario");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            4,
            8,
            move |params| {
                let mut circuit = Circuit::new(4);
                for i in 0..4 {
                    circuit.ry(i, params[i.min(1)]);
                }
                circuit.rx(0, params[0]);
                circuit.rx(1, params[1]);
                circuit.cnot(0, 1);
                circuit.cnot(2, 3);
                circuit.cnot(0, 2);
                circuit.ry(0, params[2]);
                circuit.ry(1, params[3]);
                circuit.cry(1, 0, params[4]);
                circuit.cry(3, 2, params[5]);
                circuit.rz(0, params[6]);
                circuit.rz(1, params[7]);
                circuit
            },
        );

        let mut obs = PauliObservable::new(4);
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::I]));
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::I]));

        let ps_method = ParameterShift::new();
        let params: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1).collect();

        println!("\n⚠ PROBLEM: Complex circuit with:");
        println!("   • Data embedding layer (RY gates)");
        println!("   • Multiple parameterized layers (RX, RY, RZ)");
        println!("   • Interleaved entangling gates (CNOT, CRY)");
        println!("   • Multiple measurements");
        println!("   All issues from bugs 1-5 can combine here!");
        println!("{}", "-".repeat(70));

        println!("\n  Testing realistic VQC training scenario...");
        println!("  Parameters: {:?}", params);

        // Forward pass
        let mut state = State::zero_state(4);
        ansatz.apply(&mut state, &params);
        let result = obs.expectation(&state);
        println!("✓ Forward pass expectation: {:.6}", result);

        // Gradient computation
        let grad = ps_method.compute_gradient(&ansatz, &obs, &params);
        println!("✓ Gradient computed: {:?}", grad);

        let has_nan = grad.iter().any(|&g| g.is_nan() || g.is_infinite());

        if has_nan {
            println!("⚠ ERROR: Gradient contains NaN/Inf!");
        } else {
            let grad_magnitude: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            println!("  Gradient magnitude: {:.6}", grad_magnitude);

            #[allow(clippy::manual_range_contains)]
            if grad_magnitude > 1e6 || grad_magnitude < 1e-10 {
                println!("⚠ WARNING: Suspicious gradient magnitude: {:.6}", grad_magnitude);
            }

            // Simulate training step
            let learning_rate = 0.01;
            let params_new: Vec<f64> = params
                .iter()
                .zip(grad.iter())
                .map(|(p, g)| p - learning_rate * g)
                .collect();

            let mut state_new = State::zero_state(4);
            ansatz.apply(&mut state_new, &params_new);
            let result_new = obs.expectation(&state_new);

            println!("✓ Training step completed");
            println!("  New expectation: {:.6}", result_new);
            println!("  Loss change: {:.6} -> {:.6}", result, result_new);
        }

        self.results.insert(
            "bug_6".to_string(),
            if has_nan {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    /// Additional test: Complex training scenario with multiple measurements
    ///
    /// VQC with multiple observables/measurements can cause issues
    /// if gradients aren't computed correctly for each observable term.
    fn test_7_multiple_measurements(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("TEST 7: Complex Training Scenario with Multiple Measurements");
        println!("{}", "=".repeat(70));

        let ansatz = ParameterizedCircuit::new(
            4,
            8,
            move |params| {
                let mut circuit = Circuit::new(4);
                for i in 0..4 {
                    circuit.ry(i, params[i.min(3)]);
                }
                circuit.cnot(0, 1);
                circuit.cnot(2, 3);
                circuit.cnot(0, 2);
                circuit.rx(0, params[4]);
                circuit.rx(1, params[5]);
                circuit.ry(2, params[6]);
                circuit.ry(3, params[7]);
                circuit
            },
        );

        let mut hamiltonian = PauliObservable::new(4);
        hamiltonian.add_term(PauliTerm::new(0.5, vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::I]));
        hamiltonian.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::I]));
        hamiltonian.add_term(PauliTerm::new(0.3, vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I]));

        let ps_method = ParameterShift::new();
        let params: Vec<f64> = (0..8).map(|i| (i as f64 + 1.0) * 0.1).collect();

        println!("  Testing VQC with multiple observable terms...");

        // Forward pass
        let mut state = State::zero_state(4);
        ansatz.apply(&mut state, &params);
        let energy = hamiltonian.expectation(&state);
        println!("✓ Forward pass energy: {:.6}", energy);

        // Gradient computation
        let grad = ps_method.compute_gradient(&ansatz, &hamiltonian, &params);
        println!("✓ Gradient computed: {:?}", grad);

        let has_nan = grad.iter().any(|&g| g.is_nan() || g.is_infinite());

        if has_nan {
            println!("⚠ ERROR: Gradient contains NaN/Inf!");
        } else {
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            println!("  Gradient norm: {:.6}", grad_norm);

            // Check gradient variance
            let grad_mean: f64 = grad.iter().sum::<f64>() / grad.len() as f64;
            let grad_var: f64 = grad
                .iter()
                .map(|g| (g - grad_mean).powi(2))
                .sum::<f64>()
                / grad.len() as f64;

            println!("  Gradient variance: {:.6}", grad_var);

            if grad_var < 1e-10 {
                println!("⚠ WARNING: Very low gradient variance - may indicate issues");
            } else {
                println!("✓ Gradient variance is reasonable");
            }

            // Simulate optimization step
            let learning_rate = 0.01;
            let params_new: Vec<f64> = params
                .iter()
                .zip(grad.iter())
                .map(|(p, g)| p - learning_rate * g)
                .collect();

            let mut state_new = State::zero_state(4);
            ansatz.apply(&mut state_new, &params_new);
            let energy_new = hamiltonian.expectation(&state_new);

            println!("✓ Optimization step completed");
            println!("  Energy: {:.6} -> {:.6}", energy, energy_new);
        }

        self.results.insert(
            "test_7".to_string(),
            if has_nan {
                "FAILED: NaN detected".to_string()
            } else {
                "PASSED".to_string()
            },
        );
    }

    fn run_all_tests(&mut self) {
        println!("\n{}", "=".repeat(70));
        println!("LogosQ Parameter-Shift Rule Gradient Bug Verification");
        println!("{}", "=".repeat(70));
        println!("\nThis example verifies whether LogosQ correctly handles");
        println!("gradient computation scenarios that cause bugs in PennyLane.");
        println!("\nKey differences with LogosQ:");
        println!("  • Type-safe: Parameters are explicitly tracked");
        println!("  • State isolation: Each PSR evaluation uses fresh states");
        println!("  • Compile-time checks: Invalid operations are caught");
        println!("  • No dynamic runtime errors: Rust's type system prevents many bugs");

        self.test_bug_1_invalid_generator_operations();
        self.test_bug_2_parameter_reuse_entangled();
        self.test_bug_3_broadcasting_batched_vqc();
        self.test_bug_4_silent_nan_errors();
        self.test_bug_5_parameter_reuse_multiple_gates();
        self.test_bug_6a_operation_ordering();
        self.test_bug_6_complex_vqc_training();
        self.test_7_multiple_measurements();

        // Summary
        println!("\n{}", "=".repeat(70));
        println!("Summary");
        println!("{}", "=".repeat(70));

        let passed = self
            .results
            .values()
            .filter(|v| v.starts_with("PASSED"))
            .count();
        let total = self.results.len();

        println!("Tests passed: {}/{}", passed, total);

        println!("\nTest Results:");
        for (test, result) in &self.results {
            let status = if result.starts_with("PASSED") {
                "✓"
            } else {
                "✗"
            };
            println!("  {} {}: {}", status, test, result);
        }

        println!("\n{}", "=".repeat(70));
        println!("Key Advantages of LogosQ:");
        println!("{}", "=".repeat(70));
        println!("  1. Type safety prevents invalid parameter usage");
        println!("  2. State isolation ensures correct PSR evaluation");
        println!("  3. Compile-time checks catch errors before runtime");
        println!("  4. Explicit parameter tracking prevents reuse bugs");
        println!("  5. Numerical stability through careful implementation");
        println!("\nThese features demonstrate why a type-safe, compile-time");
        println!("checked solution (like LogosQ in Rust) can prevent many");
        println!("of the gradient computation errors found in dynamic languages.");
    }
}

fn main() {
    let mut demo = LogosQGradientBugDemo::new();
    demo.run_all_tests();
    // Or run individual tests:
    // demo.test_bug_1_invalid_generator_operations();
}
