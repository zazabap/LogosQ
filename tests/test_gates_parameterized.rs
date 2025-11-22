use logosq::circuits::Circuit;
use logosq::states::State;
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================================
// SINGLE-QUBIT PARAMETERIZED ROTATION TESTS
// ============================================================================

mod tests {
    use super::*;

    #[test]
    fn test_rx_gate_pi_rotation() {
        // RX(π)|0⟩ should give -i|1⟩
        let mut circuit = Circuit::new(1);
        circuit.rx(0, PI);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let vector = state.vector().as_slice().unwrap();
        // Check |0⟩ amplitude is ~0
        assert!((vector[0].norm() - 0.0).abs() < 1e-10);
        // Check |1⟩ amplitude is 1 with phase -π/2 (which is -i)
        assert!((vector[1].norm() - 1.0).abs() < 1e-10);
        assert!((vector[1].arg() - (-PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rx_gate_half_pi() {
        // RX(π/2)|0⟩ should give (|0⟩ - i|1⟩)/√2
        let mut circuit = Circuit::new(1);
        circuit.rx(0, PI / 2.0);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate_pi_rotation() {
        // RY(π)|0⟩ should give |1⟩
        let mut circuit = Circuit::new(1);
        circuit.ry(0, PI);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let vector = state.vector().as_slice().unwrap();
        assert!((vector[0].norm() - 0.0).abs() < 1e-10);
        assert!((vector[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_ry_gate_half_pi() {
        // RY(π/2)|0⟩ should give (|0⟩ + |1⟩)/√2
        let mut circuit = Circuit::new(1);
        circuit.ry(0, PI / 2.0);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);

        // Check phases are real (no imaginary component)
        let vector = state.vector().as_slice().unwrap();
        assert!(vector[0].im.abs() < 1e-10);
        assert!(vector[1].im.abs() < 1e-10);
    }

    #[test]
    fn test_rz_gate_pi_rotation() {
        // RZ(π) on |+⟩ should give |-⟩
        let mut circuit = Circuit::new(1);
        circuit.h(0); // Create |+⟩
        circuit.rz(0, PI);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        // Apply another H to convert back to computational basis
        let mut circuit2 = Circuit::new(1);
        circuit2.h(0);
        circuit2.execute(&mut state).unwrap();

        // Should be in |1⟩ state
        let probs = state.probabilities();
        assert!(probs[0] < 1e-10);
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rz_preserves_probabilities() {
        // RZ only changes phase, not probabilities
        let mut circuit = Circuit::new(1);
        circuit.h(0); // Create |+⟩

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let probs_before = state.probabilities();

        let mut circuit2 = Circuit::new(1);
        circuit2.rz(0, PI / 4.0);
        circuit2.execute(&mut state).unwrap();

        let probs_after = state.probabilities();
        assert!((probs_before[0] - probs_after[0]).abs() < 1e-10);
        assert!((probs_before[1] - probs_after[1]).abs() < 1e-10);
    }

    #[test]
    fn test_phase_gate() {
        // Phase gate P(θ) = diag(1, e^(iθ))
        let mut circuit = Circuit::new(1);
        circuit.h(0); // Create |+⟩
        circuit.phase(0, PI / 2.0); // Apply S gate (phase π/2)

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let vector = state.vector().as_slice().unwrap();
        // |0⟩ amplitude should be unchanged (1/√2)
        assert!((vector[0] - Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0)).norm() < 1e-10);
        // |1⟩ amplitude should have phase π/2 (i/√2)
        assert!((vector[1] - Complex64::new(0.0, 1.0 / std::f64::consts::SQRT_2)).norm() < 1e-10);
    }

    #[test]
    fn test_u3_gate_as_x() {
        // U3(π, 0, π) should be equivalent to X gate
        let mut circuit = Circuit::new(1);
        circuit.u3(0, PI, 0.0, PI);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        // Should be in |1⟩ state (up to global phase)
        let probs = state.probabilities();
        assert!(probs[0] < 1e-10);
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_u3_gate_as_hadamard() {
        // U3(π/2, 0, π) should be equivalent to Hadamard
        let mut circuit = Circuit::new(1);
        circuit.u3(0, PI / 2.0, 0.0, PI);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        // Should be in |+⟩ state (up to global phase)
        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    // ============================================================================
    // TWO-QUBIT CONTROLLED ROTATION TESTS
    // ============================================================================

    #[test]
    fn test_crx_gate() {
        // CRX should only rotate target when control is |1⟩

        // Test with control = |0⟩ (no rotation)
        let mut circuit = Circuit::new(2);
        circuit.crx(0, 1, PI);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should still be |00⟩
        assert!((state.probability(0) - 1.0).abs() < 1e-10);

        // Test with control = |1⟩ (rotation applied)
        let mut circuit2 = Circuit::new(2);
        circuit2.x(0); // Set control to |1⟩
        circuit2.crx(0, 1, PI);

        let mut state2 = State::zero_state(2);
        circuit2.execute(&mut state2).unwrap();

        // Should be |11⟩ (with some phase)
        assert!(state2.probability(0) < 1e-10);
        assert!(state2.probability(1) < 1e-10);
        assert!(state2.probability(2) < 1e-10);
        assert!((state2.probability(3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cry_gate() {
        // CRY should only rotate target when control is |1⟩

        // Test with control = |1⟩
        let mut circuit = Circuit::new(2);
        circuit.x(0); // Set control to |1⟩
        circuit.cry(0, 1, PI / 2.0);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Target should be in superposition
        assert!((state.probability(2) - 0.5).abs() < 1e-10); // |10⟩
        assert!((state.probability(3) - 0.5).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn test_crz_gate() {
        // CRZ should only add phase when control is |1⟩

        // Prepare |1+⟩ state
        let mut circuit = Circuit::new(2);
        circuit.x(0); // Control to |1⟩
        circuit.h(1); // Target to |+⟩
        circuit.crz(0, 1, PI);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Probabilities should be unchanged
        assert!((state.probability(2) - 0.5).abs() < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);

        // But phase should have changed
        let vector = state.vector().as_slice().unwrap();
        // Check that the phase relationship changed
        let phase_diff = (vector[3] / vector[2]).arg();
        assert!((phase_diff.abs() - PI).abs() < 1e-10);
    }

    #[test]
    fn test_cphase_param_gate() {
        // Controlled-phase should add phase e^(iθ) when both qubits are |1⟩

        let mut circuit = Circuit::new(2);
        circuit.x(0);
        circuit.x(1);
        circuit.h(1); // Create (|10⟩ + |11⟩)/√2 after X on both

        // Actually need to create superposition differently
        let mut circuit2 = Circuit::new(2);
        circuit2.h(0);
        circuit2.h(1); // Create |++⟩
        circuit2.cphase_param(0, 1, PI / 2.0);

        let mut state = State::zero_state(2);
        circuit2.execute(&mut state).unwrap();

        // All probabilities should be equal
        let probs = state.probabilities();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    // ============================================================================
    // TWO-QUBIT ENTANGLING GATE TESTS
    // ============================================================================

    #[test]
    fn test_rxx_gate() {
        // RXX creates entanglement similar to CNOT

        let mut circuit = Circuit::new(2);
        circuit.h(0); // Create |+0⟩
        circuit.rxx(0, 1, PI / 2.0);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should create some entanglement
        let probs = state.probabilities();
        // Check that we have non-zero probabilities on multiple basis states
        let non_zero_count = probs.iter().filter(|&&p| p > 1e-10).count();
        assert!(non_zero_count > 1);
    }

    #[test]
    fn test_ryy_gate() {
        // RYY gate test

        let mut circuit = Circuit::new(2);
        circuit.h(0);
        circuit.h(1); // Create |++⟩
        circuit.ryy(0, 1, PI / 4.0);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should still have equal probabilities (RYY is diagonal in ZZ basis)
        let probs = state.probabilities();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rzz_gate_basic() {
        // RZZ gate basic test

        let mut circuit = Circuit::new(2);
        circuit.h(0);
        circuit.h(1); // Create |++⟩
        circuit.rzz(0, 1, PI / 4.0);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should still have equal probabilities (RZZ only changes phases)
        let probs = state.probabilities();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rzz_gate_phase() {
        // RZZ applies different phases based on parity

        let mut circuit = Circuit::new(2);
        circuit.rzz(0, 1, PI);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // |00⟩ should get phase e^(-iπ/2) = -i
        let vector = state.vector().as_slice().unwrap();
        assert!((vector[0] - Complex64::new(0.0, -1.0)).norm() < 1e-10);
    }

    // ============================================================================
    // COMBINED OPERATIONS TESTS
    // ============================================================================

    #[test]
    fn test_variational_circuit_pattern() {
        // Test a typical VQC pattern: RY - RY - RZZ

        let mut circuit = Circuit::new(2);
        circuit.ry(0, 0.5);
        circuit.ry(1, 0.3);
        circuit.rzz(0, 1, 0.7);
        circuit.ry(0, 0.2);
        circuit.ry(1, 0.4);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Check state is normalized
        let probs = state.probabilities();
        let total_prob: f64 = probs.iter().sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_variations() {
        // Test that changing parameters changes the output

        let mut circuit1 = Circuit::new(1);
        circuit1.ry(0, 0.5);

        let mut state1 = State::zero_state(1);
        circuit1.execute(&mut state1).unwrap();
        let probs1 = state1.probabilities();

        let mut circuit2 = Circuit::new(1);
        circuit2.ry(0, 1.0);

        let mut state2 = State::zero_state(1);
        circuit2.execute(&mut state2).unwrap();
        let probs2 = state2.probabilities();

        // Probabilities should be different
        assert!((probs1[0] - probs2[0]).abs() > 1e-5);
        assert!((probs1[1] - probs2[1]).abs() > 1e-5);
    }

    #[test]
    fn test_rotation_composition() {
        // Test that RZ(θ₁) RZ(θ₂) = RZ(θ₁ + θ₂)

        let mut circuit1 = Circuit::new(1);
        circuit1.h(0);
        circuit1.rz(0, 0.3);
        circuit1.rz(0, 0.7);

        let mut state1 = State::zero_state(1);
        circuit1.execute(&mut state1).unwrap();

        let mut circuit2 = Circuit::new(1);
        circuit2.h(0);
        circuit2.rz(0, 1.0);

        let mut state2 = State::zero_state(1);
        circuit2.execute(&mut state2).unwrap();

        // States should be equivalent
        let vector1 = state1.vector().as_slice().unwrap();
        let vector2 = state2.vector().as_slice().unwrap();

        for i in 0..2 {
            assert!((vector1[i] - vector2[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn test_zero_angle_is_identity() {
        // RY(0) should be identity

        let mut circuit = Circuit::new(1);
        circuit.ry(0, 0.0);

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        // Should still be |0⟩
        assert!((state.probability(0) - 1.0).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
    }

    #[test]
    fn test_bell_state_with_parameterized_gates() {
        // Create Bell state using parameterized gates

        let mut circuit = Circuit::new(2);
        circuit.ry(0, PI / 2.0); // Hadamard-like
        circuit.rz(0, PI); // Complete Hadamard
        circuit.cnot(0, 1);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // Should create Bell state |00⟩ + |11⟩ (up to phase)
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!(state.probability(1) < 1e-10);
        assert!(state.probability(2) < 1e-10);
        assert!((state.probability(3) - 0.5).abs() < 1e-10);
    }

    // ============================================================================
    // EDGE CASES AND SPECIAL VALUES
    // ============================================================================

    #[test]
    fn test_multiple_rzz_gates() {
        // Test multiple RZZ gates on same qubits

        let mut circuit = Circuit::new(2);
        circuit.h(0);
        circuit.h(1);
        circuit.rzz(0, 1, PI / 8.0);
        circuit.rzz(0, 1, PI / 8.0);
        circuit.rzz(0, 1, PI / 8.0);
        circuit.rzz(0, 1, PI / 8.0);

        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();

        // 4 × π/8 = π/2, should be equivalent to single RZZ(π/2)
        let mut circuit2 = Circuit::new(2);
        circuit2.h(0);
        circuit2.h(1);
        circuit2.rzz(0, 1, PI / 2.0);

        let mut state2 = State::zero_state(2);
        circuit2.execute(&mut state2).unwrap();

        let vector1 = state.vector().as_slice().unwrap();
        let vector2 = state2.vector().as_slice().unwrap();

        for i in 0..4 {
            assert!((vector1[i] - vector2[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn test_large_angle_wrap() {
        // Test that 2π rotation is equivalent to identity

        let mut circuit1 = Circuit::new(1);
        circuit1.ry(0, 2.0 * PI);

        let mut state1 = State::zero_state(1);
        circuit1.execute(&mut state1).unwrap();

        // Should be back to |0⟩ (up to global phase)
        let probs = state1.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
    }
}
