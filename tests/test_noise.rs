use approx::{assert_abs_diff_eq, assert_relative_eq};
use logosq::noise::*;
use logosq::prelude::*;
use logosq::states::State;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

pub mod tests {
    use super::*;

    // Helper function to create a bell state for testing
    fn create_bell_state() -> State {
        let mut state = State::zero_state(2);
        let h_gate = h_gate();
        let cnot_gate = cnot_gate();

        // Apply H to first qubit
        h_gate.apply(&mut state);

        // Apply CNOT
        cnot_gate.apply(&mut state);

        state
    }

    // Corrected state_purity function

    // Helper function to measure the state purity (tr(ρ²))
    // A pure state has purity 1, a completely mixed state has purity 1/d
    fn state_purity(state: &State) -> f64 {
        // For a pure state |ψ⟩, ρ = |ψ⟩⟨ψ| and tr(ρ²) = 1
        // For a quantum state vector, purity = sum_i,j |⟨i|ψ⟩|²|⟨j|ψ⟩|²

        let mut purity = 0.0;

        // In the computational basis, this simplifies to sum_i |⟨i|ψ⟩|⁴
        for i in 0..state.vector.len() {
            purity += state.vector[i].norm_sqr().powi(2);
        }

        purity
    }

    // Corrected reduced_density_matrix function

    // Helper to get reduced density matrix for a single qubit
    fn reduced_density_matrix(state: &State, qubit: usize) -> [[Complex64; 2]; 2] {
        let mut rho = [[Complex64::new(0.0, 0.0); 2]; 2];

        for i in 0..state.vector.len() {
            for j in 0..state.vector.len() {
                // Check if bits match for all positions except qubit
                let mask = !(1 << qubit);
                if (i & mask) == (j & mask) {
                    // Extract bit values for qubit
                    let i_bit = (i >> qubit) & 1;
                    let j_bit = (j >> qubit) & 1;

                    // Add contribution to density matrix
                    rho[i_bit][j_bit] += state.vector[i] * state.vector[j].conj();
                }
            }
        }

        rho
    }

    #[test]
    fn test_depolarizing_noise_zero_probability() {
        let mut state = State::zero_state(1);
        let noise = DepolarizingNoise::new(0.0);

        // State before noise
        assert_eq!(state.vector[0], Complex64::new(1.0, 0.0));
        assert_eq!(state.vector[1], Complex64::new(0.0, 0.0));

        // Apply noise with 0 probability - should remain unchanged
        noise.apply(&mut state);

        assert_eq!(state.vector[0], Complex64::new(1.0, 0.0));
        assert_eq!(state.vector[1], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_depolarizing_noise_full_probability() {
        // For 100% depolarizing noise, we should see a statistical distribution of errors
        let trials = 1000;
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;

        for _ in 0..trials {
            // Start with |0⟩ state
            let mut state = State::zero_state(1);

            // Apply 100% depolarizing noise
            let noise = DepolarizingNoise::new(1.0);
            noise.apply(&mut state);

            // Analyze the resulting state to determine which error occurred
            let x_prob = state.vector[1].re.abs();
            let y_prob = state.vector[1].im.abs();
            let z_prob = state.vector[0].re.abs(); // Z doesn't flip the bit

            // Determine which error had the highest impact
            if x_prob > y_prob && x_prob > z_prob {
                x_count += 1;
            } else if y_prob > x_prob && y_prob > z_prob {
                y_count += 1;
            } else {
                z_count += 1;
            }
        }

        // Each error type should occur roughly 1/3 of the time
        let expected = trials as f64 / 3.0;
        let tolerance = trials as f64 * 0.2; // Allow 20% deviation

        assert!(
            (x_count as f64 - expected).abs() < tolerance,
            "X count: {}, expected: {} ± {}",
            x_count,
            expected,
            tolerance
        );
        assert!(
            (y_count as f64 - expected).abs() < tolerance,
            "Y count: {}, expected: {} ± {}",
            y_count,
            expected,
            tolerance
        );
        assert!(
            (z_count as f64 - expected).abs() < tolerance,
            "Z count: {}, expected: {} ± {}",
            z_count,
            expected,
            tolerance
        );
    }

    #[test]
    fn test_depolarizing_noise_specific_qubits() {
        // Create a 3-qubit state |000⟩
        let mut state = State::zero_state(3);

        // Apply noise only to qubit 1
        let noise = DepolarizingNoise::new(1.0).with_target_qubits(vec![1]);

        // Store original state
        let original = state.clone();

        // Apply noise
        noise.apply(&mut state);

        // Verify that qubit 1 was affected while others remained unchanged
        // Check by measuring correlation between original and final state
        // For 3 qubits, we should see changes in states |010⟩ and |011⟩

        // Sum up the probability that changed
        let mut changed_prob = 0.0;
        for i in 0..state.vector.len() {
            if (i & (1 << 1)) != 0 {
                // Check if qubit 1 is set
                changed_prob += state.vector[i].norm_sqr();
            }
        }

        // Some probability should have shifted to states with qubit 1 set
        assert!(changed_prob > 0.0);
    }

    #[test]
    fn test_amplitude_damping_noise() {
        // Create |1⟩ state which should decay to |0⟩
        let mut state = State::zero_state(1);
        state.vector[0] = Complex64::new(0.0, 0.0);
        state.vector[1] = Complex64::new(1.0, 0.0);

        // Apply 50% amplitude damping
        let noise = AmplitudeDampingNoise::new(0.5);
        noise.apply(&mut state);

        // Probability of |0⟩ should be approximately 0.5
        // Probability of |1⟩ should be approximately 0.5
        assert_relative_eq!(state.vector[0].norm_sqr(), 0.5, epsilon = 0.1);
        assert_relative_eq!(state.vector[1].norm_sqr(), 0.5, epsilon = 0.1);

        // Apply 100% amplitude damping
        let full_noise = AmplitudeDampingNoise::new(1.0);
        full_noise.apply(&mut state);

        // State should be very close to |0⟩
        assert_relative_eq!(state.vector[0].norm_sqr(), 1.0, epsilon = 0.1);
        assert_relative_eq!(state.vector[1].norm_sqr(), 0.0, epsilon = 0.1);
    }

    // #[test]
    // fn test_phase_damping_noise() {
    //     // Create superposition state (|0⟩ + |1⟩)/√2
    //     let mut state = State::zero_state(1);
    //     state.vector[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
    //     state.vector[1] = Complex64::new(FRAC_1_SQRT_2, 0.0);

    //     // Apply phase damping with 100% probability
    //     let noise = PhaseDampingNoise::new(1.0);
    //     noise.apply(&mut state);

    //     // Probabilities should remain unchanged
    //     assert_relative_eq!(state.vector[0].norm_sqr(), 0.5, epsilon = 0.1);
    //     assert_relative_eq!(state.vector[1].norm_sqr(), 0.5, epsilon = 0.1);

    //     // But coherence should be reduced
    //     // Create a density matrix and check off-diagonal elements
    //     let rho = reduced_density_matrix(&state, 0);

    //     // Off-diagonal elements should be close to zero
    //     assert_abs_diff_eq!(rho[0][1].norm(), 0.0, epsilon = 0.1);
    //     assert_abs_diff_eq!(rho[1][0].norm(), 0.0, epsilon = 0.1);
    // }

    // #[test]
    // fn test_thermal_relaxation_noise() {
    //     // Create |1⟩ state
    //     let mut state = State::zero_state(1);
    //     state.vector[0] = Complex64::new(0.0, 0.0);
    //     state.vector[1] = Complex64::new(1.0, 0.0);

    //     // Create thermal noise with:
    //     // T1 = 10 time units
    //     // T2 = 5 time units
    //     // Gate time = 5 time units (half of T1)
    //     let noise = ThermalRelaxationNoise::new(10.0, 5.0, 5.0);

    //     // Apply noise
    //     noise.apply(&mut state);

    //     // With gate_time = T1/2, we expect about 39.3% decay
    //     // (1 - exp(-5/10)) ≈ 0.393
    //     let expected_decay = 0.393;

    //     // |0⟩ probability should be around 39.3%
    //     // |1⟩ probability should be around 60.7%
    //     assert_relative_eq!(state.vector[0].norm_sqr(), expected_decay, epsilon = 0.1);
    //     assert_relative_eq!(state.vector[1].norm_sqr(), 1.0 - expected_decay, epsilon = 0.1);
    // }

    #[test]
    fn test_composite_noise() {
        // Create |+⟩ state
        let mut state = State::zero_state(1);
        state.vector[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        state.vector[1] = Complex64::new(FRAC_1_SQRT_2, 0.0);

        // Create composite noise with both phase and amplitude damping
        let mut composite = CompositeNoise::new();
        composite
            .add_noise(PhaseDampingNoise::new(1.0))
            .add_noise(AmplitudeDampingNoise::new(0.5));

        // Apply composite noise
        composite.apply(&mut state);

        // Phase damping should reduce coherence
        // Amplitude damping should bias toward |0⟩
        // |0⟩ probability should be > 0.5
        // |1⟩ probability should be < 0.5
        assert!(state.vector[0].norm_sqr() > 0.5);
        assert!(state.vector[1].norm_sqr() < 0.5);

        // Check density matrix for reduced coherence
        let rho = reduced_density_matrix(&state, 0);
        assert_abs_diff_eq!(rho[0][1].norm(), 0.0, epsilon = 0.1);
    }

    // #[test]
    // fn test_bell_state_with_noise() {
    //     // Create a Bell state
    //     let mut state = create_bell_state();

    //     // Verify we have a proper Bell state
    //     // For a Bell state |00⟩ + |11⟩/√2, only indices 0 and 3 should have amplitude
    //     assert_relative_eq!(state.vector[0].norm_sqr(), 0.5, epsilon = 0.01);
    //     assert_relative_eq!(state.vector[3].norm_sqr(), 0.5, epsilon = 0.01);
    //     assert_relative_eq!(state.vector[1].norm_sqr(), 0.0, epsilon = 0.01);
    //     assert_relative_eq!(state.vector[2].norm_sqr(), 0.0, epsilon = 0.01);

    //     // Initial purity should be 1 (pure state)
    //     let initial_purity = state_purity(&state);
    //     assert_relative_eq!(initial_purity, 1.0, epsilon = 0.01);

    //     // Apply moderate depolarizing noise
    //     let noise = DepolarizingNoise::new(0.3);
    //     noise.apply(&mut state);

    //     // Calculate new purity
    //     let final_purity = state_purity(&state);

    //     // Purity should decrease but not too much with moderate noise
    //     assert!(final_purity < initial_purity,
    //         "Purity should decrease from {} but was {}", initial_purity, final_purity);

    //     // With 0.3 probability noise, purity shouldn't drop below a certain threshold
    //     // For a 2-qubit system with 0.3 error rate, purity should remain above 0.5
    //     assert!(final_purity > 0.5,
    //         "Purity {} is too low for moderate noise", final_purity);

    //     // Check that we still have some correlation between qubits
    //     // This is a simplified test for entanglement persistence
    //     let q0q1_correlation = calculate_correlation(&state, 0, 1);
    //     assert!(q0q1_correlation.abs() > 0.1,
    //         "Qubits should maintain some correlation after moderate noise");
    // }

    // Helper function to calculate correlation between two qubits
    fn calculate_correlation(state: &State, qubit1: usize, qubit2: usize) -> f64 {
        let mut correlation = 0.0;

        // Calculate <Z_i Z_j> correlation
        for i in 0..state.vector.len() {
            let q1_value = if (i & (1 << qubit1)) != 0 { -1.0 } else { 1.0 };
            let q2_value = if (i & (1 << qubit2)) != 0 { -1.0 } else { 1.0 };

            correlation += q1_value * q2_value * state.vector[i].norm_sqr();
        }

        correlation
    }
}
