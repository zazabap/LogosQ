use logosq::states::State;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

// Import conditionally
#[cfg(feature = "parallel")]
use std::time::Instant;

mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        // Test with a non-normalized state
        let vector = Array1::from_vec(vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, 3.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(0.5, -0.5),
        ]);

        // Create two identical states for testing
        let state = State::new(vector.clone(), Some(2));

        // Get the normalized values for comparison
        let expected = vector.mapv(|x| x / vector.mapv(|x| x.norm_sqr()).sum().sqrt());

        // Check that normalization worked
        let norm = state.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-10, "Normalization failed");

        // Check against expected values
        for i in 0..state.vector.len() {
            assert!(
                (state.vector[i] - expected[i]).norm() < 1e-10,
                "Normalization result differs from expected at index {}",
                i
            );
        }

        // Test with an already normalized state
        let mut state_norm = State::zero_state(2);
        let norm_before = state_norm.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();
        state_norm.normalize(); // Call normalize again
        let norm_after = state_norm.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();

        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "Normalizing an already normalized state changed it"
        );

        // Test with zero state
        let zero_vector = Array1::zeros(4);
        let zero_state = State::new(zero_vector, Some(2));

        // State should remain zeros
        for i in 0..zero_state.vector.len() {
            assert!(
                zero_state.vector[i].norm() < 1e-10,
                "Zero state normalization failed"
            );
        }
    }

    #[test]
    fn test_inner_product_parallel() {
        // Create test states
        let state1 = State::zero_state(2); // |00⟩
        let state2 = State::one_state(2); // |11⟩
        let state3 = State::plus_state(2); // |++⟩
        let state4 = State::bell_state(); // (|00⟩ + |11⟩)/√2

        // Test orthogonal states
        let prod1 = state1.inner_product_parallel(&state2);
        assert!(
            prod1.norm() < 1e-10,
            "Orthogonal states should have zero inner product"
        );

        // Test with self (should be 1)
        let prod2 = state1.inner_product_parallel(&state1);
        assert!(
            (prod2 - Complex64::new(1.0, 0.0)).norm() < 1e-10,
            "Inner product with self should be 1"
        );

        // Test with plus state
        let prod3 = state1.inner_product_parallel(&state3);
        assert!(
            (prod3 - Complex64::new(0.5, 0.0)).norm() < 1e-10,
            "Inner product of |00⟩ with |++⟩ should be 0.5"
        );

        // Test Bell state
        let prod4 = state1.inner_product_parallel(&state4);
        assert!(
            (prod4 - Complex64::new(1.0 / SQRT_2, 0.0)).norm() < 1e-10,
            "Inner product of |00⟩ with Bell state should be 1/√2"
        );

        // Compare with manual calculation
        // Create custom states for comparison
        let v1 = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.5),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.0, -0.5),
        ]);

        let v2 = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        let custom1 = State::new(v1, Some(2));
        let custom2 = State::new(v2, Some(2));

        // Calculate inner product manually
        let mut manual_result = Complex64::new(0.0, 0.0);
        for i in 0..custom1.vector.len() {
            manual_result += custom1.vector[i].conj() * custom2.vector[i];
        }

        let parallel_result = custom1.inner_product_parallel(&custom2);

        assert!(
            (manual_result - parallel_result).norm() < 1e-10,
            "Parallel inner product doesn't match manual calculation"
        );
    }

    #[test]
    fn test_measure_shots_parallel() {
        // Test with a simple state - |0⟩
        let state0 = State::zero_state(1);
        let results0 = state0.measure_shots_parallel(100);

        // All measurements should be 0
        assert_eq!(results0.len(), 1, "Zero state should only measure to 0");
        assert_eq!(
            results0.get(&0),
            Some(&100),
            "Zero state measured incorrectly"
        );

        // Test with a simple state - |1⟩
        let state1 = State::one_state(1);
        let results1 = state1.measure_shots_parallel(100);

        // All measurements should be 1
        assert_eq!(results1.len(), 1, "One state should only measure to 1");
        assert_eq!(
            results1.get(&1),
            Some(&100),
            "One state measured incorrectly"
        );

        // Test with equal superposition - |+⟩
        let state_plus = State::plus_state(1);
        let n_shots = 1000; // More shots for statistical significance
        let results_plus = state_plus.measure_shots_parallel(n_shots);

        // Should be roughly equal measurements of 0 and 1
        assert!(
            results_plus.len() <= 2,
            "Plus state should only measure to 0 or 1"
        );

        if let (Some(&count0), Some(&count1)) = (results_plus.get(&0), results_plus.get(&1)) {
            let ratio = count0 as f64 / n_shots as f64;
            assert!(
                ratio > 0.3 && ratio < 0.7,
                "Measurements of |+⟩ state should be roughly 50/50, got {}/{}",
                count0,
                count1
            );
        }

        // Test with Bell state
        let bell = State::bell_state();
        let results_bell = bell.measure_shots_parallel(200);

        // Should only have measurements of |00⟩ and |11⟩
        for &idx in &[1, 2] {
            assert!(
                results_bell.get(&idx).is_none() || results_bell[&idx] == 0,
                "Bell state measured to unexpected state |{:02b}⟩",
                idx
            );
        }

        // Check that measurements are roughly distributed as expected
        let count0 = *results_bell.get(&0).unwrap_or(&0);
        let count3 = *results_bell.get(&3).unwrap_or(&0);
        let total = count0 + count3;

        assert!(
            count0 > 0 && count3 > 0,
            "Bell state should measure to both |00⟩ and |11⟩"
        );
        let ratio0 = count0 as f64 / total as f64;
        assert!(
            ratio0 > 0.3 && ratio0 < 0.7,
            "Bell state measurements should be roughly balanced"
        );
    }

    // Only run the performance test when the parallel feature is enabled
    #[test]
    fn test_parallel_performance() {
        // This is more of a benchmark than a test, but helps verify performance gain
        let n_qubits = 16; // 2^16 = 65536 amplitudes
        let large_state = State::plus_state(n_qubits);

        // Create two identical large states
        let mut state1 = large_state.clone();
        let state2 = large_state.clone();

        // Test with and without parallel feature
        let start = Instant::now();
        state1.normalize(); // Uses parallel implementation when feature is enabled
        let time = start.elapsed();

        // Just log the time, no assertions needed
        println!("Normalization of 2^{} state: {:?}", n_qubits, time);

        // Test inner product performance
        let start = Instant::now();
        let _ = state1.inner_product_parallel(&state2);
        let time = start.elapsed();

        println!("Inner product of 2^{} states: {:?}", n_qubits, time);
    }
}
