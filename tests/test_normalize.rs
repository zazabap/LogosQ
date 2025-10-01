use logosq::states::State;
use ndarray::Array1;
use num_complex::Complex64;
use std::time::Instant;

// Define a larger test case for performance comparison
const LARGE_STATE_QUBITS: usize = 20; // 2^20 amplitudes - large enough to see parallelism benefits

mod tests {
    use super::*;

    // Basic normalization test that works with both parallel and non-parallel builds
    #[test]
    fn test_normalize_correctness() {
        // Create a non-normalized state
        let vector = Array1::from_vec(vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, 3.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(0.5, -0.5),
        ]);

        let state = State::new(vector.clone(), Some(2));

        // Calculate expected normalized values manually
        let norm = vector.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let expected = vector.mapv(|x| x / norm);

        // Check that normalization worked
        let state_norm = state.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert!((state_norm - 1.0).abs() < 1e-10, "Normalization failed");

        // Compare each element with expected value
        for i in 0..state.vector.len() {
            assert!(
                (state.vector[i] - expected[i]).norm() < 1e-10,
                "Normalized value differs from expected at index {}",
                i
            );
        }

        // Test edge cases

        // 1. Already normalized state
        let mut state_norm = State::zero_state(2);
        let norm_before = state_norm.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();
        state_norm.normalize(); // Normalize again
        let norm_after = state_norm.vector.iter().map(|c| c.norm_sqr()).sum::<f64>();

        assert!(
            (norm_before - norm_after).abs() < 1e-10,
            "Normalizing an already normalized state changed it"
        );

        // 2. Zero state
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

    // Performance test for parallel feature
    #[cfg(feature = "parallel")]
    #[test]
    fn test_normalize_parallel_performance() {
        println!("RUNNING WITH PARALLEL FEATURE ENABLED");

        // Create a large state for testing parallelism
        let large_state = State::plus_state(LARGE_STATE_QUBITS);
        println!(
            "Created state with {} qubits ({} amplitudes)",
            LARGE_STATE_QUBITS,
            large_state.vector.len()
        );

        // Measure performance
        let start = Instant::now();
        let mut state = large_state.clone();
        state.normalize();
        let duration = start.elapsed();

        println!("Normalization with parallel feature: {:?}", duration);

        // We don't assert anything about performance, just report it
        assert!(
            state.vector.iter().map(|c| c.norm_sqr()).sum::<f64>() - 1.0 < 1e-10,
            "Normalized state should have norm 1.0"
        );
    }

    // Feature detection test - just reports which implementation is active
    #[test]
    fn test_feature_detection() {
        // This test just reports which feature is active
        #[cfg(feature = "parallel")]
        println!("Test running with PARALLEL feature ENABLED");

        #[cfg(not(feature = "parallel"))]
        println!("Test running with PARALLEL feature DISABLED");

        // Create a state and normalize it
        let mut state = State::plus_state(4);
        let start = Instant::now();
        state.normalize();
        let duration = start.elapsed();

        println!("Normalization took: {:?}", duration);
    }

    // This test compares the normalize implementation to a simple manual normalization
    #[test]
    fn test_normalize_against_manual() {
        // Create random-like state vector
        let mut vector = Array1::zeros(16); // 4 qubits
        for i in 0..16 {
            let re = ((i * 17) % 100) as f64 / 50.0 - 1.0;
            let im = ((i * 23) % 100) as f64 / 50.0 - 1.0;
            vector[i] = Complex64::new(re, im);
        }

        // Create a state with this vector
        let state = State::new(vector.clone(), Some(4));

        // Create a manually normalized version for comparison
        let norm = vector.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let manual_normalized = vector.mapv(|x| x / norm);

        // Compare results
        for i in 0..state.vector.len() {
            assert!(
                (state.vector[i] - manual_normalized[i]).norm() < 1e-10,
                "State normalization differs from manual at index {}: {:?} vs {:?}",
                i,
                state.vector[i],
                manual_normalized[i]
            );
        }
    }
}
