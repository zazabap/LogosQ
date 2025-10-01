use ndarray::Array2;
use num_complex::Complex64;

// Import rayon conditionally
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Helper function for tensor product of matrices
/// Automatically uses parallel execution when the "parallel" feature is enabled
/// and matrices are large enough to benefit from it
pub fn tensor_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let result_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    use std::sync::{Arc, Mutex};
    let result = Arc::new(Mutex::new(Array2::zeros(result_shape)));

    // Use parallel execution for larger matrices where it's beneficial
    #[cfg(feature = "parallel")]
    if result_shape[0] * result_shape[1] > 256 {
        // Process each row in parallel
        (0..result_shape[0]).into_par_iter().for_each(|i| {
            let a_i = i / b_shape[0];
            for j in 0..result_shape[1] {
                let a_j = j / b_shape[1];
                let b_i = i % b_shape[0];
                let b_j = j % b_shape[1];
                let mut result_lock = result.lock().unwrap();
                result_lock[[i, j]] = a[[a_i, a_j]] * b[[b_i, b_j]];
            }
        });

        return Arc::try_unwrap(result).unwrap().into_inner().unwrap();
    }

    // Sequential implementation for smaller matrices or when parallel feature is disabled
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    result.lock().unwrap()[[i * b_shape[0] + k, j * b_shape[1] + l]] =
                        a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}
