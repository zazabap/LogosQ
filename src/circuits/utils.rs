use ndarray::Array2;
use num_complex::Complex64;

/// Helper function for tensor product of matrices
pub fn tensor_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let result_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    let mut result = Array2::zeros(result_shape);

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    result[[i * b_shape[0] + k, j * b_shape[1] + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}