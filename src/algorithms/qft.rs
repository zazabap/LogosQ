// Quantum Fourier Transform implementation

use crate::circuits::Circuit;
use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use crate::algorithms::cobra::cobra_apply;
use crate::algorithms::qftkernels::dit::{
    fft_dit_64_chunk_n_simd,
    fft_dit_chunk_16_simd_f64, fft_dit_chunk_2,
    fft_dit_chunk_32_simd_f64, fft_dit_chunk_4_simd_f64, fft_dit_chunk_64_simd_f64,
    fft_dit_chunk_8_simd_f64,
};

/// Reverse is for running the Inverse Fast Fourier Transform (IFFT)
/// Forward is for running the regular FFT
#[derive(Copy, Clone)]
pub enum Direction {
    /// Leave the exponent term in the twiddle factor alone
    Forward = 1,
    /// Multiply the exponent term in the twiddle factor by -1
    Reverse = -1,
}

/// Simple planner for DIT FFT algorithm
pub struct DitPlanner {
    /// Twiddles for each stage that needs them (stages with chunk_size > 64)
    pub stage_twiddles: Vec<(Vec<f64>, Vec<f64>)>,
    /// The direction of the FFT
    pub direction: Direction,
}

impl DitPlanner {
    /// Create a new DIT planner for an FFT of size `num_points`
    pub fn new(num_points: usize, direction: Direction) -> Self {
        assert!(num_points > 0 && num_points.is_power_of_two());

        let log_n = num_points.ilog2() as usize;
        let mut stage_twiddles = Vec::new();

        // Pre-compute twiddles for each stage that needs them
        for stage in 0..log_n {
            let dist = 1 << stage;
            let chunk_size = dist << 1;

            // Only stages with chunk_size > 64 need twiddles (we have SIMD kernels up to 64)
            if chunk_size > 64 {
                let mut twiddles_re = vec![0.0f64; dist];
                let mut twiddles_im = vec![0.0f64; dist];

                let angle_mult = -2.0 * std::f64::consts::PI / chunk_size as f64;
                for k in 0..dist {
                    let angle = angle_mult * k as f64;
                    twiddles_re[k] = angle.cos();
                    twiddles_im[k] = angle.sin();
                }

                stage_twiddles.push((twiddles_re, twiddles_im));
            }
        }

        Self {
            stage_twiddles,
            direction,
        }
    }
}

/// Creates a Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Quantum Fourier Transform");

    // QFT implementation
    for i in 0..num_qubits {
        // Hadamard gate on the current qubit
        circuit.h(i);

        // Controlled phase rotations
        for j in (i + 1)..num_qubits {
            let angle = PI / (1 << (j - i)) as f64;
            controlled_phase(&mut circuit, i, j, angle);
        }
    }

    // Swap qubits to match classical FFT ordering
    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    circuit
}

/// Creates an inverse Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_inverse_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Inverse Quantum Fourier Transform");

    // Swap qubits first for inverse QFT
    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    // Apply inverse QFT
    for i in (0..num_qubits).rev() {
        // Controlled rotations in reverse with negated angles
        for j in (i + 1)..num_qubits {
            let angle = -PI / ((1 << (j - i)) as f64);
            controlled_phase(&mut circuit, i, j, angle);
        }

        // Hadamard on the current qubit
        circuit.h(i);
    }

    circuit
}

/// Applies the QFT to a quantum state
pub fn apply(state: &mut State) {
    println!("Applying QFT to state with {} qubits", state.num_qubits);
    let num_qubits = state.num_qubits;
    let circuit = create_circuit(num_qubits);
    circuit.execute(state);
}

/// Applies the inverse QFT to a quantum state
pub fn apply_inverse(state: &mut State) {
    let num_qubits = state.num_qubits;
    let circuit = create_inverse_circuit(num_qubits);
    circuit.execute(state);
}

/// Options to tune to improve performance depending on the hardware and input size.

/// Calling FFT routines without specifying options will automatically select reasonable defaults
/// depending on the input size and other factors.
///
/// You only need to tune these options if you are trying to squeeze maximum performance
/// out of a known hardware platform that you can benchmark at varying input sizes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Options {
    pub multithreaded_bit_reversal: bool,
    pub dif_perform_bit_reversal: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            multithreaded_bit_reversal: false,
            dif_perform_bit_reversal: true,
        }
    }
}

impl Options {
    pub fn guess_options(input_size: usize) -> Options {
        let mut options = Options::default();
        let n: usize = input_size.ilog2() as usize;
        options.multithreaded_bit_reversal = n >= 22;
        options
    }
}

/// FFT implementation using DIT algorithm with options
pub fn fft_64_dit_with_opts(reals: &mut [f64], imags: &mut [f64], direction: Direction, opts: &Options) {
    assert_eq!(reals.len(), imags.len());
    assert!(reals.len().is_power_of_two());
    let n = reals.len();
    let log_n = n.ilog2() as usize;


    println!("FFT input: n={}, first few values: [{:.4}, {:.4}...]", n, reals[0], reals[1]);

    // DIT requires bit-reversed input
    if opts.multithreaded_bit_reversal {
        std::thread::scope(|s| {
            s.spawn(|| cobra_apply(reals, log_n));
            s.spawn(|| cobra_apply(imags, log_n));
        });
    } else {
        cobra_apply(reals, log_n);
        cobra_apply(imags, log_n);
    }

    println!("After bit reversal: [{:.4}, {:.4}...]", reals[0], reals[1]);


    // Create planner for twiddle factors
    let planner = DitPlanner::new(n, direction);

    // Handle inverse FFT
    if let Direction::Reverse = planner.direction {
        for z_im in imags.iter_mut() {
            *z_im = -*z_im;
        }
    }

    let mut stage_twiddle_idx = 0;
    for stage in 0..log_n {
        let dist = 1 << stage;
        let chunk_size = dist << 1;

        if chunk_size == 2 {
            fft_dit_chunk_2(reals, imags);
        } else if chunk_size == 4 {
            fft_dit_chunk_4_simd_f64(reals, imags);
        } else if chunk_size == 8 {
            fft_dit_chunk_8_simd_f64(reals, imags);
        } else if chunk_size == 16 {
            fft_dit_chunk_16_simd_f64(reals, imags);
        } else if chunk_size == 32 {
            fft_dit_chunk_32_simd_f64(reals, imags);
        } else if chunk_size == 64 {
            fft_dit_chunk_64_simd_f64(reals, imags);
        } else {
            // For larger chunks, use general kernel with twiddles from planner
            let (twiddles_re, twiddles_im) = &planner.stage_twiddles[stage_twiddle_idx];
            fft_dit_64_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
            stage_twiddle_idx += 1;
        }
    }

    // Scaling for inverse transform
    if let Direction::Reverse = planner.direction {
        let scaling_factor = 1.0 / n as f64;
        for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
            *z_re *= scaling_factor;
            *z_im *= -scaling_factor;
        }
    }
}

/// Core function that applies QFT using PhastFFT library
pub fn apply_(state: &mut State) {
    // Convert state vector to format suitable for FFT
    let n = state.vector.len();
    assert!(n.is_power_of_two(), "State vector length must be a power of two");
    
    // Extract real and imaginary parts from the quantum state
    let mut reals: Vec<f64> = state.vector.iter().map(|c| c.re).collect();
    let mut imags: Vec<f64> = state.vector.iter().map(|c| c.im).collect();

    // Apply FFT in-place with optimized options
    let opts = Options::guess_options(n);
    fft_64_dit_with_opts(&mut reals, &mut imags, Direction::Forward, &opts);

    // Update state vector with transformed values
    for i in 0..n {
        state.vector[i] = Complex64::new(reals[i], imags[i]);
    }
}

/// Apply inverse QFT directly using FFT algorithm
pub fn apply_inverse_(state: &mut State) {
    // Convert state vector to format suitable for FFT
    let n = state.vector.len();
    assert!(n.is_power_of_two(), "State vector length must be a power of two");
    
    // Extract real and imaginary parts from the quantum state
    let mut reals: Vec<f64> = state.vector.iter().map(|c| c.re).collect();
    let mut imags: Vec<f64> = state.vector.iter().map(|c| c.im).collect();

    // Apply inverse FFT in-place with optimized options
    let opts = Options::guess_options(n);
    fft_64_dit_with_opts(&mut reals, &mut imags, Direction::Reverse, &opts);
    
    // Update state vector with transformed values
    for i in 0..n {
        state.vector[i] = Complex64::new(reals[i], imags[i]);
    }
}

pub fn controlled_phase(circuit: &mut Circuit, control: usize, target: usize, angle: f64) {
    // Direct implementation of controlled phase gate
    let full_dim = 1 << circuit.num_qubits;
    let mut matrix = Array2::zeros((full_dim, full_dim));

    // Convert qubit indices to bit positions (MSB = 0, LSB = n-1)
    let control_bit = circuit.num_qubits - 1 - control;
    let target_bit = circuit.num_qubits - 1 - target;

    // Fill the matrix in parallel for large matrices
    if full_dim > 256 {
        // Parallel matrix filling for large matrices
        // Create a vector of updates in parallel
        let updates: Vec<(usize, Complex64)> = (0..full_dim)
            .into_par_iter()
            .map(|i| {
                // Extract control and target bits
                let control_val = (i >> control_bit) & 1;
                let target_val = (i >> target_bit) & 1;

                // Calculate the phase
                let phase = if control_val == 1 && target_val == 1 {
                    Complex64::from_polar(1.0, angle)
                } else {
                    Complex64::new(1.0, 0.0)
                };

                (i, phase)
            })
            .collect();

        // Apply updates sequentially
        for (i, phase) in updates {
            matrix[[i, i]] = phase;
        }
    } else {
        // Sequential for small matrices to avoid parallelism overhead
        for i in 0..full_dim {
            // Extract control and target bits
            let control_val = (i >> control_bit) & 1;
            let target_val = (i >> target_bit) & 1;

            // Apply phase only when both control and target are 1
            let phase = if control_val == 1 && target_val == 1 {
                Complex64::from_polar(1.0, angle)
            } else {
                Complex64::new(1.0, 0.0)
            };

            matrix[[i, i]] = phase;
        }
    }

    circuit.add_matrix_gate(
        matrix,
        (0..circuit.num_qubits).collect(),
        &format!("CP({:.4})", angle),
    );
}
