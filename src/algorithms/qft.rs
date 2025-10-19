// Quantum Fourier Transform implementation - optimized with FFT

use crate::circuits::Circuit;
use crate::gates::two_qubit::CPhaseGate;
use crate::states::State;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Creates a Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Quantum Fourier Transform");

    // Gate-based QFT (kept for debugging / compatibility)
    for i in 0..num_qubits {
        circuit.h(i);
        for j in (i + 1)..num_qubits {
            let angle = PI / (1 << (j - i)) as f64;
            controlled_phase(&mut circuit, i, j, angle);
        }
    }

    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    circuit
}

/// Creates an inverse Quantum Fourier Transform circuit for the specified number of qubits.
pub fn create_inverse_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits).with_name("Inverse Quantum Fourier Transform");

    for i in 0..num_qubits / 2 {
        circuit.swap(i, num_qubits - i - 1);
    }

    for i in (0..num_qubits).rev() {
        for j in (i + 1)..num_qubits {
            let angle = -PI / ((1 << (j - i)) as f64);
            controlled_phase(&mut circuit, i, j, angle);
        }
        circuit.h(i);
    }

    circuit
}

/// Applies the QFT to a quantum state (optimized using an FFT on the amplitude vector).
/// This replaces running the gate sequence and runs in O(N log N) time.
pub fn apply(state: &mut State) {
    let n = state.num_qubits;
    let len = 1usize << n;

    // Ensure state vector has contiguous slice
    let buf = state
        .vector
        .as_slice_mut()
        .expect("State vector must be contiguous for FFT optimization");

    // Use inverse FFT to compute QFT: QFT(|x>)_k = 1/sqrt(N) * sum_x exp(2π i x k / N) * a_x
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(len);
    // rustfft uses num_complex::Complex<f64> which is compatible with Complex64
    fft.process(buf);

    // Normalize by 1/sqrt(N) to match quantum normalization
    let scale = 1.0 / (len as f64).sqrt();
    for v in buf.iter_mut() {
        *v = *v * scale;
    }
}

/// Applies the inverse QFT to a quantum state (optimized using an FFT).
pub fn apply_inverse(state: &mut State) {
    let n = state.num_qubits;
    let len = 1usize << n;

    let buf = state
        .vector
        .as_slice_mut()
        .expect("State vector must be contiguous for FFT optimization");

    // Use forward FFT to compute inverse QFT: QFT^{-1} = 1/sqrt(N) * forward FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);
    fft.process(buf);

    let scale = 1.0 / (len as f64).sqrt();
    for v in buf.iter_mut() {
        *v = *v * scale;
    }
}

/// Helper function to apply a controlled phase rotation.
/// Adds an optimized CPhaseGate instead of building a full matrix.
pub fn controlled_phase(circuit: &mut Circuit, control: usize, target: usize, angle: f64) {
    let gate = CPhaseGate {
        control,
        target,
        angle,
        num_qubits: circuit.num_qubits,
    };

    circuit.add_operation(gate, vec![control, target], &format!("CP({:.4})", angle));
}
