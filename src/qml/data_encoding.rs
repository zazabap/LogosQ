//! Data encoding strategies for quantum machine learning
//!
//! This module provides various methods to encode classical data into quantum states,
//! which is essential for quantum machine learning algorithms.

use crate::circuits::Circuit;
use std::f64::consts::PI;

/// Trait for data encoding into quantum circuits
pub trait DataEncoder {
    /// Encode classical data into a quantum circuit
    ///
    /// # Arguments
    /// * `circuit` - The quantum circuit to add encoding gates to
    /// * `data` - Classical data to encode (values typically in range [0, 1] or [-1, 1])
    fn encode(&self, circuit: &mut Circuit, data: &[f64]);

    /// Get the number of qubits required for encoding the given data size
    fn required_qubits(&self, data_size: usize) -> usize;
}

/// Angle encoding: encodes data as rotation angles
///
/// Each data point is encoded as a rotation angle on a separate qubit.
/// This is the most common encoding for variational quantum circuits.
///
/// For data x ∈ [-1, 1]:
///   - Apply RY(π * x) or RX(π * x) to each qubit
#[derive(Clone, Debug)]
pub struct AngleEncoding {
    /// Rotation axis: 'X', 'Y', or 'Z'
    pub axis: char,
    /// Scaling factor for the angles (default: π)
    pub scale: f64,
    /// Number of repetitions (data re-uploading)
    pub repetitions: usize,
}

impl Default for AngleEncoding {
    fn default() -> Self {
        Self {
            axis: 'Y',
            scale: PI,
            repetitions: 1,
        }
    }
}

impl AngleEncoding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_axis(mut self, axis: char) -> Self {
        assert!(
            axis == 'X' || axis == 'Y' || axis == 'Z',
            "Axis must be 'X', 'Y', or 'Z'"
        );
        self.axis = axis;
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_repetitions(mut self, repetitions: usize) -> Self {
        self.repetitions = repetitions;
        self
    }
}

impl DataEncoder for AngleEncoding {
    fn encode(&self, circuit: &mut Circuit, data: &[f64]) {
        let num_qubits = circuit.num_qubits();

        for _ in 0..self.repetitions {
            for (i, &value) in data.iter().enumerate() {
                if i >= num_qubits {
                    break;
                }
                let angle = self.scale * value;
                match self.axis {
                    'X' => circuit.rx(i, angle),
                    'Y' => circuit.ry(i, angle),
                    'Z' => circuit.rz(i, angle),
                    _ => unreachable!(),
                };
            }
        }
    }

    fn required_qubits(&self, data_size: usize) -> usize {
        data_size
    }
}

/// Amplitude encoding: encodes data as amplitudes of the quantum state
///
/// The classical data vector is normalized and encoded as the amplitudes
/// of a quantum state. This requires O(log n) qubits for n data points.
///
/// For data vector x = [x_0, x_1, ..., x_{n-1}]:
///   |ψ⟩ = Σᵢ (xᵢ/‖x‖) |i⟩
#[derive(Clone, Debug)]
pub struct AmplitudeEncoding;

impl AmplitudeEncoding {
    pub fn new() -> Self {
        Self
    }

    /// Compute the number of qubits needed for amplitude encoding
    fn compute_qubits(data_size: usize) -> usize {
        if data_size <= 1 {
            1
        } else {
            (data_size as f64).log2().ceil() as usize
        }
    }
}

impl Default for AmplitudeEncoding {
    fn default() -> Self {
        Self::new()
    }
}

impl DataEncoder for AmplitudeEncoding {
    fn encode(&self, circuit: &mut Circuit, data: &[f64]) {
        // Amplitude encoding requires state preparation which is more complex
        // For simplicity, we use an approximation via rotation gates
        // that works for small data sizes

        let num_qubits = circuit.num_qubits();
        let _data_size = data.len();

        // Normalize data
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 1e-10 {
            data.iter().map(|x| x / norm).collect()
        } else {
            vec![0.0; data.len()]
        };

        // For amplitude encoding, we use a simplified approach with RY gates
        // This is an approximation - full amplitude encoding requires more complex circuits
        for i in 0..num_qubits.min(normalized.len()) {
            // Map normalized value to angle
            // For values in [-1, 1], we use asin to get the angle
            let clamped = normalized[i].clamp(-1.0, 1.0);
            let angle = 2.0 * clamped.asin();
            circuit.ry(i, angle);
        }
    }

    fn required_qubits(&self, data_size: usize) -> usize {
        Self::compute_qubits(data_size)
    }
}

/// IQP (Instantaneous Quantum Polynomial) encoding
///
/// A more expressive encoding that uses entangling gates between qubits.
/// This encoding can capture correlations in the data.
#[derive(Clone, Debug)]
pub struct IQPEncoding {
    /// Depth of the encoding circuit
    pub depth: usize,
}

impl IQPEncoding {
    pub fn new(depth: usize) -> Self {
        Self { depth }
    }
}

impl Default for IQPEncoding {
    fn default() -> Self {
        Self { depth: 1 }
    }
}

impl DataEncoder for IQPEncoding {
    fn encode(&self, circuit: &mut Circuit, data: &[f64]) {
        let num_qubits = circuit.num_qubits();

        for _ in 0..self.depth {
            // Layer of Hadamards
            for q in 0..num_qubits {
                circuit.h(q);
            }

            // Single-qubit Z rotations with data
            for (i, &value) in data.iter().enumerate() {
                if i >= num_qubits {
                    break;
                }
                circuit.rz(i, PI * value);
            }

            // Entangling ZZ interactions with data products
            for i in 0..num_qubits.saturating_sub(1) {
                let i_data = if i < data.len() { data[i] } else { 0.0 };
                let j_data = if i + 1 < data.len() {
                    data[i + 1]
                } else {
                    0.0
                };
                circuit.rzz(i, i + 1, PI * i_data * j_data);
            }
        }
    }

    fn required_qubits(&self, data_size: usize) -> usize {
        data_size
    }
}

/// Creates a data encoding circuit with initial superposition
///
/// # Arguments
/// * `num_qubits` - Number of qubits
/// * `data` - Classical data to encode
/// * `encoder` - The encoding strategy to use
///
/// # Returns
/// A circuit with the encoded data
pub fn create_encoding_circuit<E: DataEncoder>(
    num_qubits: usize,
    data: &[f64],
    encoder: &E,
) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    encoder.encode(&mut circuit, data);
    circuit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_encoding() {
        let encoder = AngleEncoding::new();
        let mut circuit = Circuit::new(3);

        encoder.encode(&mut circuit, &[0.5, -0.3, 0.8]);

        assert_eq!(circuit.num_operations(), 3);
    }

    #[test]
    fn test_amplitude_encoding() {
        let encoder = AmplitudeEncoding::new();
        let mut circuit = Circuit::new(3);

        encoder.encode(&mut circuit, &[0.5, 0.5, 0.5]);

        assert!(circuit.num_operations() > 0);
    }

    #[test]
    fn test_iqp_encoding() {
        let encoder = IQPEncoding::new(2);
        let mut circuit = Circuit::new(3);

        encoder.encode(&mut circuit, &[0.1, 0.2, 0.3]);

        // Should have H gates, RZ gates, and RZZ gates
        assert!(circuit.num_operations() > 3);
    }

    #[test]
    fn test_required_qubits() {
        let angle = AngleEncoding::new();
        assert_eq!(angle.required_qubits(4), 4);

        let amplitude = AmplitudeEncoding::new();
        assert_eq!(amplitude.required_qubits(4), 2); // log2(4) = 2
        assert_eq!(amplitude.required_qubits(5), 3); // ceil(log2(5)) = 3
    }
}
