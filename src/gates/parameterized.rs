//! Parameterized quantum gates for variational quantum circuits
//!
//! These gates have adjustable parameters (angles) that can be optimized
//! for VQE, QAOA, and other variational algorithms.
//!
//! All gates directly manipulate the state vector for maximum performance,
//! similar to the optimized gate implementations.

use super::*;

// ============================================================================
// ENUMS AND HELPERS
// ============================================================================

/// Rotation axis for parameterized single-qubit gates
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

// ============================================================================
// SINGLE-QUBIT PARAMETERIZED ROTATIONS
// ============================================================================

/// Parameterized RX gate: rotation around X-axis
///
/// Matrix form: RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
#[derive(Clone)]
pub struct RXGate {
    pub qubit: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RXGate {
    /// Create a new RX gate
    pub fn new(qubit: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RXGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;

                let amp0 = vector_slice[idx0];
                let amp1 = vector_slice[idx1];

                // RX matrix: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
                vector_slice[idx0] =
                    Complex64::new(cos_half, 0.0) * amp0 + Complex64::new(0.0, -sin_half) * amp1;
                vector_slice[idx1] =
                    Complex64::new(0.0, -sin_half) * amp0 + Complex64::new(cos_half, 0.0) * amp1;
            }
        }
    }
}

/// Parameterized RY gate: rotation around Y-axis
///
/// Matrix form: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
#[derive(Clone)]
pub struct RYGate {
    pub qubit: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RYGate {
    /// Create a new RY gate
    pub fn new(qubit: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RYGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;

                let amp0 = vector_slice[idx0];
                let amp1 = vector_slice[idx1];

                // RY matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                vector_slice[idx0] =
                    Complex64::new(cos_half, 0.0) * amp0 + Complex64::new(-sin_half, 0.0) * amp1;
                vector_slice[idx1] =
                    Complex64::new(sin_half, 0.0) * amp0 + Complex64::new(cos_half, 0.0) * amp1;
            }
        }
    }
}

/// Parameterized RZ gate: rotation around Z-axis
///
/// Matrix form: RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
#[derive(Clone)]
pub struct RZGate {
    pub qubit: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RZGate {
    /// Create a new RZ gate
    pub fn new(qubit: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RZGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let phase_0 = Complex64::from_polar(1.0, -self.theta / 2.0);
        let phase_1 = Complex64::from_polar(1.0, self.theta / 2.0);

        let qubit_bit = n - 1 - self.qubit;
        let mask = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & mask) == 0 {
                vector_slice[i] *= phase_0;
            } else {
                vector_slice[i] *= phase_1;
            }
        }
    }
}

/// Parameterized Phase gate: P(θ) = diag(1, e^(iθ))
///
/// This is equivalent to RZ but without the global phase
#[derive(Clone)]
pub struct PhaseGate {
    pub qubit: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl PhaseGate {
    /// Create a new Phase gate
    pub fn new(qubit: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit,
            theta,
            num_qubits,
        }
    }
}

impl Gate for PhaseGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let phase = Complex64::from_polar(1.0, self.theta);

        let qubit_bit = n - 1 - self.qubit;
        let mask = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & mask) != 0 {
                vector_slice[i] *= phase;
            }
        }
    }
}

/// General U3 gate: U3(θ, φ, λ)
///
/// Matrix form:
/// U3(θ,φ,λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)],
///              [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
///
/// This is a universal single-qubit gate that can represent any SU(2) operation
#[derive(Clone)]
pub struct U3Gate {
    pub qubit: usize,
    pub theta: f64,
    pub phi: f64,
    pub lambda: f64,
    pub num_qubits: usize,
}

impl U3Gate {
    /// Create a new U3 gate
    pub fn new(qubit: usize, theta: f64, phi: f64, lambda: f64, num_qubits: usize) -> Self {
        Self {
            qubit,
            theta,
            phi,
            lambda,
            num_qubits,
        }
    }
}

impl Gate for U3Gate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        // U3 matrix elements
        let m00 = Complex64::new(cos_half, 0.0);
        let m01 = -Complex64::from_polar(sin_half, self.lambda);
        let m10 = Complex64::from_polar(sin_half, self.phi);
        let m11 = Complex64::from_polar(cos_half, self.phi + self.lambda);

        let qubit_bit = n - 1 - self.qubit;
        let stride = 1 << qubit_bit;
        let full_dim = 1 << n;

        let vector_slice = state.vector.as_slice_mut().unwrap();

        for base in (0..full_dim).step_by(stride * 2) {
            for offset in 0..stride {
                let idx0 = base + offset;
                let idx1 = idx0 + stride;

                let amp0 = vector_slice[idx0];
                let amp1 = vector_slice[idx1];

                vector_slice[idx0] = m00 * amp0 + m01 * amp1;
                vector_slice[idx1] = m10 * amp0 + m11 * amp1;
            }
        }
    }
}

// ============================================================================
// TWO-QUBIT PARAMETERIZED CONTROLLED ROTATIONS
// ============================================================================

/// Parameterized Controlled-RX gate
///
/// Applies RX(θ) to target qubit when control qubit is |1⟩
#[derive(Clone)]
pub struct CRXGate {
    pub control: usize,
    pub target: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl CRXGate {
    /// Create a new CRX gate
    pub fn new(control: usize, target: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            control,
            target,
            theta,
            num_qubits,
        }
    }
}

impl Gate for CRXGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let control_bit = n - 1 - self.control;
        let target_bit = n - 1 - self.target;

        let control_mask = 1 << control_bit;
        let target_mask = 1 << target_bit;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            // Only apply when control is |1⟩ and target is |0⟩
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;

                let amp0 = vector_slice[i];
                let amp1 = vector_slice[j];

                vector_slice[i] =
                    Complex64::new(cos_half, 0.0) * amp0 + Complex64::new(0.0, -sin_half) * amp1;
                vector_slice[j] =
                    Complex64::new(0.0, -sin_half) * amp0 + Complex64::new(cos_half, 0.0) * amp1;
            }
        }
    }
}

/// Parameterized Controlled-RY gate
///
/// Applies RY(θ) to target qubit when control qubit is |1⟩
#[derive(Clone)]
pub struct CRYGate {
    pub control: usize,
    pub target: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl CRYGate {
    /// Create a new CRY gate
    pub fn new(control: usize, target: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            control,
            target,
            theta,
            num_qubits,
        }
    }
}

impl Gate for CRYGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let control_bit = n - 1 - self.control;
        let target_bit = n - 1 - self.target;

        let control_mask = 1 << control_bit;
        let target_mask = 1 << target_bit;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;

                let amp0 = vector_slice[i];
                let amp1 = vector_slice[j];

                vector_slice[i] =
                    Complex64::new(cos_half, 0.0) * amp0 + Complex64::new(-sin_half, 0.0) * amp1;
                vector_slice[j] =
                    Complex64::new(sin_half, 0.0) * amp0 + Complex64::new(cos_half, 0.0) * amp1;
            }
        }
    }
}

/// Parameterized Controlled-RZ gate
///
/// Applies RZ(θ) to target qubit when control qubit is |1⟩
#[derive(Clone)]
pub struct CRZGate {
    pub control: usize,
    pub target: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl CRZGate {
    /// Create a new CRZ gate
    pub fn new(control: usize, target: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            control,
            target,
            theta,
            num_qubits,
        }
    }
}

impl Gate for CRZGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let control_bit = n - 1 - self.control;
        let target_bit = n - 1 - self.target;

        let control_mask = 1 << control_bit;
        let target_mask = 1 << target_bit;

        let phase_0 = Complex64::from_polar(1.0, -self.theta / 2.0);
        let phase_1 = Complex64::from_polar(1.0, self.theta / 2.0);

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & control_mask) != 0 {
                if (i & target_mask) == 0 {
                    vector_slice[i] *= phase_0;
                } else {
                    vector_slice[i] *= phase_1;
                }
            }
        }
    }
}

/// Parameterized Controlled-Phase gate
///
/// Applies phase e^(iθ) when both qubits are |1⟩
#[derive(Clone)]
pub struct CPhaseParamGate {
    pub control: usize,
    pub target: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl CPhaseParamGate {
    /// Create a new parameterized controlled-phase gate
    pub fn new(control: usize, target: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            control,
            target,
            theta,
            num_qubits,
        }
    }
}

impl Gate for CPhaseParamGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let control_bit = n - 1 - self.control;
        let target_bit = n - 1 - self.target;

        let control_mask = 1 << control_bit;
        let target_mask = 1 << target_bit;
        let both_mask = control_mask | target_mask;

        let phase = Complex64::from_polar(1.0, self.theta);

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            if (i & both_mask) == both_mask {
                vector_slice[i] *= phase;
            }
        }
    }
}

// ============================================================================
// TWO-QUBIT ENTANGLING GATES (for QAOA, Ising models, etc.)
// ============================================================================

/// Parameterized RXX gate: exp(-i * theta * XX / 2)
///
/// This gate is useful for QAOA and simulating XX interactions
#[derive(Clone)]
pub struct RXXGate {
    pub qubit1: usize,
    pub qubit2: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RXXGate {
    /// Create a new RXX gate
    pub fn new(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit1,
            qubit2,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RXXGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let bit1 = n - 1 - self.qubit1;
        let bit2 = n - 1 - self.qubit2;

        let mask1 = 1 << bit1;
        let mask2 = 1 << bit2;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        // Process each basis state
        for i in 0..full_dim {
            let bit1_val = (i & mask1) != 0;
            let bit2_val = (i & mask2) != 0;

            if !bit1_val && !bit2_val {
                // |00⟩ ↔ |11⟩
                let j = i | mask1 | mask2;

                if i < j {
                    let amp_i = vector_slice[i];
                    let amp_j = vector_slice[j];

                    vector_slice[i] = Complex64::new(cos_half, 0.0) * amp_i
                        + Complex64::new(0.0, -sin_half) * amp_j;
                    vector_slice[j] = Complex64::new(0.0, -sin_half) * amp_i
                        + Complex64::new(cos_half, 0.0) * amp_j;
                }
            } else if bit1_val && !bit2_val {
                // |10⟩ ↔ |01⟩
                let j = (i & !mask1) | mask2;

                if i < j {
                    let amp_i = vector_slice[i];
                    let amp_j = vector_slice[j];

                    vector_slice[i] = Complex64::new(cos_half, 0.0) * amp_i
                        + Complex64::new(0.0, -sin_half) * amp_j;
                    vector_slice[j] = Complex64::new(0.0, -sin_half) * amp_i
                        + Complex64::new(cos_half, 0.0) * amp_j;
                }
            }
        }
    }
}

/// Parameterized RYY gate: exp(-i * theta * YY / 2)
///
/// This gate is useful for QAOA and simulating YY interactions
#[derive(Clone)]
pub struct RYYGate {
    pub qubit1: usize,
    pub qubit2: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RYYGate {
    /// Create a new RYY gate
    pub fn new(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit1,
            qubit2,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RYYGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        let bit1 = n - 1 - self.qubit1;
        let bit2 = n - 1 - self.qubit2;

        let mask1 = 1 << bit1;
        let mask2 = 1 << bit2;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            let bit1_val = (i & mask1) != 0;
            let bit2_val = (i & mask2) != 0;

            if !bit1_val && !bit2_val {
                // |00⟩ ↔ |11⟩ (with positive sign)
                let j = i | mask1 | mask2;

                if i < j {
                    let amp_i = vector_slice[i];
                    let amp_j = vector_slice[j];

                    vector_slice[i] = Complex64::new(cos_half, 0.0) * amp_i
                        + Complex64::new(0.0, sin_half) * amp_j;
                    vector_slice[j] = Complex64::new(0.0, sin_half) * amp_i
                        + Complex64::new(cos_half, 0.0) * amp_j;
                }
            } else if bit1_val && !bit2_val {
                // |10⟩ ↔ |01⟩ (with negative sign)
                let j = (i & !mask1) | mask2;

                if i < j {
                    let amp_i = vector_slice[i];
                    let amp_j = vector_slice[j];

                    vector_slice[i] = Complex64::new(cos_half, 0.0) * amp_i
                        + Complex64::new(0.0, -sin_half) * amp_j;
                    vector_slice[j] = Complex64::new(0.0, -sin_half) * amp_i
                        + Complex64::new(cos_half, 0.0) * amp_j;
                }
            }
        }
    }
}

/// Parameterized RZZ gate: exp(-i * theta * ZZ / 2)
///
/// This is one of the most common gates in VQE and QAOA
#[derive(Clone)]
pub struct RZZGate {
    pub qubit1: usize,
    pub qubit2: usize,
    pub theta: f64,
    pub num_qubits: usize,
}

impl RZZGate {
    /// Create a new RZZ gate
    pub fn new(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> Self {
        Self {
            qubit1,
            qubit2,
            theta,
            num_qubits,
        }
    }
}

impl Gate for RZZGate {
    fn apply(&self, state: &mut State) {
        let n = state.num_qubits;
        assert_eq!(n, self.num_qubits, "State qubit count mismatch");

        let bit1 = n - 1 - self.qubit1;
        let bit2 = n - 1 - self.qubit2;

        let mask1 = 1 << bit1;
        let mask2 = 1 << bit2;

        let full_dim = 1 << n;
        let vector_slice = state.vector.as_slice_mut().unwrap();

        for i in 0..full_dim {
            let bit1_val = (i & mask1) != 0;
            let bit2_val = (i & mask2) != 0;

            // Phase depends on parity of the two qubits
            let phase = if bit1_val == bit2_val {
                // |00⟩ or |11⟩: same parity (eigenvalue +1)
                Complex64::from_polar(1.0, -self.theta / 2.0)
            } else {
                // |01⟩ or |10⟩: different parity (eigenvalue -1)
                Complex64::from_polar(1.0, self.theta / 2.0)
            };

            vector_slice[i] *= phase;
        }
    }
}

// ============================================================================
// CONVENIENCE CONSTRUCTORS
// ============================================================================

/// Create an RX gate
pub fn rx(qubit: usize, theta: f64, num_qubits: usize) -> RXGate {
    RXGate::new(qubit, theta, num_qubits)
}

/// Create an RY gate
pub fn ry(qubit: usize, theta: f64, num_qubits: usize) -> RYGate {
    RYGate::new(qubit, theta, num_qubits)
}

/// Create an RZ gate
pub fn rz(qubit: usize, theta: f64, num_qubits: usize) -> RZGate {
    RZGate::new(qubit, theta, num_qubits)
}

/// Create a Phase gate
pub fn phase(qubit: usize, theta: f64, num_qubits: usize) -> PhaseGate {
    PhaseGate::new(qubit, theta, num_qubits)
}

/// Create a U3 gate
pub fn u3(qubit: usize, theta: f64, phi: f64, lambda: f64, num_qubits: usize) -> U3Gate {
    U3Gate::new(qubit, theta, phi, lambda, num_qubits)
}

/// Create a CRX gate
pub fn crx(control: usize, target: usize, theta: f64, num_qubits: usize) -> CRXGate {
    CRXGate::new(control, target, theta, num_qubits)
}

/// Create a CRY gate
pub fn cry(control: usize, target: usize, theta: f64, num_qubits: usize) -> CRYGate {
    CRYGate::new(control, target, theta, num_qubits)
}

/// Create a CRZ gate
pub fn crz(control: usize, target: usize, theta: f64, num_qubits: usize) -> CRZGate {
    CRZGate::new(control, target, theta, num_qubits)
}

/// Create a parameterized controlled-phase gate
pub fn cphase_param(
    control: usize,
    target: usize,
    theta: f64,
    num_qubits: usize,
) -> CPhaseParamGate {
    CPhaseParamGate::new(control, target, theta, num_qubits)
}

/// Create an RXX gate
pub fn rxx(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> RXXGate {
    RXXGate::new(qubit1, qubit2, theta, num_qubits)
}

/// Create an RYY gate
pub fn ryy(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> RYYGate {
    RYYGate::new(qubit1, qubit2, theta, num_qubits)
}

/// Create an RZZ gate
pub fn rzz(qubit1: usize, qubit2: usize, theta: f64, num_qubits: usize) -> RZZGate {
    RZZGate::new(qubit1, qubit2, theta, num_qubits)
}
