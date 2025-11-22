//! Parameterized gate methods for Circuit
//!
//! This module provides circuit builder methods for parameterized quantum gates
//! used in variational quantum algorithms.

use super::*;
use crate::gates::parameterized::*;

impl Circuit {
    // ========================================================================
    // SINGLE-QUBIT PARAMETERIZED ROTATIONS
    // ========================================================================

    /// Add RX gate (rotation around X-axis)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.rx(0, PI / 2.0);
    /// ```
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        assert!(qubit < self.num_qubits, "Qubit index out of range");

        let gate = RXGate {
            qubit,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit], "RX");
        self
    }

    /// Add RY gate (rotation around Y-axis)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.ry(0, PI / 2.0);
    /// ```
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        assert!(qubit < self.num_qubits, "Qubit index out of range");

        let gate = RYGate {
            qubit,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit], "RY");
        self
    }

    /// Add RZ gate (rotation around Z-axis)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.rz(0, PI / 4.0);
    /// ```
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        assert!(qubit < self.num_qubits, "Qubit index out of range");

        let gate = RZGate {
            qubit,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit], "RZ");
        self
    }

    /// Add Phase gate P(θ)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `theta` - Phase angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.phase(0, PI / 2.0);  // Equivalent to S gate
    /// ```
    pub fn phase(&mut self, qubit: usize, theta: f64) -> &mut Self {
        assert!(qubit < self.num_qubits, "Qubit index out of range");

        let gate = PhaseGate {
            qubit,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit], "P");
        self
    }

    /// Add U3 gate (general single-qubit unitary)
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `theta` - Rotation angle
    /// * `phi` - Phase angle 1
    /// * `lambda` - Phase angle 2
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.u3(0, PI / 2.0, 0.0, PI);  // Equivalent to Hadamard
    /// ```
    pub fn u3(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> &mut Self {
        assert!(qubit < self.num_qubits, "Qubit index out of range");

        let gate = U3Gate {
            qubit,
            theta,
            phi,
            lambda,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit], "U3");
        self
    }

    // ========================================================================
    // TWO-QUBIT PARAMETERIZED CONTROLLED ROTATIONS
    // ========================================================================

    /// Add controlled-RX gate
    ///
    /// # Arguments
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.crx(0, 1, PI / 2.0);
    /// ```
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CRXGate {
            control,
            target,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CRX");
        self
    }

    /// Add controlled-RY gate
    ///
    /// # Arguments
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.cry(0, 1, PI / 2.0);
    /// ```
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CRYGate {
            control,
            target,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CRY");
        self
    }

    /// Add controlled-RZ gate
    ///
    /// # Arguments
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.crz(0, 1, PI / 4.0);
    /// ```
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CRZGate {
            control,
            target,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CRZ");
        self
    }

    /// Add parameterized controlled-phase gate
    ///
    /// # Arguments
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    /// * `theta` - Phase angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.cphase_param(0, 1, PI / 2.0);
    /// ```
    pub fn cphase_param(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        assert!(
            control < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(control != target, "Control and target must be different");

        let gate = CPhaseParamGate {
            control,
            target,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![control, target], "CP");
        self
    }

    // ========================================================================
    // TWO-QUBIT ENTANGLING GATES (for QAOA, VQE, etc.)
    // ========================================================================

    /// Add RXX gate (two-qubit XX rotation)
    ///
    /// # Arguments
    /// * `qubit1` - First qubit index
    /// * `qubit2` - Second qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.rxx(0, 1, PI / 4.0);
    /// ```
    pub fn rxx(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> &mut Self {
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        let gate = RXXGate {
            qubit1,
            qubit2,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit1, qubit2], "RXX");
        self
    }

    /// Add RYY gate (two-qubit YY rotation)
    ///
    /// # Arguments
    /// * `qubit1` - First qubit index
    /// * `qubit2` - Second qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.ryy(0, 1, PI / 4.0);
    /// ```
    pub fn ryy(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> &mut Self {
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        let gate = RYYGate {
            qubit1,
            qubit2,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit1, qubit2], "RYY");
        self
    }

    /// Add RZZ gate (two-qubit ZZ rotation)
    ///
    /// This is one of the most common gates in VQE and QAOA algorithms.
    ///
    /// # Arguments
    /// * `qubit1` - First qubit index
    /// * `qubit2` - Second qubit index
    /// * `theta` - Rotation angle in radians
    ///
    /// # Example
    /// ```
    /// use logosq::circuits::Circuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.rzz(0, 1, 0.5);  // Common in QAOA
    /// ```
    pub fn rzz(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> &mut Self {
        assert!(
            qubit1 < self.num_qubits && qubit2 < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(qubit1 != qubit2, "Qubits must be different");

        let gate = RZZGate {
            qubit1,
            qubit2,
            theta,
            num_qubits: self.num_qubits,
        };

        self.add_operation_unchecked(gate, vec![qubit1, qubit2], "RZZ");
        self
    }
}
