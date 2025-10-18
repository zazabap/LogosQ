//! Common types and traits, re-exported for convenience.
//!
//! Users can import everything needed with: `use logosq::prelude::*;`

// ============================================================================
// CORE MODULES
// ============================================================================

// Re-export entire modules
pub use crate::circuits;
pub use crate::gates;
pub use crate::states;

// Re-export core types for direct access
pub use crate::circuits::Circuit;
pub use crate::gates::{Gate, MatrixGate};
pub use crate::states::State;

// ============================================================================
// MATRIX-BASED GATES (for reference/testing/small systems)
// ============================================================================

pub use crate::gates::matrix::{
    cnot_gate, cz_gate, h_gate, rx_gate, ry_gate, rz_gate, s_gate, swap_gate, t_gate,
    toffoli_gate, x_gate, y_gate, z_gate,
};

// ============================================================================
// OPTIMIZED SINGLE-QUBIT GATES (recommended for production)
// ============================================================================

pub use crate::gates::single_qubit::{
    // Gate structs
    HGate,
    SGate,
    TGate,
    XGate,
    YGate,
    ZGate,
    // Convenience constructors
    h,
    s,
    t,
    x,
    y,
    z,
};

// ============================================================================
// OPTIMIZED TWO-QUBIT GATES (recommended for production)
// ============================================================================

pub use crate::gates::two_qubit::{
    // Gate structs
    CNOTGate,
    CPhaseGate,
    CZGate,
    SWAPGate,
    // Convenience constructors
    cnot,
    cphase,
    cz,
    swap,
};

// ============================================================================
// OPTIMIZED THREE-QUBIT GATES
// ============================================================================

pub use crate::gates::three_qubit::{
    // Gate structs
    ToffoliGate,
    // Convenience constructor
    toffoli,
};

// ============================================================================
// PARAMETERIZED GATES (for variational quantum circuits)
// ============================================================================

// pub use crate::gates::parameterized::{
//     // Single-qubit parameterized rotations
//     PhaseGate,
//     RXGate,
//     RYGate,
//     RZGate,
//     U3Gate,
//     // Two-qubit parameterized rotations
//     CRXGate,
//     CRYGate,
//     CRZGate,
//     CPhaseParamGate,
//     // Two-qubit entangling gates
//     RXXGate,
//     RYYGate,
//     RZZGate,
//     // Enums and helpers
//     RotationAxis,
// };

// ============================================================================
// GATE UTILITIES
// ============================================================================

pub use crate::gates::utils::{
    flip_qubit, is_qubit_set, qubit_bit_position, qubit_mask,
};

// ============================================================================
// ALGORITHMS (uncomment when implemented)
// ============================================================================

// pub use crate::algorithms::{
//     qft, inverse_qft,           // Quantum Fourier Transform
//     grover, grover_diffusion,   // Grover's algorithm
//     qaoa,                       // Quantum Approximate Optimization
// };

// ============================================================================
// OPTIMIZATION (uncomment when implemented)
// ============================================================================

// pub use crate::optimization::{
//     // Core optimization
//     ParameterizedCircuit,
//     Optimizer,
//     OptimizerConfig,
//     OptimizationResult,
//     // Gradient methods
//     GradientMethod,
//     ParameterShift,
//     FiniteDifference,
//     // Optimizers
//     GradientDescent,
//     Adam,
//     QuantumNaturalGradient,
//     // Observables
//     Observable,
//     Pauli,
//     PauliTerm,
// };

// ============================================================================
// NOISE MODELS (uncomment when implemented)
// ============================================================================

// pub use crate::noise::{
//     NoiseModel,
//     DepolarizingNoise,
//     AmplitudeDamping,
//     PhaseDamping,
// };

// ============================================================================
// COMMON CONSTANTS
// ============================================================================

pub use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

// ============================================================================
// TYPE ALIASES
// ============================================================================

/// Complex number type used throughout the library
pub use num_complex::Complex64;

/// Convenient type alias for state vectors
pub type StateVector = ndarray::Array1<Complex64>;

/// Convenient type alias for gate matrices
pub type GateMatrix = ndarray::Array2<Complex64>;
