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
pub use crate::error::{LogosQError, Result};
pub use crate::gates::{Gate, MatrixGate};
pub use crate::states::{QuantumStateBackend, State};

// ============================================================================
// MATRIX-BASED GATES (for reference/testing/small systems)
// ============================================================================

pub use crate::gates::matrix::{
    cnot_gate, cz_gate, h_gate, rx_gate, ry_gate, rz_gate, s_gate, swap_gate, t_gate, toffoli_gate,
    x_gate, y_gate, z_gate,
};

// ============================================================================
// OPTIMIZED SINGLE-QUBIT GATES (recommended for production)
// ============================================================================

pub use crate::gates::single_qubit::{
    // Convenience constructors
    h,
    s,
    t,
    x,
    y,
    z,
    // Gate structs
    HGate,
    SGate,
    TGate,
    XGate,
    YGate,
    ZGate,
};

// ============================================================================
// OPTIMIZED TWO-QUBIT GATES (recommended for production)
// ============================================================================

pub use crate::gates::two_qubit::{
    // Convenience constructors
    cnot,
    cphase,
    cz,
    swap,
    // Gate structs
    CNOTGate,
    CPhaseGate,
    CZGate,
    SWAPGate,
};

// ============================================================================
// OPTIMIZED THREE-QUBIT GATES
// ============================================================================

pub use crate::gates::three_qubit::{
    // Convenience constructor
    toffoli,
    // Gate structs
    ToffoliGate,
};

// ============================================================================
// PARAMETERIZED GATES (for variational quantum circuits)
// ============================================================================

pub use crate::gates::parameterized::{
    CPhaseParamGate,
    // Two-qubit parameterized rotations
    CRXGate,
    CRYGate,
    CRZGate,
    // Single-qubit parameterized rotations
    PhaseGate,
    RXGate,
    // Two-qubit entangling gates
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    // Enums and helpers
    RotationAxis,
    U3Gate,
};

// ============================================================================
// GATE UTILITIES
// ============================================================================

pub use crate::gates::utils::{flip_qubit, is_qubit_set, qubit_bit_position, qubit_mask};

// ============================================================================
// ALGORITHMS (uncomment when implemented)
// ============================================================================

// pub use crate::algorithms::{
//     qft, inverse_qft,           // Quantum Fourier Transform
//     grover, grover_diffusion,   // Grover's algorithm
//     qaoa,                       // Quantum Approximate Optimization
// };

// ============================================================================
// OPTIMIZATION
// ============================================================================

pub use crate::optimization::{
    // Gradient methods
    gradient::{FiniteDifference, GradientMethod, ParameterShift},
    // Observables
    observable::{Observable, PauliObservable, PauliTerm},
    // Optimizers
    optimizer::{Adam, GradientDescent, Optimizer, OptimizerResult},
    qng::QuantumNaturalGradient,
    // VQE
    vqe::{VQEResult, VQE},
    // Core optimization
    Ansatz,
    AnsatzType,
    HardwareEfficientAnsatz,
    ParameterizedCircuit,
};

// ============================================================================
// SIMULATOR BACKENDS
// ============================================================================

pub use crate::simulators::mps::{
    calculate_energy_mps, evolve_heisenberg_mps, MpsConfig, MpsState,
};

// ============================================================================
// NOISE MODELS
// ============================================================================

pub use crate::noise::{
    AmplitudeDampingNoise, CompositeNoise, DepolarizingNoise, NoiseModel, NoiseParams,
    PhaseDampingNoise, ThermalRelaxationNoise,
};

// ============================================================================
// QUANTUM MACHINE LEARNING
// ============================================================================

pub use crate::qml::{
    // Data encoding
    data_encoding::{AmplitudeEncoding, AngleEncoding, DataEncoder, IQPEncoding},
    // QLSTM
    qlstm::{mse_loss, cross_entropy_loss, cross_entropy_loss_probabilities, QLSTMCell, QLSTMConfig, QLSTMOutput, QLSTMTrainer, QLSTM},
    // VQC
    vqc::{DressedVQC, VQCBuilder, VQCType, VariationalQuantumCircuit},
};

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
