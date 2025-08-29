//! Common types and traits, re-exported for convenience.
//!
//! Users can import everything needed with: `use logosq::prelude::*;`

// Re-export entire modules
pub use crate::circuits;
pub use crate::gates;
pub use crate::states;

// Re-export core types for direct access
pub use crate::circuits::Circuit;
pub use crate::gates::{Gate, MatrixGate};
pub use crate::states::State;

// Re-export common gate functions
pub use crate::gates::{
    // Multi-qubit gates
    cnot_gate,
    cz_gate,
    h_gate,
    rx_gate,
    ry_gate,
    rz_gate,

    s_gate,
    swap_gate,
    t_gate,
    toffoli_gate,
    // Single-qubit gates
    x_gate,
    y_gate,
    z_gate,
};

// When you implement these modules, add their exports here
// pub use crate::algorithms::{Grover, QFT, QAOA};
// pub use crate::noise::NoiseModel;
