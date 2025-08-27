//! Common types and traits, re-exported for convenience.
//!
//! Users can import everything needed with: `use logosq::prelude::*;`

// Re-export entire modules
pub use crate::states;
pub use crate::circuits;
pub use crate::gates;

// Re-export core types for direct access
pub use crate::gates::{Gate, MatrixGate};
pub use crate::states::State;
pub use crate::circuits::Circuit;

// Re-export common gate functions
pub use crate::gates::{
    // Single-qubit gates
    x_gate, y_gate, z_gate, h_gate, s_gate, t_gate,
    rx_gate, ry_gate, rz_gate,

    // Multi-qubit gates
    cnot_gate, cz_gate, swap_gate, toffoli_gate
};

// When you implement these modules, add their exports here
// pub use crate::algorithms::{Grover, QFT, QAOA};
// pub use crate::noise::NoiseModel;
