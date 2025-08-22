//! Common types and traits, re-exported for convenience.
//!
//! Users can import everything needed with: `use qforge::prelude::*;`

// Re-export core types
pub use crate::gates::{Gate, MatrixGate};
pub use crate::states::State;

// When you implement these modules, add their exports here
// pub use crate::circuits::Circuit;
// pub use crate::algorithms::{Grover, QFT, QAOA};
// pub use crate::noise::NoiseModel;
