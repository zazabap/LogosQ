//! logosq: A quantum computing library in Rust
//!
//! This library provides tools for quantum computing simulation, including
//! quantum states, gates, circuits, algorithms, and noise simulation.
//!
//! # Example
//!
//! ```rust
//! use logosq::prelude::*;
//!
//! // Create a 2-qubit state
//! let mut state = State::zero_state(2);
//!
//! // Create a circuit
//! let mut circuit = Circuit::new(2);
//! circuit.h(0).cnot(0, 1);
//!
//! // Execute the circuit
//! circuit.execute(&mut state);
//!
//! // Measure
//! let result = state.measure();
//! ```

// Core modules
pub mod algorithms;
pub mod circuits;
pub mod error;
pub mod gates;
pub mod noise;
pub mod optimization;
pub mod prelude;
pub mod simulators;
pub mod states;
pub mod utils;
pub mod vis;

// Re-export common types at the crate root
pub use error::{LogosQError, Result};
pub use gates::{Gate, MatrixGate};
pub use states::State;

// Commonly used dependencies re-exported for convenience
pub use ndarray;
pub use num_complex;
