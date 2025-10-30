//! logosq: A quantum computing library in Rust
//!
//! This library provides tools for quantum computing simulation, including
//! quantum states, gates, circuits, algorithms, and noise simulation.

// Keep the modules
pub mod algorithms;
pub mod circuits;
pub mod gates;
pub mod noise;
pub mod optimization;
pub mod prelude;
pub mod states;
pub mod utils;
pub mod vis;

// Re-export common types at the crate root
pub use gates::{Gate, MatrixGate};
pub use states::State;

// Commonly used dependencies re-exported for convenience
pub use ndarray;
pub use num_complex;
