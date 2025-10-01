// This file defines the algorithms module for the quantum computing library.
// It exports functions for implementing various quantum algorithms.

// Import necessary dependencies

// Declare and re-export each algorithm module
pub mod qft;
pub mod rqc;
pub mod xyz_heisenberg;

// Optional: Re-export commonly used functions at the module level
// for easier access (without having to use the submodule path)
pub use qft::{apply as apply_qft, create_circuit as create_qft_circuit};
pub use rqc::generate as generate_random_circuit;
pub use xyz_heisenberg::simulate as simulate_heisenberg;

// Example usage functions can stay in mod.rs as they demonstrate
