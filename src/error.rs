//! Error types for the logosq quantum computing library
//!
//! This module provides comprehensive error handling using the `thiserror` crate
//! for better error messages and error chaining.

use thiserror::Error;

/// Main error type for the logosq library
#[derive(Error, Debug)]
pub enum LogosQError {
    /// Invalid qubit index
    #[error("Invalid qubit index: {index} (must be < {num_qubits})")]
    InvalidQubitIndex {
        index: usize,
        num_qubits: usize,
    },

    /// Invalid state dimension
    #[error("Invalid state dimension: {dimension} (must be a power of 2)")]
    InvalidStateDimension {
        dimension: usize,
    },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

    /// Invalid gate matrix dimensions
    #[error("Invalid gate matrix dimensions: expected {expected:?}, got {actual:?}")]
    InvalidGateMatrix {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Parameter count mismatch
    #[error("Parameter count mismatch: expected {expected}, got {actual}")]
    ParameterCountMismatch {
        expected: usize,
        actual: usize,
    },

    /// Circuit qubit count mismatch
    #[error("Circuit qubit count mismatch: circuit has {circuit_qubits}, state has {state_qubits}")]
    CircuitQubitMismatch {
        circuit_qubits: usize,
        state_qubits: usize,
    },

    /// Invalid probability value
    #[error("Invalid probability: {value} (must be in [0, 1])")]
    InvalidProbability {
        value: f64,
    },

    /// Invalid angle value
    #[error("Invalid angle: {value}")]
    InvalidAngle {
        value: f64,
    },

    /// Measurement error
    #[error("Measurement error: {message}")]
    MeasurementError {
        message: String,
    },

    /// Optimization error
    #[error("Optimization error: {message}")]
    OptimizationError {
        message: String,
    },

    /// Gradient computation error
    #[error("Gradient computation error: {message}")]
    GradientError {
        message: String,
    },
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, LogosQError>;

