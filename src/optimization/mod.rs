//! Optimization module for variational quantum algorithms
//!
//! This module provides tools for optimizing parameterized quantum circuits,
//! including VQE, QAOA, and other variational algorithms.

pub mod ansatz;
pub mod gradient;
pub mod observable;
pub mod optimizer;
pub mod qng;
pub mod vqe;

// Re-exports for convenience
pub use ansatz::{Ansatz, AnsatzType, HardwareEfficientAnsatz, ParameterizedCircuit};
pub use gradient::{GradientMethod, ParameterShift};
pub use observable::{Observable, PauliObservable, PauliTerm};
pub use optimizer::{Adam, GradientDescent, Optimizer, OptimizerResult};
pub use qng::QuantumNaturalGradient;
pub use vqe::{VQEResult, VQE};
