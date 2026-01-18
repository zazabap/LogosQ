//! Quantum Machine Learning module
//!
//! This module provides quantum machine learning components including:
//! - Quantum Long Short-Term Memory (QLSTM)
//! - Variational Quantum Circuits for ML
//! - Data encoding strategies
//!
//! Based on:
//! - "Quantum Long Short-Term Memory" (arXiv:2009.01783) by Chen et al.
//! - PennyLane's quantum neural network implementations
//!
//! # Example: QLSTM for Sequence Prediction
//!
//! ```rust
//! use logosq::qml::{QLSTM, QLSTMConfig};
//!
//! // Create a QLSTM with input size 1, hidden size 2
//! let config = QLSTMConfig::new(1, 2).with_num_layers(1);
//! let qlstm = QLSTM::new(config);
//!
//! // Initialize parameters (randomly in practice)
//! let num_params = qlstm.num_parameters();
//! let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();
//!
//! // Process a sequence
//! let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];
//! let output = qlstm.forward(&sequence, &params, None, None);
//!
//! println!("Output: {:?}", output);
//! ```

pub mod data_encoding;
pub mod qlstm;
pub mod vqc;

// Re-exports for convenience
pub use data_encoding::{AmplitudeEncoding, AngleEncoding, DataEncoder, IQPEncoding};
pub use qlstm::{cross_entropy_loss, cross_entropy_loss_probabilities, mse_loss, QLSTMCell, QLSTMConfig, QLSTMOutput, QLSTMTrainer, QLSTM};
pub use vqc::{DressedVQC, VQCBuilder, VQCType, VariationalQuantumCircuit};
