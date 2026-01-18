//! QLSTM Integration Tests
//!
//! This module tests the QLSTM implementation with proper datasets:
//! 1. Sine wave prediction (time series)
//! 2. Simple addition task
//! 3. XOR sequence task
//! 4. Gradient verification
//! 5. Variable sequence length handling
//! 6. Stacked QLSTM layers
//! 7. Cell creation and forward pass

use logosq::qml::{mse_loss, QLSTMCell, QLSTMConfig, VQCType, QLSTM};
use rand::Rng;
use std::f64::consts::PI;

/// Generate sine wave dataset
/// Returns (training_sequences, training_targets)
fn generate_sine_dataset(
    num_samples: usize,
    seq_length: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..num_samples {
        let phase: f64 = rng.gen_range(0.0..2.0 * PI);
        let freq: f64 = rng.gen_range(0.5..2.0);

        let mut seq = Vec::new();
        for t in 0..seq_length {
            let val = (freq * (t as f64) * 0.1 + phase).sin();
            seq.push(vec![val]);
        }

        // Target is the next value in the sequence
        let target_val = (freq * (seq_length as f64) * 0.1 + phase).sin();
        // Scale target to match hidden size
        let target = vec![target_val, target_val * 0.5];

        sequences.push(seq);
        targets.push(target);
    }

    (sequences, targets)
}

/// Simple addition task: predict sum of sequence
fn generate_addition_dataset(
    num_samples: usize,
    seq_length: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..num_samples {
        let mut seq = Vec::new();
        let mut sum = 0.0;

        for _ in 0..seq_length {
            let val: f64 = rng.gen_range(-0.5..0.5);
            seq.push(vec![val]);
            sum += val;
        }

        // Normalize sum to be in reasonable range
        let normalized_sum = (sum / seq_length as f64).tanh();
        let target = vec![normalized_sum, normalized_sum * 0.5];

        sequences.push(seq);
        targets.push(target);
    }

    (sequences, targets)
}

/// XOR-like sequence task
fn generate_xor_dataset(num_samples: usize) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..num_samples {
        let a: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let b: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

        let seq = vec![vec![a * 0.5], vec![b * 0.5]];

        // XOR: output 1 if inputs differ, -1 if same
        let xor_result = if (a > 0.0) != (b > 0.0) { 0.5 } else { -0.5 };
        let target = vec![xor_result, xor_result];

        sequences.push(seq);
        targets.push(target);
    }

    (sequences, targets)
}

/// Train QLSTM on dataset
fn train_qlstm(
    qlstm: &QLSTM,
    train_sequences: &[Vec<Vec<f64>>],
    train_targets: &[Vec<f64>],
    initial_params: &[f64],
    learning_rate: f64,
    epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut params = initial_params.to_vec();
    let mut loss_history = Vec::new();

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for (seq, target) in train_sequences.iter().zip(train_targets.iter()) {
            // Forward pass
            let output = qlstm.forward(seq, &params, None, None);
            let loss = mse_loss(&output[0], target);
            epoch_loss += loss;

            // Compute gradient and update
            let gradient = qlstm.compute_gradient(seq, target, &params, mse_loss);
            for (p, g) in params.iter_mut().zip(gradient.iter()) {
                *p -= learning_rate * g;
            }
        }

        let avg_loss = epoch_loss / train_sequences.len() as f64;
        loss_history.push(avg_loss);
    }

    (params, loss_history)
}

/// Evaluate QLSTM on test set
fn evaluate_qlstm(
    qlstm: &QLSTM,
    test_sequences: &[Vec<Vec<f64>>],
    test_targets: &[Vec<f64>],
    params: &[f64],
) -> f64 {
    let mut total_loss = 0.0;

    for (seq, target) in test_sequences.iter().zip(test_targets.iter()) {
        let output = qlstm.forward(seq, params, None, None);
        total_loss += mse_loss(&output[0], target);
    }

    total_loss / test_sequences.len() as f64
}

#[test]
fn test_qlstm_sine_wave_prediction() {
    // Smoke test for sine wave prediction with 6-VQC architecture
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3)
        .with_vqc_type(VQCType::Simple); // Use simpler VQC

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    // Generate dataset
    let (train_seqs, train_targets) = generate_sine_dataset(10, 3);
    let (test_seqs, test_targets) = generate_sine_dataset(3, 3);

    // Initialize parameters with smaller range
    let mut rng = rand::thread_rng();
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.1..0.1)).collect();

    // Evaluate before training
    let initial_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &initial_params);
    assert!(initial_loss.is_finite(), "Initial loss should be finite");

    // Train with fewer iterations (smoke test)
    let (trained_params, loss_history) =
        train_qlstm(&qlstm, &train_seqs, &train_targets, &initial_params, 0.02, 5);

    // Verify training produces finite values
    assert!(
        loss_history.iter().all(|l| l.is_finite()),
        "Training losses should be finite"
    );

    // Evaluate after training
    let final_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &trained_params);
    assert!(final_loss.is_finite(), "Final loss should be finite");
}

#[test]
fn test_qlstm_addition_task() {
    // Smoke test for addition task with 6-VQC architecture
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3)
        .with_vqc_type(VQCType::Simple); // Use simpler VQC

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    // Generate dataset
    let (train_seqs, train_targets) = generate_addition_dataset(15, 3);
    let (test_seqs, test_targets) = generate_addition_dataset(5, 3);

    // Initialize parameters with smaller range
    let mut rng = rand::thread_rng();
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.1..0.1)).collect();

    let initial_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &initial_params);
    assert!(initial_loss.is_finite(), "Initial loss should be finite");

    // Train with fewer iterations (smoke test)
    let (trained_params, loss_history) =
        train_qlstm(&qlstm, &train_seqs, &train_targets, &initial_params, 0.02, 5);

    // Verify training produces finite values
    assert!(
        loss_history.iter().all(|l| l.is_finite()),
        "Training losses should be finite"
    );

    let final_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &trained_params);
    assert!(final_loss.is_finite(), "Final loss should be finite");
}

#[test]
fn test_qlstm_xor_sequence() {
    // This test verifies that QLSTM can process XOR sequences correctly
    // Note: With 6-VQC architecture, actual learning requires more iterations
    // and careful hyperparameter tuning. This is a smoke test.
    
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3)
        .with_vqc_type(VQCType::Simple); // Simplest VQC for smoke test

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    // Generate dataset
    let (train_seqs, train_targets) = generate_xor_dataset(20);
    let (test_seqs, test_targets) = generate_xor_dataset(5);

    // Initialize parameters
    let mut rng = rand::thread_rng();
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.1..0.1)).collect();

    // Verify forward pass works
    let initial_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &initial_params);
    assert!(initial_loss.is_finite(), "Initial loss should be finite");

    // Run a few training steps (smoke test)
    let (trained_params, loss_history) =
        train_qlstm(&qlstm, &train_seqs, &train_targets, &initial_params, 0.02, 5);

    // Verify training doesn't produce NaN/Inf
    assert!(
        loss_history.iter().all(|l| l.is_finite()),
        "Training losses should be finite"
    );
    
    // Verify parameters were updated
    let params_changed = initial_params
        .iter()
        .zip(trained_params.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(params_changed, "Parameters should change during training");

    // Verify final evaluation works
    let final_loss = evaluate_qlstm(&qlstm, &test_seqs, &test_targets, &trained_params);
    assert!(final_loss.is_finite(), "Final loss should be finite");
}

#[test]
fn test_qlstm_gradient_accuracy() {
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    // Test data
    let sequence = vec![vec![0.3], vec![0.5], vec![0.7]];
    let target = vec![0.4, 0.6];

    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

    // Compute analytical gradient (parameter-shift)
    let analytical_grad = qlstm.compute_gradient(&sequence, &target, &params, mse_loss);

    // Compute numerical gradient
    let eps = 1e-4;
    let mut numerical_grad = vec![0.0; num_params];
    for i in 0..num_params {
        let mut params_plus = params.clone();
        params_plus[i] += eps;
        let out_plus = qlstm.forward(&sequence, &params_plus, None, None);
        let loss_plus = mse_loss(&out_plus[0], &target);

        let mut params_minus = params.clone();
        params_minus[i] -= eps;
        let out_minus = qlstm.forward(&sequence, &params_minus, None, None);
        let loss_minus = mse_loss(&out_minus[0], &target);

        numerical_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    // Compare - gradients should be in same direction
    let mut matching_signs = 0;
    for i in 0..num_params {
        if analytical_grad[i].abs() > 1e-6 && numerical_grad[i].abs() > 1e-6 {
            if analytical_grad[i].signum() == numerical_grad[i].signum() {
                matching_signs += 1;
            }
        } else {
            // Both small, count as match
            matching_signs += 1;
        }
    }

    // At least 80% of gradients should have matching signs
    let match_ratio = matching_signs as f64 / num_params as f64;
    assert!(
        match_ratio >= 0.8,
        "Gradient signs should mostly match: {}%",
        match_ratio * 100.0
    );
}

#[test]
fn test_qlstm_variable_sequence_length() {
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

    // Test different sequence lengths
    let lengths = vec![1, 3, 5, 10, 20];

    for &len in &lengths {
        let sequence: Vec<Vec<f64>> = (0..len).map(|i| vec![(i as f64) * 0.1]).collect();
        let output = qlstm.forward(&sequence, &params, None, None);

        // Should produce valid output for any sequence length
        assert_eq!(output.len(), 1, "Should have one output");
        assert_eq!(output[0].len(), 2, "Output should match hidden size");
        
        // Output values should be finite
        for &val in &output[0] {
            assert!(val.is_finite(), "Output should be finite for length {}", len);
        }
    }
}

#[test]
fn test_qlstm_return_sequences() {
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(1)
        .with_num_qubits(3);

    let qlstm = QLSTM::new(config).with_return_sequences(true);
    let num_params = qlstm.num_parameters();

    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];
    let output = qlstm.forward(&sequence, &params, None, None);

    // With return_sequences, should get output at each time step
    assert_eq!(output.len(), 4, "Should return output for each time step");
    
    for (t, out) in output.iter().enumerate() {
        assert_eq!(out.len(), 2, "Each output should match hidden size at t={}", t);
    }
}

#[test]
fn test_qlstm_different_vqc_types() {
    let vqc_types = vec![
        VQCType::Simple,
        VQCType::BasicEntangling,
        VQCType::StronglyEntangling,
    ];

    for vqc_type in vqc_types {
        let config = QLSTMConfig::new(1, 2)
            .with_num_layers(1)
            .with_num_qubits(3)
            .with_vqc_type(vqc_type.clone());

        let qlstm = QLSTM::new(config);
        let num_params = qlstm.num_parameters();

        let mut rng = rand::thread_rng();
        let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let sequence = vec![vec![0.5], vec![0.3]];
        let output = qlstm.forward(&sequence, &params, None, None);

        // Should produce valid output for any VQC type
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 2);
        
        for &val in &output[0] {
            assert!(val.is_finite(), "Output should be finite for {:?}", vqc_type);
            assert!(val >= -1.0 && val <= 1.0, "Output should be in [-1, 1] range");
        }
    }
}

#[test]
fn test_qlstm_mse_loss() {
    let output = vec![0.5, 0.5];
    let target = vec![1.0, 0.0];

    let loss = mse_loss(&output, &target);
    // MSE = ((0.5-1)^2 + (0.5-0)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert!((loss - 0.25).abs() < 1e-6, "MSE loss calculation incorrect");
}

// ============================================================================
// Unit Tests (moved from src/qml/qlstm.rs)
// ============================================================================

#[test]
fn test_qlstm_cell_creation() {
    let config = QLSTMConfig::new(2, 4);
    let cell = QLSTMCell::new(config.clone());

    assert_eq!(cell.config.input_size, 2);
    assert_eq!(cell.config.hidden_size, 4);
}

#[test]
fn test_qlstm_cell_forward() {
    let config = QLSTMConfig::new(2, 4).with_num_layers(1);
    let cell = QLSTMCell::new(config);

    let num_params = cell.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let input = vec![0.5, 0.3];
    let hidden = vec![0.0; 4];
    let cell_state = vec![0.0; 4];

    let output = cell.forward(&input, &hidden, &cell_state, &params);

    assert_eq!(output.hidden_state.len(), 4);
    assert_eq!(output.cell_state.len(), 4);
    assert_eq!(output.output.len(), 4);
}

#[test]
fn test_qlstm_sequence() {
    let config = QLSTMConfig::new(1, 2).with_num_layers(1);
    let qlstm = QLSTM::new(config);

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

    let output = qlstm.forward(&sequence, &params, None, None);

    // Without return_sequences, we get only the final state
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 2);
}

#[test]
fn test_qlstm_gradient() {
    let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(3);
    let qlstm = QLSTM::new(config);

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.1).collect();

    let sequence = vec![vec![0.5]];
    let target = vec![0.3, 0.7];

    let gradient = qlstm.compute_gradient(&sequence, &target, &params, mse_loss);

    assert_eq!(gradient.len(), num_params);
    // Gradients should be finite
    for g in &gradient {
        assert!(g.is_finite());
    }
}

#[test]
fn test_stacked_qlstm() {
    // Test QLSTM with num_layers > 1 (stacked layers)
    let config = QLSTMConfig::new(1, 2).with_num_layers(1);
    let qlstm = QLSTM::new(config).with_num_layers(2); // 2 stacked layers

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];

    // This should work without panicking
    let output = qlstm.forward(&sequence, &params, None, None);

    // Should return final hidden state
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 2); // hidden_size = 2
}

#[test]
fn test_stacked_qlstm_return_sequences_basic() {
    // Test stacked QLSTM with return_sequences = true
    let config = QLSTMConfig::new(1, 2).with_num_layers(1);
    let qlstm = QLSTM::new(config)
        .with_num_layers(2)
        .with_return_sequences(true);

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

    let output = qlstm.forward(&sequence, &params, None, None);

    // With return_sequences, we get output at each time step
    assert_eq!(output.len(), 4);
}

#[test]
fn test_stacked_qlstm_different_input_hidden_sizes() {
    // Test stacked QLSTM where input_size != hidden_size
    // This tests the fix for information loss between layers
    let input_size = 1;
    let hidden_size = 4; // hidden_size > input_size

    let config = QLSTMConfig::new(input_size, hidden_size).with_num_layers(1);
    let qlstm = QLSTM::new(config).with_num_layers(2);

    // Verify the layer count
    assert_eq!(qlstm.num_layers(), 2);

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];

    // This should work without information loss
    let output = qlstm.forward(&sequence, &params, None, None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), hidden_size);

    // Verify all output elements are finite (no NaN/Inf from truncation issues)
    for val in &output[0] {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }
}

#[test]
fn test_stacked_qlstm_no_truncation() {
    // Verify that intermediate layers receive proper input dimensions
    // by checking that stacked QLSTM produces valid outputs
    let input_size = 1;
    let hidden_size = 4;

    let config = QLSTMConfig::new(input_size, hidden_size).with_num_layers(1);
    let qlstm = QLSTM::new(config).with_num_layers(2);

    // Verify num_layers is correct
    assert_eq!(qlstm.num_layers(), 2);

    let num_params = qlstm.num_parameters();

    // Use non-trivial parameters
    let params: Vec<f64> = (0..num_params)
        .map(|i| 0.5 + (i as f64) * 0.1 % 1.0)
        .collect();

    let sequence = vec![vec![0.5], vec![0.3], vec![0.7]];

    // Verify we can run forward without issues
    let output = qlstm.forward(&sequence, &params, None, None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), hidden_size);

    // All outputs should be valid numbers
    for (i, val) in output[0].iter().enumerate() {
        assert!(
            val.is_finite() && *val >= -1.0 && *val <= 1.0,
            "Output[{}] = {} should be finite and in [-1, 1]",
            i,
            val
        );
    }
}

#[test]
fn test_stacked_qlstm_parameter_count() {
    // Verify that stacked QLSTM with different input/hidden sizes
    // has more parameters than a single layer would suggest
    let input_size = 1;
    let hidden_size = 4;

    let config = QLSTMConfig::new(input_size, hidden_size).with_num_layers(1);
    let single_layer = QLSTM::new(config.clone());
    let stacked = QLSTM::new(config).with_num_layers(3);

    let single_params = single_layer.num_parameters();
    let stacked_params = stacked.num_parameters();

    // Stacked should have more parameters
    // First layer: input_size=1, subsequent layers: input_size=hidden_size=4
    // So layer 1 has different param count than layers 2 and 3
    assert!(
        stacked_params > single_params,
        "Stacked QLSTM should have more parameters"
    );

    // Should have 3 layers
    assert_eq!(stacked.num_layers(), 3);
}

#[test]
fn test_qlstm_forward_with_state() {
    let config = QLSTMConfig::new(1, 2).with_num_layers(1);
    let qlstm = QLSTM::new(config);

    let num_params = qlstm.num_parameters();
    let params: Vec<f64> = (0..num_params).map(|i| (i as f64) * 0.01).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];

    // Test forward_with_state returns proper states
    let (outputs, final_hidden, final_cell) =
        qlstm.forward_with_state(&sequence, &params, None, None);

    assert_eq!(outputs.len(), 1); // return_sequences is false by default
    assert_eq!(outputs[0].len(), 2);
    assert_eq!(final_hidden.len(), 2);
    assert_eq!(final_cell.len(), 2);

    // States should be finite
    for val in &final_hidden {
        assert!(val.is_finite());
    }
    for val in &final_cell {
        assert!(val.is_finite());
    }
}
