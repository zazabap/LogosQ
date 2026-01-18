//! Quantum Long Short-Term Memory (QLSTM) Demonstration
//!
//! This example demonstrates the QLSTM implementation based on:
//! - "Quantum Long Short-Term Memory" (arXiv:2009.01783) by Chen et al.
//! - PennyLane's Learning to Learn with Quantum Neural Networks
//!
//! The QLSTM is a hybrid quantum-classical model where the LSTM gates
//! are implemented using Variational Quantum Circuits (VQCs).
//!
//! # Tasks Demonstrated:
//! 1. Basic QLSTM forward pass
//! 2. Sequence prediction (sine wave)
//! 3. Simple pattern learning
//! 4. Comparison of different VQC types
//! 5. QLSTM visualization and diagram generation

use logosq::qml::{
    mse_loss, QLSTMConfig, QLSTMTrainer, VQCType, VariationalQuantumCircuit, QLSTM,
};
use logosq::vis::{qlstm_text, vqc_text, save_qlstm_svg, save_vqc_svg, Visualizable};
use rand::Rng;
use std::f64::consts::PI;

/// Generate a sine wave sequence
fn generate_sine_sequence(num_samples: usize, frequency: f64) -> Vec<Vec<f64>> {
    (0..num_samples)
        .map(|i| {
            let t = (i as f64) / (num_samples as f64) * 2.0 * PI * frequency;
            vec![t.sin()]
        })
        .collect()
}


/// Demo 1: Basic QLSTM Forward Pass
fn demo_basic_forward() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 1: Basic QLSTM Forward Pass");
    println!("{}", "=".repeat(70));

    // Create QLSTM configuration
    let config = QLSTMConfig::new(1, 2) // input_size=1, hidden_size=2
        .with_num_layers(1)
        .with_num_qubits(3);

    println!("QLSTM Configuration:");
    println!("  Input size: {}", config.input_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Number of qubits: {}", config.num_qubits);
    println!("  Number of VQC layers: {}", config.num_layers);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();
    println!("  Total parameters: {}", num_params);

    // Initialize random parameters
    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-PI..PI)).collect();

    // Create a simple sequence
    let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4], vec![0.5]];

    println!("\nInput sequence: {:?}", sequence);

    // Forward pass
    let output = qlstm.forward(&sequence, &params, None, None);

    println!("Output (final hidden state): {:?}", output[0]);
    println!(
        "Output values are in range [-1, 1] (transformed by sigmoid/tanh)"
    );
}

/// Demo 2: QLSTM with Return Sequences
fn demo_return_sequences() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 2: QLSTM with Return Sequences");
    println!("{}", "=".repeat(70));

    let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(3);

    let qlstm = QLSTM::new(config).with_return_sequences(true);

    let num_params = qlstm.num_parameters();
    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-PI..PI)).collect();

    let sequence = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];

    println!("Input sequence length: {}", sequence.len());

    let outputs = qlstm.forward(&sequence, &params, None, None);

    println!("Number of outputs: {}", outputs.len());
    println!("\nOutputs at each time step:");
    for (t, output) in outputs.iter().enumerate() {
        println!("  t={}: {:?}", t, output);
    }
}

/// Demo 3: VQC Types Comparison
fn demo_vqc_types() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 3: VQC Types Comparison");
    println!("{}", "=".repeat(70));

    let vqc_types = vec![
        ("Simple", VQCType::Simple),
        ("BasicEntangling", VQCType::BasicEntangling),
        ("StronglyEntangling", VQCType::StronglyEntangling),
    ];

    let input = vec![0.5, 0.3, 0.1];

    for (name, vqc_type) in vqc_types {
        let vqc = VariationalQuantumCircuit {
            num_qubits: 3,
            num_layers: 2,
            vqc_type: vqc_type.clone(),
            include_input_encoding: true,
        };

        let num_params = vqc.num_variational_params();
        let mut rng = rand::thread_rng();
        let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-PI..PI)).collect();

        let output = vqc.forward(&input, &params);

        println!("\n{} VQC:", name);
        println!("  Parameters: {}", num_params);
        println!("  Output: {:?}", output);
    }
}

/// Demo 4: Simple Training Example
fn demo_training() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 4: Simple QLSTM Training");
    println!("{}", "=".repeat(70));

    // Create a small QLSTM
    let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(3);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    println!("Training QLSTM to predict a simple pattern...");
    println!("Number of parameters: {}", num_params);

    // Initialize parameters
    let mut rng = rand::thread_rng();
    let initial_params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

    // Training data: simple ascending pattern
    let sequence = vec![vec![0.1], vec![0.2], vec![0.3]];
    let target = vec![0.4, 0.5]; // Expected output pattern

    // Create trainer
    let trainer = QLSTMTrainer {
        learning_rate: 0.05,
        max_iterations: 50,
        tolerance: 1e-4,
        verbose: true,
    };

    // Train
    let (optimized_params, final_loss) = trainer.train(&qlstm, &sequence, &target, &initial_params);

    println!("\nTraining completed!");
    println!("Final loss: {:.6}", final_loss);

    // Test the trained model
    let output = qlstm.forward(&sequence, &optimized_params, None, None);
    println!("Target: {:?}", target);
    println!("Predicted: {:?}", output[0]);
}

/// Demo 5: Sine Wave Prediction
fn demo_sine_prediction() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 5: Sine Wave Sequence Prediction");
    println!("{}", "=".repeat(70));

    let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(4);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    // Generate sine wave data
    let full_sequence = generate_sine_sequence(10, 1.0);
    let input_seq: Vec<Vec<f64>> = full_sequence[..5].to_vec();
    let target: Vec<f64> = full_sequence[5].clone();

    println!("Input sequence (first 5 points of sine wave):");
    for (i, x) in input_seq.iter().enumerate() {
        println!("  t={}: {:.4}", i, x[0]);
    }
    println!("Target (6th point): {:.4}", target[0]);

    // Initialize and predict
    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-PI..PI)).collect();

    let output = qlstm.forward(&input_seq, &params, None, None);

    println!("\nPrediction (untrained): {:?}", output[0]);
    println!(
        "Note: The untrained model gives random predictions."
    );
    println!("Training would optimize parameters to minimize prediction error.");
}

/// Demo 6: Computing Gradients
fn demo_gradients() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 6: Computing Gradients via Parameter-Shift Rule");
    println!("{}", "=".repeat(70));

    let config = QLSTMConfig::new(1, 2).with_num_layers(1).with_num_qubits(3);

    let qlstm = QLSTM::new(config);
    let num_params = qlstm.num_parameters();

    println!("Computing gradients for {} parameters...", num_params);

    let mut rng = rand::thread_rng();
    let params: Vec<f64> = (0..num_params).map(|_| rng.gen_range(-0.5..0.5)).collect();

    let sequence = vec![vec![0.5], vec![0.3]];
    let target = vec![0.7, 0.2];

    // Compute initial loss
    let output = qlstm.forward(&sequence, &params, None, None);
    let loss = mse_loss(&output[0], &target);
    println!("Initial loss: {:.6}", loss);

    // Compute gradients
    let gradients = qlstm.compute_gradient(&sequence, &target, &params, mse_loss);

    println!("\nGradients (first 8 parameters):");
    for (i, g) in gradients.iter().take(8).enumerate() {
        println!("  ∂L/∂θ[{}] = {:.6}", i, g);
    }

    // Verify gradient by numerical approximation
    let eps = 1e-4;
    println!("\nNumerical gradient verification (first 3 parameters):");
    for i in 0..3.min(num_params) {
        let mut params_plus = params.clone();
        params_plus[i] += eps;
        let out_plus = qlstm.forward(&sequence, &params_plus, None, None);
        let loss_plus = mse_loss(&out_plus[0], &target);

        let mut params_minus = params.clone();
        params_minus[i] -= eps;
        let out_minus = qlstm.forward(&sequence, &params_minus, None, None);
        let loss_minus = mse_loss(&out_minus[0], &target);

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
        println!(
            "  θ[{}]: analytical={:.6}, numerical={:.6}, diff={:.6}",
            i,
            gradients[i],
            numerical_grad,
            (gradients[i] - numerical_grad).abs()
        );
    }
}

/// Demo 7: QLSTM Architecture Overview
fn demo_architecture() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 7: QLSTM Architecture Overview");
    println!("{}", "=".repeat(70));

    println!("\nClassical LSTM Cell:");
    println!("  f_t = σ(W_f · [h_{{t-1}}, x_t] + b_f)  [Forget Gate]");
    println!("  i_t = σ(W_i · [h_{{t-1}}, x_t] + b_i)  [Input Gate]");
    println!("  c̃_t = tanh(W_c · [h_{{t-1}}, x_t] + b_c)  [Cell Gate]");
    println!("  o_t = σ(W_o · [h_{{t-1}}, x_t] + b_o)  [Output Gate]");
    println!("  c_t = f_t ⊙ c_{{t-1}} + i_t ⊙ c̃_t");
    println!("  h_t = o_t ⊙ tanh(c_t)");

    println!("\nQLSTM Cell (Quantum Version):");
    println!("  f_t = σ(VQC_f(h_{{t-1}}, x_t; θ_f))  [Quantum Forget Gate]");
    println!("  i_t = σ(VQC_i(h_{{t-1}}, x_t; θ_i))  [Quantum Input Gate]");
    println!("  c̃_t = tanh(VQC_c(h_{{t-1}}, x_t; θ_c))  [Quantum Cell Gate]");
    println!("  o_t = σ(VQC_o(h_{{t-1}}, x_t; θ_o))  [Quantum Output Gate]");
    println!("  (Cell state and hidden state updates remain classical)");

    println!("\nVQC Structure:");
    println!("  1. Input encoding: RY(π·x_i) for each input on qubit i");
    println!("  2. Variational layers:");
    println!("     - Rotation gates: RY(θ), RZ(θ)");
    println!("     - Entangling gates: CNOT ladder");
    println!("  3. Measurement: Z expectation on each qubit");

    println!("\nReferences:");
    println!("  [1] Chen et al., 'Quantum Long Short-Term Memory'");
    println!("      arXiv:2009.01783 (2020)");
    println!("  [2] PennyLane: Learning to Learn with Quantum Neural Networks");
    println!("      https://pennylane.ai/qml/demos/learning2learn");
}

/// Demo 8: QLSTM Visualization
fn demo_visualization() {
    println!("\n{}", "=".repeat(70));
    println!("Demo 8: QLSTM Visualization & Diagram Generation");
    println!("{}", "=".repeat(70));

    // Create QLSTM configuration
    let config = QLSTMConfig::new(1, 2)
        .with_num_layers(2)
        .with_num_qubits(4)
        .with_vqc_type(VQCType::BasicEntangling);

    // Print text diagram
    println!("\n--- QLSTM Cell Text Diagram ---\n");
    println!("{}", qlstm_text(&config));

    // Create and visualize VQC
    let vqc = VariationalQuantumCircuit {
        num_qubits: 4,
        num_layers: 2,
        vqc_type: VQCType::BasicEntangling,
        include_input_encoding: true,
    };

    println!("\n--- VQC Text Diagram ---\n");
    println!("{}", vqc_text(&vqc));

    // Save SVG diagrams
    println!("\nGenerating SVG diagrams...");
    
    match save_qlstm_svg(&config, "qlstm_cell.svg") {
        Ok(_) => println!("  ✓ Saved QLSTM cell diagram to: qlstm_cell.svg"),
        Err(e) => println!("  ✗ Failed to save QLSTM diagram: {}", e),
    }

    match save_vqc_svg(&vqc, "qlstm_vqc.svg") {
        Ok(_) => println!("  ✓ Saved VQC diagram to: qlstm_vqc.svg"),
        Err(e) => println!("  ✗ Failed to save VQC diagram: {}", e),
    }

    // Also save using the Visualizable trait
    let qlstm = QLSTM::new(config.clone());
    match qlstm.save_visualization("qlstm_full.svg") {
        Ok(_) => println!("  ✓ Saved full QLSTM diagram to: qlstm_full.svg"),
        Err(e) => println!("  ✗ Failed to save full QLSTM diagram: {}", e),
    }

    // Save different VQC types for comparison
    println!("\nGenerating VQC type comparison diagrams...");
    
    let vqc_types = [
        ("Simple", VQCType::Simple),
        ("BasicEntangling", VQCType::BasicEntangling),
        ("StronglyEntangling", VQCType::StronglyEntangling),
    ];

    for (name, vqc_type) in &vqc_types {
        let vqc = VariationalQuantumCircuit {
            num_qubits: 3,
            num_layers: 2,
            vqc_type: vqc_type.clone(),
            include_input_encoding: true,
        };
        
        let filename = format!("vqc_{}.svg", name.to_lowercase());
        match save_vqc_svg(&vqc, &filename) {
            Ok(_) => println!("  ✓ Saved {} VQC diagram to: {}", name, filename),
            Err(e) => println!("  ✗ Failed to save {} VQC diagram: {}", name, e),
        }
    }

    println!("\nVisualization complete!");
    println!("Open the SVG files in a web browser to view the diagrams.");
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Quantum Long Short-Term Memory (QLSTM) Demonstration");
    println!("{}", "=".repeat(70));
    println!("\nThis example demonstrates the QLSTM implementation based on:");
    println!("  - arXiv:2009.01783 'Quantum Long Short-Term Memory'");
    println!("  - PennyLane's Learning to Learn tutorial");

    // Run demos
    demo_architecture();
    demo_basic_forward();
    demo_return_sequences();
    demo_vqc_types();
    demo_gradients();
    demo_sine_prediction();
    demo_training();
    demo_visualization();

    println!("\n{}", "=".repeat(70));
    println!("QLSTM Demonstration Complete!");
    println!("{}", "=".repeat(70));
}
