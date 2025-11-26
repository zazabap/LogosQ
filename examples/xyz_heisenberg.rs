//! XYZ-Heisenberg Model Example
//!
//! This example demonstrates the simulation of the XYZ-Heisenberg model using logosq.
//! The XYZ-Heisenberg model is a quantum spin model that describes interacting spins
//! on a lattice with coupling in the X, Y, and Z directions.
//!
//! The Hamiltonian is:
//! H = -Σᵢⱼ [Jₓ XᵢXⱼ + Jᵧ YᵢYⱼ + Jᵧ ZᵢZⱼ] - h Σᵢ Zᵢ
//!
//! This example showcases:
//! - Creating Heisenberg model circuits with different parameter sets
//! - Simulating time evolution
//! - Calculating energy expectation values
//! - Visualizing quantum circuits
//! - Comparing different model configurations

use logosq::algorithms::xyz_heisenberg::{
    calculate_energy_efficient, create_circuit, HeisenbergParameters,
};
use logosq::prelude::*;
use logosq::vis::save_circuit_svg;

fn main() {
    println!("{}", "=".repeat(70));
    println!("XYZ-Heisenberg Model Simulation");
    println!("{}", "=".repeat(70));
    println!();

    let num_qubits = 4;

    // ============================================================================
    // Example 1: Isotropic Heisenberg Model (Jx = Jy = Jz = 1.0)
    // ============================================================================
    println!("Example 1: Isotropic Heisenberg Model");
    println!("{}", "-".repeat(70));
    let isotropic_params = HeisenbergParameters {
        jx: 1.0,
        jy: 1.0,
        jz: 1.0,
        external_field: 0.0,
        time_steps: 10,
        dt: 0.1,
    };

    demonstrate_heisenberg_model(num_qubits, &isotropic_params, "isotropic");

    // ============================================================================
    // Example 2: Anisotropic Heisenberg Model (Jx ≠ Jy ≠ Jz)
    // ============================================================================
    println!("\n\nExample 2: Anisotropic Heisenberg Model");
    println!("{}", "-".repeat(70));
    let anisotropic_params = HeisenbergParameters {
        jx: 1.0,
        jy: 0.5,
        jz: 1.5,
        external_field: 0.0,
        time_steps: 10,
        dt: 0.1,
    };

    demonstrate_heisenberg_model(num_qubits, &anisotropic_params, "anisotropic");

    // ============================================================================
    // Example 3: Heisenberg Model with External Magnetic Field
    // ============================================================================
    println!("\n\nExample 3: Heisenberg Model with External Magnetic Field");
    println!("{}", "-".repeat(70));
    let field_params = HeisenbergParameters {
        jx: 1.0,
        jy: 1.0,
        jz: 1.0,
        external_field: 0.5,
        time_steps: 10,
        dt: 0.1,
    };

    demonstrate_heisenberg_model(num_qubits, &field_params, "with_field");

    // ============================================================================
    // Example 4: XXZ Model (Jx = Jy ≠ Jz, common in quantum magnetism)
    // ============================================================================
    println!("\n\nExample 4: XXZ Model (Jx = Jy ≠ Jz)");
    println!("{}", "-".repeat(70));
    let xxz_params = HeisenbergParameters {
        jx: 1.0,
        jy: 1.0,
        jz: 0.5,
        external_field: 0.0,
        time_steps: 10,
        dt: 0.1,
    };

    demonstrate_heisenberg_model(num_qubits, &xxz_params, "xxz");

    // ============================================================================
    // Example 5: Long-time Evolution
    // ============================================================================
    println!("\n\nExample 5: Long-time Evolution");
    println!("{}", "-".repeat(70));
    let long_time_params = HeisenbergParameters {
        jx: 1.0,
        jy: 1.0,
        jz: 1.0,
        external_field: 0.0,
        time_steps: 50,
        dt: 0.05,
    };

    demonstrate_time_evolution(num_qubits, &long_time_params);

    println!("\n{}", "=".repeat(70));
    println!("All examples completed successfully!");
    println!("{}", "=".repeat(70));
}

/// Demonstrate a Heisenberg model configuration
fn demonstrate_heisenberg_model(
    num_qubits: usize,
    params: &HeisenbergParameters,
    name: &str,
) {
    println!("Parameters:");
    println!("  Jx = {:.3}, Jy = {:.3}, Jz = {:.3}", params.jx, params.jy, params.jz);
    println!("  External field h = {:.3}", params.external_field);
    println!("  Time steps = {}, dt = {:.3}", params.time_steps, params.dt);
    println!("  Total time = {:.3}", params.time_steps as f64 * params.dt);

    // Create initial state (all spins up: |1111...⟩)
    let state = State::one_state(num_qubits);

    // Calculate initial energy
    let initial_energy = calculate_energy_efficient(&state, params);
    println!("\nInitial state: |{}⟩", "1".repeat(num_qubits));
    println!("Initial energy: {:.6}", initial_energy);

    // Create and visualize the circuit
    let circuit = create_circuit(num_qubits, params);
    println!("\nCircuit created with {} operations", circuit.num_operations());

    // Save circuit visualization
    let filename = format!("xyz_heisenberg_{}.svg", name);
    save_circuit_svg(&circuit, &filename).unwrap_or_else(|e| {
        println!("Warning: Failed to save circuit diagram: {}", e);
    });
    println!("Circuit diagram saved to '{}'", filename);

    // Simulate time evolution
    println!("\nSimulating time evolution...");
    let mut evolved_state = state.clone();
    circuit.execute(&mut evolved_state).expect("Circuit execution failed");

    // Calculate final energy
    let final_energy = calculate_energy_efficient(&evolved_state, params);
    println!("Final energy: {:.6}", final_energy);
    println!("Energy change: {:.6}", final_energy - initial_energy);

    // Show some state probabilities
    println!("\nFinal state probabilities (top 5):");
    let mut probs: Vec<(usize, f64)> = (0..evolved_state.vector().len())
        .map(|i| (i, evolved_state.probability(i)))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, prob) in probs.iter().take(5) {
        let state_str = format!("{:0width$b}", i, width = num_qubits);
        println!("  |{}⟩: {:.4}", state_str, prob);
    }
}

/// Demonstrate time evolution with intermediate steps
fn demonstrate_time_evolution(num_qubits: usize, params: &HeisenbergParameters) {
    println!("Parameters:");
    println!("  Jx = {:.3}, Jy = {:.3}, Jz = {:.3}", params.jx, params.jy, params.jz);
    println!("  Time steps = {}, dt = {:.3}", params.time_steps, params.dt);
    println!("  Total time = {:.3}", params.time_steps as f64 * params.dt);

    // Create initial state (all spins up)
    let mut state = State::one_state(num_qubits);
    let initial_energy = calculate_energy_efficient(&state, params);
    println!("\nInitial energy: {:.6}", initial_energy);

    // Evolve step by step to show intermediate energies
    println!("\nEnergy evolution:");
    println!("  Time    Energy");
    println!("  ----    ------");
    println!("  {:.3}    {:.6}", 0.0, initial_energy);

    // Create a circuit for a single time step
    let single_step_params = HeisenbergParameters {
        jx: params.jx,
        jy: params.jy,
        jz: params.jz,
        external_field: params.external_field,
        time_steps: 1,
        dt: params.dt,
    };

    for step in 1..=params.time_steps {
        let step_circuit = create_circuit(num_qubits, &single_step_params);
        step_circuit.execute(&mut state).expect("Circuit execution failed");
        
        let energy = calculate_energy_efficient(&state, params);
        let time = step as f64 * params.dt;
        println!("  {:.3}    {:.6}", time, energy);
    }
}


