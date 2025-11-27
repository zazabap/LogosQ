//! XYZ-Heisenberg Model Benchmark
//!
//! Benchmark for the XYZ-Heisenberg model using logosq.
//! The XYZ-Heisenberg model is a quantum spin model that describes interacting spins
//! on a lattice with coupling in the X, Y, and Z directions.
//!
//! The Hamiltonian is:
//! H = -Σᵢⱼ [Jₓ XᵢXⱼ + Jᵧ YᵢYⱼ + Jᵧ ZᵢZⱼ] - h Σᵢ Zᵢ
//!
//! This benchmark measures:
//! - Circuit execution time
//! - Energy expectation values
//! - Resource usage

use logosq::algorithms::xyz_heisenberg::{
    calculate_energy_efficient, create_circuit, HeisenbergParameters,
};
use logosq::prelude::*;
use std::time::Instant;

fn main() {
    // Parse configuration from environment variables
    let num_qubits = std::env::var("XYZ_QUBITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    
    let time_steps = std::env::var("XYZ_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    
    let dt = std::env::var("XYZ_DT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.1);
    
    let jx = std::env::var("XYZ_JX")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    
    let jy = std::env::var("XYZ_JY")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    
    let jz = std::env::var("XYZ_JZ")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    
    let external_field = std::env::var("XYZ_FIELD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    // Create Heisenberg parameters
    let params = HeisenbergParameters {
        jx,
        jy,
        jz,
        external_field,
        time_steps,
        dt,
    };

    // Create initial state (all spins up: |1111...⟩)
    let mut state = State::one_state(num_qubits);
    
    // Calculate initial energy
    let initial_energy = calculate_energy_efficient(&state, &params);
    
    // Create circuit
    let circuit = create_circuit(num_qubits, &params);
    let num_operations = circuit.num_operations();
    
    // Measure execution time
    let start = Instant::now();
    circuit.execute(&mut state).expect("Circuit execution failed");
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    // Calculate final energy
    let final_energy = calculate_energy_efficient(&state, &params);
    let energy_change = final_energy - initial_energy;
    
    // Output JSON result
    let json_output = format!(
        r#"{{
  "framework": "LogosQ (Rust)",
  "qubits": {},
  "time_steps": {},
  "dt": {:.6},
  "jx": {:.6},
  "jy": {:.6},
  "jz": {:.6},
  "external_field": {:.6},
  "initial_energy": {:.10},
  "final_energy": {:.10},
  "energy_change": {:.10},
  "runtime_ms": {:.2},
  "num_operations": {}
}}"#,
        num_qubits,
        time_steps,
        dt,
        jx,
        jy,
        jz,
        external_field,
        initial_energy,
        final_energy,
        energy_change,
        runtime_ms,
        num_operations
    );

    // Write to JSON file
    let output_file = std::env::var("XYZ_OUTPUT_FILE")
        .unwrap_or_else(|_| "logosq_xyz_heisenberg.json".to_string());
    std::fs::write(&output_file, json_output).expect("Failed to write JSON file");
}


