// XYZ-Heisenberg Model Dynamics implementation

use std::f64::consts::PI;
use crate::circuits::Circuit;
use crate::states::State;

/// Parameters for the XYZ-Heisenberg model
pub struct HeisenbergParameters {
    pub jx: f64,         // Coupling strength in X direction
    pub jy: f64,         // Coupling strength in Y direction
    pub jz: f64,         // Coupling strength in Z direction
    pub external_field: f64,  // External magnetic field strength
    pub time_steps: usize,    // Number of time steps for evolution
    pub dt: f64,         // Time step size
}

impl Default for HeisenbergParameters {
    fn default() -> Self {
        Self {
            jx: 1.0,
            jy: 1.0,
            jz: 1.0,
            external_field: 0.0,
            time_steps: 100,
            dt: 0.01,
        }
    }
}

/// Creates a circuit that simulates the XYZ-Heisenberg model for the given parameters
pub fn create_circuit(num_qubits: usize, params: &HeisenbergParameters) -> Circuit {
    let mut circuit = Circuit::new(num_qubits)
        .with_name("XYZ-Heisenberg Model");

    // Time evolution using Trotter-Suzuki decomposition
    for _ in 0..params.time_steps {
        // Add XX interactions
        for i in 0..num_qubits-1 {
            add_xx_interaction(&mut circuit, i, i+1, params.jx * params.dt);
        }

        // Add YY interactions
        for i in 0..num_qubits-1 {
            add_yy_interaction(&mut circuit, i, i+1, params.jy * params.dt);
        }

        // Add ZZ interactions
        for i in 0..num_qubits-1 {
            add_zz_interaction(&mut circuit, i, i+1, params.jz * params.dt);
        }

        // Add external field
        if params.external_field != 0.0 {
            for i in 0..num_qubits {
                circuit.rz(i, params.external_field * params.dt);
            }
        }
    }

    circuit
}

/// Simulate the XYZ-Heisenberg model dynamics on a given state
pub fn simulate(state: &mut State, params: &HeisenbergParameters) {
    let circuit = create_circuit(state.num_qubits, params);
    circuit.execute(state);
}

// Helper functions for adding interactions
fn add_xx_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Implementation of e^(-i * strength * X⊗X)
    circuit.h(qubit1);
    circuit.h(qubit2);
    add_zz_interaction(circuit, qubit1, qubit2, strength);
    circuit.h(qubit1);
    circuit.h(qubit2);
}

fn add_yy_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Implementation of e^(-i * strength * Y⊗Y)
    circuit.rx(qubit1, PI/2.0);
    circuit.rx(qubit2, PI/2.0);
    add_zz_interaction(circuit, qubit1, qubit2, strength);
    circuit.rx(qubit1, -PI/2.0);
    circuit.rx(qubit2, -PI/2.0);
}

fn add_zz_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Implementation of e^(-i * strength * Z⊗Z)
    circuit.cnot(qubit1, qubit2);
    circuit.rz(qubit2, 2.0 * strength);
    circuit.cnot(qubit1, qubit2);
}

/// Calculate energy of the XYZ-Heisenberg model for a given state
pub fn calculate_energy(state: &State, params: &HeisenbergParameters) -> f64 {
    let num_qubits = state.num_qubits;
    let mut energy = 0.0;

    // Calculate nearest-neighbor interaction energies
    for i in 0..num_qubits-1 {
        // Calculate XX interaction energy: Jx * <ψ|X_i X_{i+1}|ψ>
        let mut xx_state = state.clone();
        let mut xx_circuit = Circuit::new(num_qubits);
        xx_circuit.x(i);
        xx_circuit.x(i+1);
        xx_circuit.execute(&mut xx_state);
        let xx_energy = params.jx * state.vector.dot(&xx_state.vector).re;
        energy += xx_energy;

        // Calculate YY interaction energy: Jy * <ψ|Y_i Y_{i+1}|ψ>
        let mut yy_state = state.clone();
        let mut yy_circuit = Circuit::new(num_qubits);
        yy_circuit.y(i);
        yy_circuit.y(i+1);
        yy_circuit.execute(&mut yy_state);
        let yy_energy = params.jy * state.vector.dot(&yy_state.vector).re;
        energy += yy_energy;
        
        // Calculate ZZ interaction energy: Jz * <ψ|Z_i Z_{i+1}|ψ>
        let mut zz_state = state.clone();
        let mut zz_circuit = Circuit::new(num_qubits);
        zz_circuit.z(i);
        zz_circuit.z(i+1);
        zz_circuit.execute(&mut zz_state);
        let zz_energy = params.jz * state.vector.dot(&zz_state.vector).re;
        energy += zz_energy;
    }
    
    // Calculate external field contribution: h * <ψ|Z_i|ψ>
    if params.external_field != 0.0 {
        for i in 0..num_qubits {
            let mut z_state = state.clone();
            let mut z_circuit = Circuit::new(num_qubits);
            z_circuit.z(i);
            z_circuit.execute(&mut z_state);
            let z_energy = params.external_field * state.vector.dot(&z_state.vector).re;
            energy += z_energy;
        }
    }

    energy
}

/// Calculate energy using a more efficient approach with expectation values
pub fn calculate_energy_efficient(state: &State, params: &HeisenbergParameters) -> f64 {
    let num_qubits = state.num_qubits;
    let mut energy = 0.0;
    
    // Calculate nearest-neighbor interaction energies
    for i in 0..num_qubits-1 {
        energy += params.jx * expectation_xx(state, i, i+1);
        energy += params.jy * expectation_yy(state, i, i+1);
        energy += params.jz * expectation_zz(state, i, i+1);
    }
    
    // Calculate external field contribution
    if params.external_field != 0.0 {
        for i in 0..num_qubits {
            energy += params.external_field * expectation_z(state, i);
        }
    }
    
    energy
}

/// Calculate expectation value of X_i X_j
fn expectation_xx(state: &State, i: usize, j: usize) -> f64 {
    let mut state_copy = state.clone();
    
    // Create a temporary circuit to apply X gates to specific qubits
    let mut circuit = Circuit::new(state.num_qubits);
    circuit.x(i);
    circuit.x(j);
    
    // Execute the circuit on our state copy
    circuit.execute(&mut state_copy);
    
    // Calculate inner product
    state.vector.dot(&state_copy.vector).re
}

/// Calculate expectation value of Y_i Y_j
fn expectation_yy(state: &State, i: usize, j: usize) -> f64 {
    let mut state_copy = state.clone();
    
    // Create a temporary circuit to apply Y gates to specific qubits
    let mut circuit = Circuit::new(state.num_qubits);
    circuit.y(i);
    circuit.y(j);
    
    // Execute the circuit on our state copy
    circuit.execute(&mut state_copy);
    
    // Calculate inner product
    state.vector.dot(&state_copy.vector).re
}

/// Calculate expectation value of Z_i Z_j
fn expectation_zz(state: &State, i: usize, j: usize) -> f64 {
    let mut state_copy = state.clone();
    
    // Create a temporary circuit to apply Z gates to specific qubits
    let mut circuit = Circuit::new(state.num_qubits);
    circuit.z(i);
    circuit.z(j);
    
    // Execute the circuit on our state copy
    circuit.execute(&mut state_copy);
    
    // Calculate inner product
    state.vector.dot(&state_copy.vector).re
}

/// Calculate expectation value of Z_i
fn expectation_z(state: &State, i: usize) -> f64 {
    let mut state_copy = state.clone();
    
    // Create a temporary circuit to apply Z gate to specific qubit
    let mut circuit = Circuit::new(state.num_qubits);
    circuit.z(i);
    
    // Execute the circuit on our state copy
    circuit.execute(&mut state_copy);
    
    // Calculate inner product
    state.vector.dot(&state_copy.vector).re
}