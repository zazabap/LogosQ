// XYZ-Heisenberg Model Dynamics implementation

use crate::circuits::Circuit;
use crate::states::State;
use num_complex::Complex64;

/// Parameters for the XYZ-Heisenberg model
pub struct HeisenbergParameters {
    pub jx: f64,             // Coupling strength in X direction
    pub jy: f64,             // Coupling strength in Y direction
    pub jz: f64,             // Coupling strength in Z direction
    pub external_field: f64, // External magnetic field strength
    pub time_steps: usize,   // Number of time steps for evolution
    pub dt: f64,             // Time step size
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
    let mut circuit = Circuit::new(num_qubits).with_name("XYZ-Heisenberg Model");

    // Time evolution using Trotter-Suzuki decomposition
    for _ in 0..params.time_steps {
        // Add XX interactions
        for i in 0..num_qubits - 1 {
            add_xx_interaction(&mut circuit, i, i + 1, params.jx * params.dt);
        }

        // Add YY interactions
        for i in 0..num_qubits - 1 {
            add_yy_interaction(&mut circuit, i, i + 1, params.jy * params.dt);
        }

        // Add ZZ interactions
        for i in 0..num_qubits - 1 {
            add_zz_interaction(&mut circuit, i, i + 1, params.jz * params.dt);
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
    let circuit = create_circuit(state.num_qubits(), params);
    circuit.execute(state).expect("Circuit execution failed");
}

// Helper functions for adding interactions
fn add_xx_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Direct exp(-i * strength * X⊗X) using the native RXX rotation (angle convention matches 2*strength)
    circuit.rxx(qubit1, qubit2, 2.0 * strength);
}

fn add_yy_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Direct exp(-i * strength * Y⊗Y)
    circuit.ryy(qubit1, qubit2, 2.0 * strength);
}

fn add_zz_interaction(circuit: &mut Circuit, qubit1: usize, qubit2: usize, strength: f64) {
    // Direct exp(-i * strength * Z⊗Z)
    circuit.rzz(qubit1, qubit2, 2.0 * strength);
}

/// Calculate energy of the XYZ-Heisenberg model for a given state
pub fn calculate_energy(state: &State, params: &HeisenbergParameters) -> f64 {
    let num_qubits = state.num_qubits();
    let mut energy = 0.0;

    // Calculate nearest-neighbor interaction energies
    for i in 0..num_qubits - 1 {
        // Calculate XX interaction energy: Jx * <ψ|X_i X_{i+1}|ψ>
        let mut xx_state = state.clone();
        let mut xx_circuit = Circuit::new(num_qubits);
        xx_circuit.x(i);
        xx_circuit.x(i + 1);
        xx_circuit
            .execute(&mut xx_state)
            .expect("Circuit execution failed");
        let xx_energy = params.jx * state.vector().dot(xx_state.vector()).re;
        energy += xx_energy;

        // Calculate YY interaction energy: Jy * <ψ|Y_i Y_{i+1}|ψ>
        let mut yy_state = state.clone();
        let mut yy_circuit = Circuit::new(num_qubits);
        yy_circuit.y(i);
        yy_circuit.y(i + 1);
        yy_circuit
            .execute(&mut yy_state)
            .expect("Circuit execution failed");
        let yy_energy = params.jy * state.vector().dot(yy_state.vector()).re;
        energy += yy_energy;

        // Calculate ZZ interaction energy: Jz * <ψ|Z_i Z_{i+1}|ψ>
        let mut zz_state = state.clone();
        let mut zz_circuit = Circuit::new(num_qubits);
        zz_circuit.z(i);
        zz_circuit.z(i + 1);
        zz_circuit
            .execute(&mut zz_state)
            .expect("Circuit execution failed");
        let zz_energy = params.jz * state.vector().dot(zz_state.vector()).re;
        energy += zz_energy;
    }

    // Calculate external field contribution: h * <ψ|Z_i|ψ>
    if params.external_field != 0.0 {
        for i in 0..num_qubits {
            let mut z_state = state.clone();
            let mut z_circuit = Circuit::new(num_qubits);
            z_circuit.z(i);
            z_circuit
                .execute(&mut z_state)
                .expect("Circuit execution failed");
            let z_energy = params.external_field * state.vector().dot(z_state.vector()).re;
            energy += z_energy;
        }
    }

    energy
}

/// Calculate energy using a more efficient approach with expectation values
pub fn calculate_energy_efficient(state: &State, params: &HeisenbergParameters) -> f64 {
    let num_qubits = state.num_qubits();
    let mut energy = 0.0;

    // Calculate nearest-neighbor interaction energies
    for i in 0..num_qubits - 1 {
        energy += params.jx * expectation_xx(state, i, i + 1);
        energy += params.jy * expectation_yy(state, i, i + 1);
        energy += params.jz * expectation_zz(state, i, i + 1);
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
/// Efficient direct computation: ⟨ψ|X_i X_j|ψ⟩ = Σ_k ψ_k* · ψ_{k XOR (1<<i) XOR (1<<j)}
fn expectation_xx(state: &State, i: usize, j: usize) -> f64 {
    let vector = state.vector();
    let n = state.num_qubits();
    let mask_i = 1 << (n - 1 - i);
    let mask_j = 1 << (n - 1 - j);

    let mut result = Complex64::new(0.0, 0.0);
    for k in 0..vector.len() {
        let flipped = k ^ mask_i ^ mask_j;
        result += vector[k].conj() * vector[flipped];
    }
    result.re
}

/// Calculate expectation value of Y_i Y_j
/// Efficient direct computation: ⟨ψ|Y_i Y_j|ψ⟩ = Σ_k ψ_k* · phase · ψ_{k XOR (1<<i) XOR (1<<j)}
/// where phase = i if bit_i=0 else -i, times i if bit_j=0 else -i
/// This is because Y|0⟩ = i|1⟩ and Y|1⟩ = -i|0⟩
fn expectation_yy(state: &State, i: usize, j: usize) -> f64 {
    let vector = state.vector();
    let n = state.num_qubits();
    let mask_i = 1 << (n - 1 - i);
    let mask_j = 1 << (n - 1 - j);

    let mut result = Complex64::new(0.0, 0.0);
    for k in 0..vector.len() {
        let bit_i = (k & mask_i) != 0;
        let bit_j = (k & mask_j) != 0;
        let flipped = k ^ mask_i ^ mask_j;

        // Y_i phase: i if bit_i=0 (|0⟩ -> i|1⟩), -i if bit_i=1 (|1⟩ -> -i|0⟩)
        // Y_j phase: i if bit_j=0, -i if bit_j=1
        // Since Y_i and Y_j commute, the total phase is the product
        let phase_i = if bit_i {
            Complex64::new(0.0, -1.0)
        } else {
            Complex64::new(0.0, 1.0)
        };
        let phase_j = if bit_j {
            Complex64::new(0.0, -1.0)
        } else {
            Complex64::new(0.0, 1.0)
        };
        let phase = phase_i * phase_j;

        result += vector[k].conj() * phase * vector[flipped];
    }
    result.re
}

/// Calculate expectation value of Z_i Z_j
/// Efficient direct computation: ⟨ψ|Z_i Z_j|ψ⟩ = Σ_k |ψ_k|^2 · (-1)^{bit_i + bit_j}
fn expectation_zz(state: &State, i: usize, j: usize) -> f64 {
    let vector = state.vector();
    let n = state.num_qubits();
    let mask_i = 1 << (n - 1 - i);
    let mask_j = 1 << (n - 1 - j);

    let mut result = 0.0;
    for k in 0..vector.len() {
        let bit_i = (k & mask_i) != 0;
        let bit_j = (k & mask_j) != 0;
        let sign = if (bit_i as usize + bit_j as usize) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        result += vector[k].norm_sqr() * sign;
    }
    result
}

/// Calculate expectation value of Z_i
/// Efficient direct computation: ⟨ψ|Z_i|ψ⟩ = Σ_k |ψ_k|^2 · (-1)^{bit_i}
fn expectation_z(state: &State, i: usize) -> f64 {
    let vector = state.vector();
    let n = state.num_qubits();
    let mask_i = 1 << (n - 1 - i);

    let mut result = 0.0;
    for k in 0..vector.len() {
        let bit_i = (k & mask_i) != 0;
        let sign = if bit_i { -1.0 } else { 1.0 };
        result += vector[k].norm_sqr() * sign;
    }
    result
}
