// Random Quantum Circuits implementation

use crate::circuits::Circuit;
use crate::states::State;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use rand::{thread_rng, Rng};
use std::f64::consts::PI;

/// Random gate types that can be applied
#[derive(Debug, Clone)]
pub enum RandomGateType {
    Single,   // Single-qubit gates like H, X, Y, Z, S, T
    TwoQubit, // Two-qubit gates like CNOT, CZ, SWAP
    Rotation, // Rotation gates with random angles
}

/// Configuration for random circuit generation
pub struct RandomCircuitConfig {
    pub num_qubits: usize,
    pub depth: usize,
    pub gate_types: Vec<(RandomGateType, f64)>, // Gate type and probability
    pub seed: Option<u64>,                      // Optional seed for reproducibility
}

impl Default for RandomCircuitConfig {
    fn default() -> Self {
        Self {
            num_qubits: 3,
            depth: 10,
            gate_types: vec![
                (RandomGateType::Single, 0.5),
                (RandomGateType::TwoQubit, 0.3),
                (RandomGateType::Rotation, 0.2),
            ],
            seed: None,
        }
    }
}

/// Generates a random quantum circuit according to the provided configuration
pub fn generate(config: &RandomCircuitConfig) -> Circuit {
    let mut circuit =
        Circuit::new(config.num_qubits).with_name(&format!("Random Circuit (d={})", config.depth));

    let mut rng = match config.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };

    for _ in 0..config.depth {
        // Select a random gate type based on probabilities
        let gate_type = select_gate_type(&config.gate_types, &mut rng);

        match gate_type {
            RandomGateType::Single => {
                add_random_single_qubit_gate(&mut circuit, config.num_qubits, &mut rng);
            }
            RandomGateType::TwoQubit => {
                if config.num_qubits >= 2 {
                    add_random_two_qubit_gate(&mut circuit, config.num_qubits, &mut rng);
                } else {
                    // Fall back to single-qubit gate if not enough qubits
                    add_random_single_qubit_gate(&mut circuit, config.num_qubits, &mut rng);
                }
            }
            RandomGateType::Rotation => {
                add_random_rotation_gate(&mut circuit, config.num_qubits, &mut rng);
            }
        }
    }

    circuit
}

/// Helper functions for gate selection and addition
fn select_gate_type<R: Rng>(gate_types: &[(RandomGateType, f64)], rng: &mut R) -> RandomGateType {
    // Extract probabilities for weighted selection
    let weights: Vec<f64> = gate_types.iter().map(|(_, prob)| *prob).collect();

    // Create a weighted distribution
    let dist = match WeightedIndex::new(&weights) {
        Ok(dist) => dist,
        Err(_) => {
            // Fallback to uniform distribution if weights are invalid
            return RandomGateType::Single;
        }
    };

    // Select an index based on the distribution
    let selected_idx = dist.sample(rng);
    gate_types[selected_idx].0.clone()
}

/// Adds a random single-qubit gate to the circuit
fn add_random_single_qubit_gate<R: Rng>(circuit: &mut Circuit, num_qubits: usize, rng: &mut R) {
    // Select a random qubit
    let qubit = rng.gen_range(0..num_qubits);

    // Select a random single-qubit gate
    match rng.gen_range(0..6) {
        0 => {
            circuit.h(qubit);
        }
        1 => {
            circuit.x(qubit);
        }
        2 => {
            circuit.y(qubit);
        }
        3 => {
            circuit.z(qubit);
        }
        4 => {
            circuit.s(qubit);
        }
        5 => {
            circuit.t(qubit);
        }
        _ => unreachable!(),
    }
}

/// Adds a random two-qubit gate to the circuit
fn add_random_two_qubit_gate<R: Rng>(circuit: &mut Circuit, num_qubits: usize, rng: &mut R) {
    // Ensure we have at least 2 qubits
    if num_qubits < 2 {
        return;
    }

    // Select two different random qubits
    let qubit1 = rng.gen_range(0..num_qubits);
    let mut qubit2 = rng.gen_range(0..num_qubits);

    // Make sure qubits are different
    while qubit2 == qubit1 {
        qubit2 = rng.gen_range(0..num_qubits);
    }

    // Select a random two-qubit gate
    match rng.gen_range(0..3) {
        0 => {
            circuit.cnot(qubit1, qubit2);
        }
        1 => {
            // Controlled-Z implementation
            circuit.h(qubit2);
            circuit.cnot(qubit1, qubit2);
            circuit.h(qubit2);
        }
        2 => {
            // SWAP implementation using 3 CNOTs
            circuit.cnot(qubit1, qubit2);
            circuit.cnot(qubit2, qubit1);
            circuit.cnot(qubit1, qubit2);
        }
        _ => unreachable!(),
    }
}

/// Adds a random rotation gate to the circuit
fn add_random_rotation_gate<R: Rng>(circuit: &mut Circuit, num_qubits: usize, rng: &mut R) {
    // Select a random qubit
    let qubit = rng.gen_range(0..num_qubits);

    // Generate a random angle between 0 and 2π
    let angle = rng.gen_range(0.0..2.0 * PI);

    // Select a random rotation axis
    match rng.gen_range(0..3) {
        0 => {
            circuit.rx(qubit, angle);
        }
        1 => {
            circuit.ry(qubit, angle);
        }
        2 => {
            circuit.rz(qubit, angle);
        }
        _ => unreachable!(),
    }
}

/// Applies a random circuit to a quantum state
pub fn apply(state: &mut State, config: &RandomCircuitConfig) {
    let circuit = generate(config);
    circuit.execute(state);
}

/// Generates a random circuit with specified number of qubits and depth
/// using default probability distribution for gate types
pub fn generate_simple(num_qubits: usize, depth: usize, seed: Option<u64>) -> Circuit {
    let config = RandomCircuitConfig {
        num_qubits,
        depth,
        gate_types: vec![
            (RandomGateType::Single, 0.4),
            (RandomGateType::TwoQubit, 0.4),
            (RandomGateType::Rotation, 0.2),
        ],
        seed,
    };

    generate(&config)
}

/// Creates a circuit that approximates a 2-design
/// (A circuit that approximately achieves Haar randomness)
pub fn generate_2design(num_qubits: usize) -> Circuit {
    // For a 2-design, we need O(n²) depth where n is the number of qubits
    let depth = 2 * num_qubits * num_qubits;

    // Use a specific distribution optimized for 2-designs
    let config = RandomCircuitConfig {
        num_qubits,
        depth,
        gate_types: vec![
            (RandomGateType::Single, 0.3),
            (RandomGateType::TwoQubit, 0.5),
            (RandomGateType::Rotation, 0.2),
        ],
        seed: None,
    };

    let circuit = generate(&config).with_name(&format!(
        "2-Design Random Circuit (n={}, d={})",
        num_qubits, depth
    ));
    circuit
}

/// Generates a circuit for quantum supremacy experiments
/// Similar to the Google Sycamore architecture
pub fn generate_supremacy_circuit(num_qubits: usize, depth: usize) -> Circuit {
    let mut circuit =
        Circuit::new(num_qubits).with_name(&format!("Supremacy Circuit (d={})", depth));

    let mut rng = thread_rng();

    // For each layer
    for layer in 0..depth {
        // Apply single-qubit gates to all qubits
        for q in 0..num_qubits {
            // Random rotation angles
            let theta = rng.gen_range(0.0..2.0 * PI);
            let phi = rng.gen_range(0.0..2.0 * PI);
            let lambda = rng.gen_range(0.0..2.0 * PI);

            // Apply single-qubit rotations (universal single-qubit gate)
            circuit.rz(q, phi);
            circuit.ry(q, theta);
            circuit.rz(q, lambda);
        }

        // Apply two-qubit gates in alternating pattern
        // Even layers: connect (0,1), (2,3), ...
        // Odd layers: connect (1,2), (3,4), ...
        let start_idx = layer % 2;

        for q in (start_idx..num_qubits - 1).step_by(2) {
            circuit.cnot(q, q + 1);
        }
    }

    circuit
}
