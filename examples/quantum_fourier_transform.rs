use logosq::algorithms::qft;
use logosq::circuits::Circuit;
use logosq::prelude::*;
use logosq::vis::{circuit_text, save_circuit_svg, Visualizable};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let num_qubits = env_usize("QFT_QUBITS", 10);
    let shots = env_usize("QFT_SHOTS", 1024);

    let mps_config = MpsConfig {
        max_bond_dim: env_usize("MPS_MAX_BOND", 64),
        truncation_threshold: env_f64("MPS_TRUNC_EPS", 1e-8),
    };

    println!(
        "Comparing QFT implementations on {} qubits ({} shots)...",
        num_qubits, shots
    );

    let dense = run_qft_dense(num_qubits, shots);
    let mps = run_qft_mps(num_qubits, shots, mps_config);

    print_run_summary(&dense, num_qubits);
    print_run_summary(&mps, num_qubits);
    println!(
        "Runtime summary (ms): dense = {:.2}, mps = {:.2}",
        dense.runtime_ms, mps.runtime_ms
    );
}

pub fn quantum_fourier_transform_example(
    num_qubits: usize,
    shots: usize,
    backend: &str,
    mps_config: MpsConfig,
) -> HashMap<usize, usize> {
    match backend {
        "mps" | "tensor" => run_qft_mps(num_qubits, shots, mps_config).counts,
        _ => run_qft_dense(num_qubits, shots).counts,
    }
}

struct QftRun {
    backend: &'static str,
    counts: HashMap<usize, usize>,
    runtime_ms: f64,
    shots: usize,
}

fn run_qft_dense(num_qubits: usize, shots: usize) -> QftRun {
    let start = Instant::now();

    let mut state = State::zero_state(num_qubits);
    let mut circuit = Circuit::new(num_qubits);
    circuit.x(0);
    circuit
        .execute(&mut state)
        .expect("Failed to prepare input state");

    println!("Dense backend | initial state:\n{}", state.visualize());
    qft::apply(&mut state);
    println!("After QFT:\n{}", state.visualize());
    qft::apply_inverse(&mut state);
    println!("After inverse QFT:\n{}", state.visualize());

    let counts = sample_counts(&state, shots);
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;

    QftRun {
        backend: "dense",
        counts,
        runtime_ms,
        shots,
    }
}

fn run_qft_mps(num_qubits: usize, shots: usize, config: MpsConfig) -> QftRun {
    let start = Instant::now();

    let mut mps_state = MpsState::zero_state(num_qubits, config);
    mps_state.apply_pauli_x(0);
    println!(
        "MPS backend | initial state:\n{}",
        mps_state.to_dense_state().visualize()
    );

    apply_qft_mps(&mut mps_state);
    println!(
        "After QFT (MPS):\n{}",
        mps_state.to_dense_state().visualize()
    );

    apply_inverse_qft_mps(&mut mps_state);
    let dense_state = mps_state.to_dense_state();
    println!("After inverse QFT (MPS):\n{}", dense_state.visualize());

    let counts = sample_counts(&dense_state, shots);
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;

    QftRun {
        backend: "mps",
        counts,
        runtime_ms,
        shots,
    }
}

fn apply_qft_mps(state: &mut MpsState) {
    let num_qubits = state.num_qubits();
    for i in 0..num_qubits {
        state.apply_hadamard(i);
        for j in (i + 1)..num_qubits {
            let angle = PI / (1 << (j - i)) as f64;
            state.apply_controlled_phase(i, j, angle);
        }
    }

    for i in 0..num_qubits / 2 {
        swap_qubits(state, i, num_qubits - 1 - i);
    }
}

fn apply_inverse_qft_mps(state: &mut MpsState) {
    let num_qubits = state.num_qubits();

    for i in 0..num_qubits / 2 {
        swap_qubits(state, i, num_qubits - 1 - i);
    }

    for i in (0..num_qubits).rev() {
        for j in (i + 1)..num_qubits {
            let angle = -PI / (1 << (j - i)) as f64;
            state.apply_controlled_phase(i, j, angle);
        }
        state.apply_hadamard(i);
    }
}

fn sample_counts(state: &State, shots: usize) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    for _ in 0..shots {
        let outcome = state.measure();
        *counts.entry(outcome).or_insert(0) += 1;
    }
    counts
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

fn print_run_summary(run: &QftRun, num_qubits: usize) {
    println!(
        "QFT measurement counts (backend={}, shots={}, runtime={:.2} ms):",
        run.backend, run.shots, run.runtime_ms
    );
    let mut sorted: Vec<(usize, usize)> = run.counts.iter().map(|(k, v)| (*k, *v)).collect();
    sorted.sort_by_key(|(state, _)| *state);
    for (state, count) in sorted {
        println!("|{:0width$b}⟩ : {}", state, count, width = num_qubits);
    }
}

fn swap_qubits(state: &mut MpsState, left: usize, right: usize) {
    if left == right {
        return;
    }

    let (left_idx, mut right_idx) = if left < right { (left, right) } else { (right, left) };
    assert!(
        right_idx < state.num_qubits(),
        "Swap indices must be within range"
    );

    // Bring the right qubit next to the left qubit using nearest-neighbor swaps,
    // tracking the swap locations so we can restore the intermediate ordering later.
    let mut swaps = Vec::new();
    while right_idx > left_idx + 1 {
        let swap_site = right_idx - 1;
        state.apply_swap_gate(swap_site);
        swaps.push(swap_site);
        right_idx -= 1;
    }

    // Swap the now-adjacent qubits.
    state.apply_swap_gate(left_idx);

    // Undo the temporary swaps in reverse order to restore intermediate qubits.
    for site in swaps.into_iter().rev() {
        state.apply_swap_gate(site);
    }
}

/// Renders and saves a QFT circuit diagram for reference.
pub fn qft_c2() {
    let num_qubits = 3;
    let circuit = qft::create_circuit(num_qubits);

    print!("\nCircuit diagram:\n");
    print!("{}", circuit_text(&circuit));
    save_circuit_svg(&circuit, "quantum_fourier_transform.svg").unwrap_or_else(|e| {
        println!("Failed to save circuit diagram: {}", e);
    });
}
