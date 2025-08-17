// This file contains benchmarks to measure the performance of various library functions and algorithms.
// It uses the criterion crate for benchmarking.

// use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use qforge::gates::{Gate, Pauli, Hadamard, CNOT};
// use qforge::states::State;
// use qforge::circuits::Circuit;

// fn benchmark_gate_operations(c: &mut Criterion) {
//     c.bench_function("Pauli X gate", |b| {
//         let state = State::new(2); // Initialize a 2-qubit state
//         let gate = Pauli::X; // Create a Pauli X gate
//         b.iter(|| {
//             let mut new_state = state.clone();
//             gate.apply(&mut new_state); // Apply the gate
//             black_box(new_state);
//         });
//     });

//     c.bench_function("Hadamard gate", |b| {
//         let state = State::new(1); // Initialize a 1-qubit state
//         let gate = Hadamard; // Create a Hadamard gate
//         b.iter(|| {
//             let mut new_state = state.clone();
//             gate.apply(&mut new_state); // Apply the gate
//             black_box(new_state);
//         });
//     });

//     c.bench_function("CNOT gate", |b| {
//         let state = State::new(2); // Initialize a 2-qubit state
//         let gate = CNOT; // Create a CNOT gate
//         b.iter(|| {
//             let mut new_state = state.clone();
//             gate.apply(&mut new_state); // Apply the gate
//             black_box(new_state);
//         });
//     });
// }

// fn benchmark_circuit_execution(c: &mut Criterion) {
//     c.bench_function("Circuit execution", |b| {
//         let mut circuit = Circuit::new();
//         circuit.add_gate(Pauli::X, 0); // Add a Pauli X gate on qubit 0
//         circuit.add_gate(Hadamard, 1); // Add a Hadamard gate on qubit 1

//         let state = State::new(2); // Initialize a 2-qubit state
//         b.iter(|| {
//             let mut new_state = state.clone();
//             circuit.execute(&mut new_state); // Execute the circuit
//             black_box(new_state);
//         });
//     });
// }

// criterion_group!(benches, benchmark_gate_operations, benchmark_circuit_execution);
// criterion_main!(benches);