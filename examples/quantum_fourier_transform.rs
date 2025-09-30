use logosq::algorithms::qft;
use logosq::circuits::Circuit;

fn main() {
    let num_qubits = 3; // Example with 3 qubits
    let results = quantum_fourier_transform_example(num_qubits);
    println!("Measurement results after QFT: {:?}", results);
}
// how to combine algorithms
pub fn quantum_fourier_transform_example(num_qubits: usize) -> Vec<usize> {
    // Create a circuit that demonstrates QFT
    let mut circuit = Circuit::new(num_qubits);

    // Put qubits into superposition
    for i in 0..num_qubits {
        circuit.h(i);
    }

    // Apply QFT
    let qft_circuit = qft::create_circuit(num_qubits);
    circuit.compose(&qft_circuit);

    // Measure all qubits
    circuit.execute_and_measure()
}
