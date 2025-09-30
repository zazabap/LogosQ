use logosq::algorithms::qft;
use logosq::circuits::Circuit;
use logosq::vis::circuit_text;
use logosq::vis::save_circuit_svg;

fn main() {
    let num_qubits = 5; // Example with 5 qubits
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

    print!("\nCircuit diagram:\n");
    print!("{}", circuit_text(&circuit));
    save_circuit_svg(&circuit, "quantum_fourier_transform.svg").unwrap_or_else(|e| {
        println!("Failed to save circuit diagram: {}", e);
    });

    // Measure all qubits
    circuit.execute_and_measure()
}
