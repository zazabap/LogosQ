use logosq::algorithms::qft;
use logosq::circuits::Circuit;
use logosq::vis::circuit_text;
use logosq::vis::save_circuit_svg;
use logosq::vis::Visualizable;
use logosq::State;

fn main() {
    let num_qubits = 5; // Example with 5 qubits
    let results = quantum_fourier_transform_example(num_qubits);
    println!("Measurement results after QFT: {:?}", results);
}
// how to combine algorithms
pub fn quantum_fourier_transform_example(num_qubits: usize) -> Vec<usize> {

    let mut state = State::zero_state(num_qubits);
    // Create a circuit that demonstrates QFT
    let mut circuit = Circuit::new(num_qubits);
    circuit.x(0);
    circuit.execute(&mut state);

    // Apply QFT
    qft::apply(&mut state);
    print!("State after QFT: {}\n", state.visualize());
    qft::apply_inverse(&mut state);
    print!("State after inverse QFT: {}\n", state.visualize());
    circuit.execute_and_measure()
}


pub fn qft_c2 (){
    let num_qubits = 3; // Example with 3 qubits
    let circuit = Circuit::new(num_qubits);
    // circuit.compose(&qft::create_circuit(num_qubits));

    print!("\nCircuit diagram:\n");
    print!("{}", circuit_text(&circuit));
    save_circuit_svg(&circuit, "quantum_fourier_transform.svg").unwrap_or_else(|e| {
        println!("Failed to save circuit diagram: {}", e);
    });

    // Measure all qubits
    circuit.execute_and_measure();
}