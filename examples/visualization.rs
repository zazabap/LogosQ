use logosq::circuits::Circuit;
use logosq::gates::matrix::{cnot_gate, h_gate, rz_gate, swap_gate, toffoli_gate, x_gate};

use logosq::vis::{circuit_text, save_circuit_svg};
// Use the trait methods
use logosq::vis::Visualizable;

fn main() -> std::io::Result<()> {
    // Example 1: Bell State Circuit
    println!("=== Example 1: Bell State Circuit ===");
    let mut bell_circuit = Circuit::new(2).with_name("Bell State");

    // Using the single_qubit::h_gate() function that returns a MatrixGate
    let h = h_gate();
    bell_circuit.add_operation(h, vec![0], "H").unwrap();

    // Using the multi_qubit::cnot_gate() function
    let cnot = cnot_gate();
    bell_circuit.add_operation(cnot, vec![0, 1], "CNOT").unwrap();

    // Using direct functions
    println!("Text visualization:");
    println!("{}", circuit_text(&bell_circuit));

    // Save SVG to file
    save_circuit_svg(&bell_circuit, "bell_circuit.svg")?;
    println!("SVG visualization saved to 'bell_circuit.svg'");

    // Example 2: GHZ State Circuit
    println!("\n=== Example 2: GHZ State Circuit ===");
    let mut ghz_circuit = Circuit::new(3).with_name("GHZ State Preparation");

    // Add H gate to first qubit
    ghz_circuit.add_operation(h_gate(), vec![0], "H").unwrap();

    // Add CNOT gates to create GHZ state
    ghz_circuit.add_operation(cnot_gate(), vec![0, 1], "CNOT").unwrap();
    ghz_circuit.add_operation(cnot_gate(), vec![1, 2], "CNOT").unwrap();

    // Using trait methods
    println!("Text visualization:");
    println!("{}", ghz_circuit.visualize());

    // Save visualization
    ghz_circuit.save_visualization("ghz_circuit.svg")?;
    println!("SVG visualization saved to 'ghz_circuit.svg'");

    // Example 3: Quantum Fourier Transform (QFT) - 3 qubit
    println!("\n=== Example 3: Quantum Fourier Transform (3 qubits) ===");
    let mut qft_circuit = Circuit::new(3).with_name("3-Qubit QFT");

    // QFT implementation
    // Qubit 0
    qft_circuit.add_operation(h_gate(), vec![0], "H").unwrap();
    qft_circuit.add_operation(rz_gate(std::f64::consts::PI / 2.0), vec![1, 0], "Rz(π/2)").unwrap();
    qft_circuit.add_operation(rz_gate(std::f64::consts::PI / 4.0), vec![2, 0], "Rz(π/4)").unwrap();

    // Qubit 1
    qft_circuit.add_operation(h_gate(), vec![1], "H").unwrap();
    qft_circuit.add_operation(rz_gate(std::f64::consts::PI / 2.0), vec![2, 1], "Rz(π/2)").unwrap();

    // Qubit 2
    qft_circuit.add_operation(h_gate(), vec![2], "H").unwrap();

    // Swap qubits 0 and 2
    qft_circuit.add_operation(swap_gate(), vec![0, 2], "SWAP").unwrap();

    println!("Text visualization:");
    println!("{}", qft_circuit.visualize());
    qft_circuit.save_visualization("qft_circuit.svg")?;
    println!("SVG visualization saved to 'qft_circuit.svg'");

    // Example 4: Complex circuit with Toffoli
    println!("\n=== Example 4: Complex Circuit with Toffoli Gate ===");
    let mut complex_circuit = Circuit::new(4).with_name("Complex Example");

    // Initial layer of Hadamards
    for i in 0..4 {
        complex_circuit.add_operation(h_gate(), vec![i], "H").unwrap();
    }

    // Add some X gates
    let x_gate = x_gate();
    complex_circuit.add_operation(x_gate.clone(), vec![1], "X").unwrap();
    complex_circuit.add_operation(x_gate, vec![3], "X").unwrap();

    // Add CNOT gates
    let cnot_gate = cnot_gate();
    complex_circuit.add_operation(cnot_gate.clone(), vec![0, 1], "CNOT").unwrap();
    complex_circuit.add_operation(cnot_gate.clone(), vec![2, 3], "CNOT").unwrap();

    // Add Toffoli gate
    complex_circuit.add_operation(toffoli_gate(), vec![0, 1, 2], "Toffoli").unwrap();

    // More gates
    complex_circuit.add_operation(h_gate(), vec![0], "H").unwrap();
    complex_circuit.add_operation(cnot_gate, vec![3, 0], "CNOT").unwrap();

    println!("Text visualization:");
    println!("{}", complex_circuit.visualize());
    complex_circuit.save_visualization("complex_circuit.svg")?;
    println!("SVG visualization saved to 'complex_circuit.svg'");

    // Optional: View the circuits in a browser
    println!("\nWould you like to view the circuits in your browser? (Uncomment the lines below)");
    // bell_circuit.view()?;
    // ghz_circuit.view()?;
    // qft_circuit.view()?;
    // complex_circuit.view()?;

    Ok(())
}
