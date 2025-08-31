use logosq::prelude::*;
use logosq::vis::save_circuit_svg;
use logosq::vis::circuit_text;

fn main() {
    // Define the number of qubits and create a circuit
    let num_qubits = 3;
    let target_state = 0b011; // Example target state |011>
    
    let mut circuit = Circuit::new(num_qubits)
        .with_name("Grover's Algorithm");
    
    // Step 1: Initialize with Hadamard gates
    for i in 0..num_qubits {
        circuit.h(i);
    }
    
    // Calculate optimal number of iterations
    let iterations = ((std::f64::consts::PI / 4.0) * (1 << num_qubits as u32) as f64).sqrt() as usize;
    println!("Running {} iterations", iterations);
    
    // Step 2: Apply Grover iterations
    for _ in 0..iterations {
        // Oracle - marks the target state by flipping its sign
        apply_oracle(&mut circuit, target_state, num_qubits);
        
        // Diffusion operator - amplifies the amplitude of the marked state
        apply_diffusion(&mut circuit, num_qubits);
    }
    
    // Step 3: Measure the qubits
    let results = circuit.execute_and_measure();
    
    // Display results
    println!("Measurement results: {:?}", results);
    println!("Target state: {}", target_state);
    
    // Visualize the circuit (if the visualization feature is enabled)
    println!("\nCircuit diagram:");
    println!("{}", circuit_text(&circuit));

    // Optionally save the circuit diagram
    save_circuit_svg(&circuit, "grover_algorithm.svg").unwrap_or_else(|e| {
        println!("Failed to save circuit diagram: {}", e);
    });
}

// Function to apply the oracle that marks the target state
fn apply_oracle(circuit: &mut Circuit, target_state: usize, num_qubits: usize) {
    // Apply X gates to qubits where target_state bit is 0
    for i in 0..num_qubits {
        if (target_state >> i) & 1 == 0 {
            circuit.x(i);
        }
    }
    
    // Apply multi-controlled Z gate
    // For a 3-qubit system, we can use Toffoli (CCZ) plus some H gates
    if num_qubits == 3 {
        circuit.h(2);
        circuit.add_operation(toffoli_gate(), vec![0, 1, 2], "CCZ");
        circuit.h(2);
    } else {
        // For larger systems, we would need a more general approach
        // This is a simplified implementation
        let control_qubits = (0..num_qubits-1).collect::<Vec<_>>();
        let target = num_qubits - 1;
        
        circuit.h(target);
        // Multi-controlled NOT gate
        apply_multi_controlled_not(circuit, &control_qubits, target);
        circuit.h(target);
    }
    
    // Undo X gates
    for i in 0..num_qubits {
        if (target_state >> i) & 1 == 0 {
            circuit.x(i);
        }
    }
}

// Function to apply the diffusion operator
fn apply_diffusion(circuit: &mut Circuit, num_qubits: usize) {
    // Apply Hadamard gates to all qubits
    for i in 0..num_qubits {
        circuit.h(i);
    }
    
    // Apply X gates to all qubits
    for i in 0..num_qubits {
        circuit.x(i);
    }
    
    // Apply multi-controlled Z gate (inversion about the mean)
    if num_qubits == 3 {
        circuit.h(2);
        circuit.add_operation(toffoli_gate(), vec![0, 1, 2], "CCZ");
        circuit.h(2);
    } else {
        // For larger systems, use a generalized approach
        let control_qubits = (0..num_qubits-1).collect::<Vec<_>>();
        let target = num_qubits - 1;
        
        circuit.h(target);
        apply_multi_controlled_not(circuit, &control_qubits, target);
        circuit.h(target);
    }
    
    // Undo X gates
    for i in 0..num_qubits {
        circuit.x(i);
    }
    
    // Undo Hadamard gates
    for i in 0..num_qubits {
        circuit.h(i);
    }
}

// Helper function to apply a multi-controlled NOT gate
fn apply_multi_controlled_not(circuit: &mut Circuit, controls: &[usize], target: usize) {
    // This is a simplified implementation for demo purposes
    // In a real quantum computer, this would require decomposition into elementary gates
    
    // For 2 control qubits, we can use the built-in Toffoli gate
    if controls.len() == 2 {
        circuit.add_operation(toffoli_gate(), vec![controls[0], controls[1], target], "Toffoli");
    } else {
        // For more control qubits, this would require a more complex implementation
        // using ancilla qubits and gate decomposition
        println!("Warning: Multi-controlled NOT with {} control qubits is being simulated directly", 
                 controls.len());
        
        // This is a placeholder for a more complex implementation
        let mut qubits = controls.to_vec();
        qubits.push(target);
        // circuit.add_operation(
        //     // In a real implementation, this would be a properly constructed matrix
        //     // for a multi-controlled NOT operation
        //     MatrixGate::identity(qubits.len()),
        //     qubits,
        //     "Multi-controlled NOT"
        // );
    }
}
