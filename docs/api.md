# API Documentation for logosq Quantum Computing Library

## Overview

The logosq library provides a comprehensive set of tools for quantum computing, including the ability to define and manipulate quantum gates, circuits, and states, as well as implement various quantum algorithms. This document outlines the public API of the library, detailing the available types, functions, and usage examples.

## Modules

### Gates

The `gates` module provides functionality for creating and manipulating quantum gates.

#### Types

- **Gate**: Represents a quantum gate.

#### Functions

- `fn pauli_x() -> Gate`: Returns the Pauli-X gate.
- `fn hadamard() -> Gate`: Returns the Hadamard gate.
- `fn cnot(control: usize, target: usize) -> Gate`: Returns a CNOT gate with specified control and target qubits.

### Circuits

The `circuits` module allows users to define quantum circuits.

#### Types

- **Circuit**: Represents a sequence of quantum gates.

#### Functions

- `fn new() -> Circuit`: Creates a new empty circuit.
- `fn add_gate(&mut self, gate: Gate)`: Adds a gate to the circuit.
- `fn measure(&self, qubit: usize) -> MeasurementResult`: Measures the specified qubit.

### States

The `states` module manages quantum states.

#### Types

- **State**: Represents a quantum state vector.

#### Functions

- `fn initialize(size: usize) -> State`: Initializes a quantum state of the given size.
- `fn apply_gate(&mut self, gate: &Gate)`: Applies a gate to the quantum state.

### Algorithms

The `algorithms` module implements various quantum algorithms.

#### Functions

- `fn grover_search(target: &State) -> Circuit`: Implements Grover's search algorithm.
- `fn quantum_teleportation(state: &State) -> Circuit`: Implements the quantum teleportation protocol.

### Noise

The `noise` module simulates noise in quantum systems.

#### Functions

- `fn depolarizing_noise(state: &State, probability: f64) -> State`: Applies depolarizing noise to the state.
- `fn amplitude_damping(state: &State) -> State`: Applies amplitude damping noise to the state.

### Utils

The `utils` module contains utility functions and types.

#### Functions

- `fn complex_add(a: Complex, b: Complex) -> Complex`: Adds two complex numbers.
- `fn normalize(state: &mut State)`: Normalizes the quantum state vector.

## Usage Example

```rust
use logosq::gates::{pauli_x, hadamard, cnot};
use logosq::circuits::Circuit;
use logosq::states::State;

fn main() {
    let mut circuit = Circuit::new();
    let initial_state = State::initialize(2);
    
    circuit.add_gate(hadamard());
    circuit.add_gate(cnot(0, 1));
    
    let final_state = initial_state.apply_gate(&circuit);
    let measurement = circuit.measure(1);
}
```

## Conclusion

This API documentation provides a high-level overview of the logosq library's capabilities. For more detailed information, please refer to the individual module documentation and usage examples provided in the `examples` directory.