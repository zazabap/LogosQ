# Quantum Computing Library in Rust

## Overview
This project is a quantum computing library developed in Rust, inspired by existing libraries such as QPP in C++, Yao.jl in Julia, and Pennylane in Python. The library aims to provide a comprehensive set of tools for quantum computing, including quantum gates, circuits, states, algorithms, and noise simulation.

## Project Structure
The project is organized into several modules, each responsible for a specific aspect of quantum computing:

- **src/lib.rs**: Main library entry point, defining the public API.
- **src/gates**: Handles quantum gates and their operations.
- **src/circuits**: Defines quantum circuits and their manipulation.
- **src/states**: Manages quantum states and their representation.
- **src/algorithms**: Implements various quantum algorithms.
- **src/noise**: Simulates noise in quantum systems.
- **src/utils**: Contains utility functions and types.

## Installation
To build and run the project, ensure you have Rust installed on your machine. You can install Rust using [rustup](https://rustup.rs/).

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/zazabap/qforge.git
cd qforge
```

Then, build the project using Cargo:

```bash
cargo build
```

## Usage
To use the library, you can include it in your Rust project by adding it as a dependency in your `Cargo.toml` file. Here’s an example of how to create and manipulate basic quantum gates:

```rust
use qforge::gates::{Gate, Hadamard};

fn main() {
    let h_gate = Hadamard::new();
    // Use the Hadamard gate...
}
```

## Examples
The `examples` directory contains several example files demonstrating how to use the library:

- **basic_gates.rs**: Example of creating and manipulating basic quantum gates.
- **quantum_teleportation.rs**: Implementation of the quantum teleportation algorithm.
- **grover_algorithm.rs**: Showcase of Grover's search algorithm.

## Development Roadmap
1. Define the core data structures (gates, states, circuits).
2. Implement basic quantum gates and their operations.
3. Develop quantum state representation and manipulation functions.
4. Create the circuit model to combine gates and states.
5. Implement key quantum algorithms (e.g., Grover's, teleportation).
6. Add noise simulation features for realistic quantum computing.
7. Write comprehensive tests and benchmarks for performance evaluation.
8. Document the API and provide usage examples in the README and docs.
9. Gather feedback and iterate on the library based on user needs and performance metrics.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.