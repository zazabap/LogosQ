<!-- # Quantum Computing Library in Rust -->

<div align="center">
  <h1>
    <img src="docs/LogosQ.png" alt="LogosQ Logo" width="80" height="80" style="vertical-align: middle;">
    LogosQ: Quantum Computing Library in Rust
  </h1>
</div>

## Overview
This project is a quantum computing library developed in Rust, inspired by existing libraries such as QPP in C++, Yao.jl in Julia, and Pennylane in Python. The library aims to provide a comprehensive set of tools for quantum computing, including quantum gates, circuits, states, algorithms, and noise simulation.

λόγος (lógos) comes from Greek means the logic.

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


Add LogosQ to your project by adding the following to your `Cargo.toml`:

```toml
[dependencies]
logosq = "0.1.3"
```
or clone the repository and navigate to the project directory:

```bash
git clone https://github.com/zazabap/logosq.git
cd logosq
cargo build
```

## Usage
To use the library, you can include it in your Rust project by adding it as a dependency in your `Cargo.toml` file. Here’s an example of how to create and manipulate basic quantum gates

The `examples` directory contains several example files demonstrating how to use the library:

- **basic_gates.rs**: Example of creating and manipulating basic quantum gates.
- **quantum_teleportation.rs**: Implementation of the quantum teleportation algorithm.
- **grover_algorithm.rs**: Showcase of Grover's search algorithm.

## Development Roadmap
#### Completed
- ✅ Core data structures for quantum gates, states, and circuits
- ✅ Implementation of basic quantum gates and their operations
- ✅ Quantum state representation and manipulation
- ✅ Circuit model for combining gates and executing on states
- ✅ Circuit visualization tools (text and SVG)
- ✅ State visualization tools

#### In Progress
- 🚧 Intermediate Representation (IR) for circuit optimization
- 🚧 More quantum algorithms implementations
  - Quantum Fourier Transform
  - Variational Quantum Circuit
  - Quantum Gradient Descent
  -
- 🚧 Comprehensive testing suite
- 🚧 Page for documentation with example
- 🚧 Noise models and error simulation

#### Planned
- 📅 Hardware backend interfaces
- 📅 Advanced optimization techniques
- 📅 Quantum machine learning modules
- 📅 Distributed simulation capabilities
- 📅 Integration with other quantum frameworks

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## FAQ

1. Some bugs might appear when using the crate, one of the error related with missing package in libfontconfig1-dev, in such case, running on debian/linux (tested on ubuntu) could fix the build error.
```
sudo apt-get update
sudo apt-get install -y libfontconfig1-dev pkg-config
```
