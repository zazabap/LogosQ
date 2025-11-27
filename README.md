<!-- # Quantum Computing Library in Rust -->

<div align="center">
  <h1>
    <img src="docs/LogosQ.png" alt="LogosQ Logo" width="80" height="80" style="vertical-align: middle;">
    LogosQ: Quantum Computing Library in Rust
  </h1>
</div>

This project is a quantum computing library developed in Rust, inspired by existing libraries such as QPP in C++, Yao.jl in Julia, and Pennylane in Python. The library aims to provide a comprehensive set of tools for quantum computing, including quantum gates, circuits, states, algorithms, and noise simulation.

λόγος (lógos) comes from Greek means the logic.

## Installation
To build and run the project, ensure you have Rust installed on your machine. You can install Rust using [rustup](https://rustup.rs/).


Add LogosQ to your project by adding the following to your `Cargo.toml`:

```toml
[dependencies]
logosq = "0.2.3"
```
or clone the repository and navigate to the project directory:

```bash
git clone https://github.com/zazabap/logosq.git
cd logosq
cargo build
```

## Usage
To use the library, you can include it in your Rust project by adding it as a dependency in your `Cargo.toml` file. Here’s an example of how to create and manipulate basic quantum gates

The `examples` directory contains several example files demonstrating how to use the library. The official [documentation](https://logosqbook.vercel.app/) carefully listed the API usage and give some great tutorial for the beginners. If you are completely new to quantum computing, feel free to checkout! 

Because LogosQ is written in Rust, the memory safety is the biggest advantage over the other library, and could be verified through benchmark. Please also feel free to checkout the [benchmark](https://github.com/zazabap/LogosQBenchmarks) we have compare with the Pennylane and other library! 

## Feature 
The most important feature comes from the implementation on the parameter shift rules and the ansatz structure, where such feature is rarely implemented for quantum computing library in Rust! You could create a parameterized ansatz/quantum circuit that helps you find out the optimal solution. Here are some basic instruction to create Variational Quantum Eigensolver, Quantum Natural Gradient and Ansatz in [Optimization Module](https://logosqbook.vercel.app/docs/modules/Optimization). 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes. For further details, please check the DevGuide.md

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=zazabap/LogosQ&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=zazabap/LogosQ&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=zazabap/LogosQ&type=Date" />
</picture>

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## FAQ

1. Some bugs might appear when using the crate, one of the error related with missing package in libfontconfig1-dev, in such case, running on debian/linux (tested on ubuntu) could fix the build error.
```
sudo apt-get update
sudo apt-get install -y libfontconfig1-dev pkg-config
```
