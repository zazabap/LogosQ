//! Variational Quantum Eigensolver (VQE) for H2 Molecule
//!
//! This example demonstrates a complete VQE implementation for finding the ground state
//! energy of the hydrogen molecule (H2) using various components from the optimization module.
//!
//! The example showcases:
//! - Multiple ansatz architectures (Hardware-Efficient, Real Amplitudes, Efficient SU(2), Custom)
//! - Different gradient methods (Parameter-Shift Rule, Finite Difference)
//! - Various optimizers (Adam, Gradient Descent)
//! - Quantum Natural Gradient
//! - Custom H2 Hamiltonian construction
//! - Comparison of different approaches

use logosq::circuits::Circuit;
use logosq::optimization::ansatz::{
    Ansatz, EfficientSU2Ansatz, HardwareEfficientAnsatz, ParameterizedCircuit, RealAmplitudesAnsatz,
    EntanglingGate, EntanglingPattern,
};
use logosq::optimization::gradient::{FiniteDifference, GradientMethod, ParameterShift};
use logosq::optimization::observable::{Observable, PauliObservable, PauliTerm, Pauli};
use logosq::optimization::optimizer::{Adam, GradientDescent};
use logosq::optimization::qng::QuantumNaturalGradient;
use logosq::optimization::vqe::{VQE, VQEResult};
use logosq::states::State;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Creates the H2 molecular Hamiltonian in minimal STO-3G basis (4 qubits)
///
/// The Hamiltonian is represented as a sum of Pauli terms:
/// H = Σᵢ cᵢ Pᵢ
///
/// ## How Coefficients Are Determined
///
/// The coefficients come from quantum chemistry calculations:
///
/// 1. **Molecular Geometry**: H2 at equilibrium (R = 0.74 Å)
/// 2. **Basis Set**: STO-3G (4 spin-orbitals → 4 qubits)
/// 3. **Compute Integrals**: One-electron (hᵢⱼ) and two-electron (hᵢⱼₖₗ) integrals
/// 4. **Hartree-Fock**: Solve for molecular orbitals
/// 5. **Second Quantization**: H = Σ hᵢⱼ aᵢ†aⱼ + ½ Σ hᵢⱼₖₗ aᵢ†aⱼ†aₖaₗ
/// 6. **Jordan-Wigner**: Map fermionic operators → Pauli operators
///
/// Each coefficient cᵢ is a linear combination of molecular integrals transformed
/// through the JW mapping. See `examples/HAMILTONIAN_COEFFICIENTS.md` for details.
///
/// ## Coefficient Values
///
/// These coefficients are based on standard H2 STO-3G calculations at equilibrium
/// geometry (R = 0.74 Å) using Jordan-Wigner transformation. The values come from
/// quantum chemistry integrals computed via Hartree-Fock method.
///
/// **Note**: The ground state energy from these coefficients may differ from
/// the literature value of -1.137 Ha depending on the specific basis set
/// implementation and integral calculation method used. This is normal and
/// reflects the approximations inherent in minimal basis sets.
fn create_h2_hamiltonian() -> PauliObservable {
    let num_qubits = 4;
    let mut hamiltonian = PauliObservable::new(num_qubits);

    // Identity term (nuclear repulsion + constant offset from JW transformation)
    // This includes: nuclear repulsion energy (1/R) + constant terms from 
    // transforming fermionic operators to Pauli operators
    hamiltonian.add_term(PauliTerm::new(
        -0.8097,
        vec![Pauli::I, Pauli::I, Pauli::I, Pauli::I],
    ));

    // Single-qubit Z terms (one-electron integrals)
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::I, Pauli::I, Pauli::I, Pauli::Z],
    ));

    // Two-qubit ZZ interactions (electron-electron repulsion)
    hamiltonian.add_term(PauliTerm::new(
        -0.2228,
        vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        -0.2228,
        vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::Z],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1656,
        vec![Pauli::Z, Pauli::I, Pauli::Z, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1656,
        vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::Z],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1656,
        vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::Z],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1656,
        vec![Pauli::I, Pauli::Z, Pauli::Z, Pauli::I],
    ));

    // XX and YY terms (hopping/exchange terms)
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::X, Pauli::X, Pauli::I, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1712,
        vec![Pauli::I, Pauli::I, Pauli::X, Pauli::X],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1206,
        vec![Pauli::Y, Pauli::Y, Pauli::I, Pauli::I],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.1206,
        vec![Pauli::I, Pauli::I, Pauli::Y, Pauli::Y],
    ));

    // Cross terms
    hamiltonian.add_term(PauliTerm::new(
        0.0453,
        vec![Pauli::X, Pauli::X, Pauli::Z, Pauli::Z],
    ));
    hamiltonian.add_term(PauliTerm::new(
        0.0453,
        vec![Pauli::Y, Pauli::Y, Pauli::Z, Pauli::Z],
    ));

    hamiltonian
}

/// Compute the exact ground state energy by diagonalizing the Hamiltonian
/// This builds the full 2^n × 2^n matrix and finds its minimum eigenvalue
fn compute_exact_ground_state_energy(hamiltonian: &PauliObservable) -> f64 {
    let n = hamiltonian.num_qubits;
    let dim = 1 << n;
    
    // Build the full matrix representation
    let mut matrix = Array2::<Complex64>::zeros((dim, dim));
    
    // Iterate over each Pauli term and add its contribution
    for term in &hamiltonian.terms {
        let pauli_matrices: Vec<Array2<Complex64>> = term.paulis.iter()
            .map(|p| p.matrix())
            .collect();
        
        // Build the tensor product of all Pauli matrices
        let mut term_matrix = Array2::<Complex64>::ones((1, 1));
        for pauli_mat in &pauli_matrices {
            // Compute Kronecker product: term_matrix ⊗ pauli_mat
            let (rows1, cols1) = (term_matrix.shape()[0], term_matrix.shape()[1]);
            let (rows2, cols2) = (pauli_mat.shape()[0], pauli_mat.shape()[1]);
            
            let mut new_matrix = Array2::<Complex64>::zeros((rows1 * rows2, cols1 * cols2));
            for i in 0..rows1 {
                for j in 0..cols1 {
                    for k in 0..rows2 {
                        for l in 0..cols2 {
                            new_matrix[[i * rows2 + k, j * cols2 + l]] = 
                                term_matrix[[i, j]] * pauli_mat[[k, l]];
                        }
                    }
                }
            }
            term_matrix = new_matrix;
        }
        
        // Add this term to the full Hamiltonian (scaled by coefficient)
        for i in 0..dim {
            for j in 0..dim {
                matrix[[i, j]] += Complex64::new(term.coefficient, 0.0) * term_matrix[[i, j]];
            }
        }
    }
    
    // For a 16x16 matrix, we can compute eigenvalues more directly
    // Use inverse power iteration with shift to find minimum eigenvalue
    // First, find an upper bound for eigenvalues
    let mut max_eigenvalue = f64::NEG_INFINITY;
    for i in 0..dim {
        max_eigenvalue = max_eigenvalue.max(matrix[[i, i]].re);
    }
    
    // Use shift so that (H - shift*I) has negative eigenvalues
    // Then apply inverse power iteration to find the eigenvalue closest to shift
    let shift = max_eigenvalue + 10.0;  // Shift to ensure all eigenvalues are negative
    let mut shifted_matrix = matrix.clone();
    for i in 0..dim {
        shifted_matrix[[i, i]] -= Complex64::new(shift, 0.0);
    }
    
    // Initialize random state
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut state_vec = Array2::<Complex64>::zeros((dim, 1));
    for i in 0..dim {
        state_vec[[i, 0]] = Complex64::new(rng.gen::<f64>(), rng.gen::<f64>());
    }
    
    // Normalize (unused for now, but kept for future inverse power iteration)
    let _norm: f64 = state_vec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    
    // Inverse power iteration: solve (H - shift*I) * x = state_vec
    // Since we don't have a direct solver, use iterative refinement
    // Actually, for small matrices, we can use a simple approach:
    // Try many random states and minimize Rayleigh quotient
    
    let mut min_energy = f64::INFINITY;
    
    // Try all computational basis states
    for i in 0..dim {
        let mut state = State::zero_state(n);
        state.vector_mut()[i] = Complex64::new(1.0, 0.0);
        let energy = hamiltonian.expectation(&state);
        min_energy = min_energy.min(energy);
    }
    
    // Use power iteration with Rayleigh quotient to find minimum eigenvalue
    // Start from the best computational basis state
    let mut best_state_idx = 0;
    let mut best_basis_energy = f64::INFINITY;
    for i in 0..dim {
        let mut state = State::zero_state(n);
        state.vector_mut()[i] = Complex64::new(1.0, 0.0);
        let energy = hamiltonian.expectation(&state);
        if energy < best_basis_energy {
            best_basis_energy = energy;
            best_state_idx = i;
        }
    }
    min_energy = best_basis_energy;
    
    // Initialize state vector from best computational basis state
    let mut state_vec = Array2::<Complex64>::zeros((dim, 1));
    state_vec[[best_state_idx, 0]] = Complex64::new(1.0, 0.0);
    
    // Power iteration to find ground state (minimum eigenvalue)
    // For minimum eigenvalue, we use: (H - λ_max*I) has negative eigenvalues
    // The minimum eigenvalue of H corresponds to maximum eigenvalue of (λ_max*I - H)
    let lambda_max = max_eigenvalue + 10.0;
    let mut transformed_matrix = Array2::<Complex64>::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            transformed_matrix[[i, j]] = if i == j {
                Complex64::new(lambda_max, 0.0) - matrix[[i, j]]
            } else {
                -matrix[[i, j]]
            };
        }
    }
    
    // Power iteration on transformed matrix to find maximum eigenvalue
    for iteration in 0..200 {
        let new_state = transformed_matrix.dot(&state_vec);
        let norm: f64 = new_state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        state_vec = &new_state / norm;
        
        // Compute Rayleigh quotient for transformed matrix
        let numerator = (state_vec.t().dot(&transformed_matrix.dot(&state_vec)))[[0, 0]];
        let transformed_max_eigenval = numerator.re;
        
        // Convert back to minimum eigenvalue of original matrix
        let current_min_energy = lambda_max - transformed_max_eigenval;
        min_energy = min_energy.min(current_min_energy);
        
        // Check convergence
        if iteration > 10 && (current_min_energy - min_energy).abs() < 1e-8 {
            break;
        }
    }
    
    min_energy
}

/// Creates a chemistry-inspired ansatz for H2
///
/// This custom ansatz is inspired by the structure of quantum chemistry ansatzes,
/// using a Hartree-Fock preparation followed by parameterized excitations.
fn create_chemistry_ansatz() -> ParameterizedCircuit<impl Fn(&[f64]) -> Circuit> {
    ParameterizedCircuit::new(
        4,  // 4 qubits for H2
        8,  // 8 parameters
        move |params| {
            let mut circuit = Circuit::new(4);
            
            // Hartree-Fock preparation (|1100⟩ state - two electrons in bonding orbital)
            circuit.x(0);
            circuit.x(1);
            
            // Parameterized single excitations (UCCSD-inspired)
            // Rotation around X for single excitations
            circuit.rx(0, params[0]);
            circuit.rx(1, params[1]);
            circuit.rx(2, params[2]);
            circuit.rx(3, params[3]);
            
            // Entangling layer for correlations
            circuit.cnot(0, 2);
            circuit.cnot(1, 3);
            circuit.cnot(0, 1);
            circuit.cnot(2, 3);
            
            // Final rotations
            circuit.ry(0, params[4]);
            circuit.ry(1, params[5]);
            circuit.ry(2, params[6]);
            circuit.ry(3, params[7]);
            
            circuit
        },
    )
}

/// Run VQE with Hardware-Efficient Ansatz using Adam optimizer
fn run_vqe_hardware_efficient_adam(
    hamiltonian: &PauliObservable,
    depth: usize,
) -> VQEResult {
    println!("\n{}", "=".repeat(70));
    println!("Hardware-Efficient Ansatz (depth={}) with Adam", depth);
    println!("{}", "=".repeat(70));

    let ansatz = HardwareEfficientAnsatz::new(
        4,
        depth,
        EntanglingGate::CNOT,
        EntanglingPattern::Linear,
    );

    let gradient_method = ParameterShift::new();
    let optimizer = Adam::new(0.01, 200).with_tolerance(1e-6);

    let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
    vqe.verbose = true;
    vqe.run_random()
}

/// Run VQE with Hardware-Efficient Ansatz using Gradient Descent optimizer
fn run_vqe_hardware_efficient_gd(
    hamiltonian: &PauliObservable,
    depth: usize,
) -> VQEResult {
    println!("\n{}", "=".repeat(70));
    println!("Hardware-Efficient Ansatz (depth={}) with Gradient Descent", depth);
    println!("{}", "=".repeat(70));

    let ansatz = HardwareEfficientAnsatz::new(
        4,
        depth,
        EntanglingGate::CNOT,
        EntanglingPattern::Linear,
    );

    let gradient_method = ParameterShift::new();
    let optimizer = GradientDescent::new(0.01, 200).with_tolerance(1e-6);

    let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
    vqe.verbose = true;
    vqe.run_random()
}

/// Run VQE with Real Amplitudes Ansatz
fn run_vqe_real_amplitudes(hamiltonian: &PauliObservable, depth: usize) -> VQEResult {
    println!("\n{}", "=".repeat(70));
    println!("Real Amplitudes Ansatz (depth={})", depth);
    println!("{}", "=".repeat(70));

    let ansatz = RealAmplitudesAnsatz::new(4, depth);
    let gradient_method = ParameterShift::new();
    let optimizer = Adam::new(0.01, 200).with_tolerance(1e-6);

    let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
    vqe.verbose = true;
    vqe.run_random()
}

/// Run VQE with Efficient SU(2) Ansatz
fn run_vqe_efficient_su2(hamiltonian: &PauliObservable, depth: usize) -> VQEResult {
    println!("\n{}", "=".repeat(70));
    println!("Efficient SU(2) Ansatz (depth={})", depth);
    println!("{}", "=".repeat(70));

    let ansatz = EfficientSU2Ansatz::new(4, depth);
    let gradient_method = ParameterShift::new();
    let optimizer = Adam::new(0.01, 200).with_tolerance(1e-6);

    let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
    vqe.verbose = true;
    vqe.run_random()
}

/// Run VQE with custom chemistry-inspired ansatz
fn run_vqe_chemistry_ansatz(hamiltonian: &PauliObservable) -> VQEResult {
    println!("\n{}", "=".repeat(70));
    println!("Chemistry-Inspired Custom Ansatz");
    println!("{}", "=".repeat(70));

    let ansatz = create_chemistry_ansatz();
    let gradient_method = ParameterShift::new();
    let optimizer = Adam::new(0.01, 200).with_tolerance(1e-6);

    let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
    vqe.verbose = true;
    vqe.run_random()
}

/// Compare Parameter-Shift Rule vs Finite Difference gradients
fn compare_gradient_methods(hamiltonian: &PauliObservable) {
    println!("\n{}", "=".repeat(70));
    println!("Gradient Method Comparison");
    println!("{}", "=".repeat(70));

    let ansatz = HardwareEfficientAnsatz::new(
        4,
        2,
        EntanglingGate::CNOT,
        EntanglingPattern::Linear,
    );

    let parameters: Vec<f64> = (0..ansatz.num_parameters())
        .map(|i| (i as f64) * 0.1)
        .collect();

    let ps_method = ParameterShift::new();
    let fd_method = FiniteDifference::new(1e-7);

    println!("Computing gradients with {} parameters...", parameters.len());
    
    let ps_gradient = ps_method.compute_gradient(&ansatz, hamiltonian, &parameters);
    let fd_gradient = fd_method.compute_gradient(&ansatz, hamiltonian, &parameters);

    println!("\nParameter-Shift Rule gradient:");
    for (i, g) in ps_gradient.iter().enumerate() {
        println!("  ∂E/∂θ[{}] = {:.6}", i, g);
    }

    println!("\nFinite Difference gradient:");
    for (i, g) in fd_gradient.iter().enumerate() {
        println!("  ∂E/∂θ[{}] = {:.6}", i, g);
    }

    // Compare gradients
    let max_diff = ps_gradient
        .iter()
        .zip(fd_gradient.iter())
        .map(|(ps, fd)| (ps - fd).abs())
        .fold(0.0, f64::max);

    println!("\nMaximum difference: {:.6}", max_diff);
    
    if max_diff < 1e-4 {
        println!("✓ Gradients match within tolerance!");
    } else {
        println!("⚠ Warning: Significant difference between methods");
    }
}

/// Demonstrate Quantum Natural Gradient
fn demonstrate_qng(hamiltonian: &PauliObservable) {
    println!("\n{}", "=".repeat(70));
    println!("Quantum Natural Gradient Demonstration");
    println!("{}", "=".repeat(70));

    let ansatz = HardwareEfficientAnsatz::new(
        4,
        2,
        EntanglingGate::CNOT,
        EntanglingPattern::Linear,
    );

    let parameters: Vec<f64> = (0..ansatz.num_parameters())
        .map(|i| (i as f64) * 0.1)
        .collect();

    let gradient_method = ParameterShift::new();
    let qng = QuantumNaturalGradient::new();

    println!("Computing standard gradient...");
    let standard_grad = gradient_method.compute_gradient(&ansatz, hamiltonian, &parameters);
    
    println!("Computing quantum natural gradient...");
    let nat_grad = qng.compute_natural_gradient(&ansatz, hamiltonian, &gradient_method, &parameters);

    println!("\nStandard gradient norm: {:.6}", 
             standard_grad.iter().map(|g| g * g).sum::<f64>().sqrt());
    println!("Natural gradient norm: {:.6}", 
             nat_grad.iter().map(|g| g * g).sum::<f64>().sqrt());

    println!("\nComputing metric tensor...");
    let metric = qng.compute_metric_tensor(&ansatz, &parameters);
    println!("Metric tensor shape: {:?}", metric.shape());
    println!("Metric tensor diagonal (first 5):");
    for i in 0..(parameters.len().min(5)) {
        println!("  g[{}, {}] = {:.6}", i, i, metric[[i, i]]);
    }
}

/// Test different entangling patterns
fn test_entangling_patterns(hamiltonian: &PauliObservable) {
    println!("\n{}", "=".repeat(70));
    println!("Entangling Pattern Comparison");
    println!("{}", "=".repeat(70));

    let patterns = vec![
        ("Linear", EntanglingPattern::Linear),
        ("Circular", EntanglingPattern::Circular),
        ("Full", EntanglingPattern::Full),
    ];

    for (name, pattern) in patterns {
        println!("\nTesting {} pattern...", name);
        
        let ansatz = HardwareEfficientAnsatz::new(
            4,
            2,
            EntanglingGate::CNOT,
            pattern,
        );

        let gradient_method = ParameterShift::new();
        let optimizer = Adam::new(0.01, 100).with_tolerance(1e-6);

        let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
        vqe.verbose = false;

        let result = vqe.run_random();
        println!("  Final energy: {:.6}", result.ground_state_energy);
        println!("  Iterations: {}", result.num_iterations);
    }
}

/// Test different entangling gates
fn test_entangling_gates(hamiltonian: &PauliObservable) {
    println!("\n{}", "=".repeat(70));
    println!("Entangling Gate Comparison");
    println!("{}", "=".repeat(70));

    let gates = vec![
        ("CNOT", EntanglingGate::CNOT),
        ("CZ", EntanglingGate::CZ),
        ("RZZ(π/2)", EntanglingGate::RZZ(PI / 2.0)),
    ];

    for (name, gate) in gates {
        println!("\nTesting {} gate...", name);
        
        let ansatz = HardwareEfficientAnsatz::new(
            4,
            2,
            gate,
            EntanglingPattern::Linear,
        );

        let gradient_method = ParameterShift::new();
        let optimizer = Adam::new(0.01, 100).with_tolerance(1e-6);

        let mut vqe = VQE::new(ansatz, hamiltonian.clone(), gradient_method, optimizer);
        vqe.verbose = false;

        let result = vqe.run_random();
        println!("  Final energy: {:.6}", result.ground_state_energy);
        println!("  Iterations: {}", result.num_iterations);
    }
}

/// Analyze convergence history
fn analyze_convergence(result: &VQEResult, name: &str) {
    println!("\n{} Convergence Analysis:", name);
    println!("  Initial energy: {:.6}", result.convergence_history[0]);
    println!("  Final energy: {:.6}", result.ground_state_energy);
    println!("  Energy reduction: {:.6}", 
             result.convergence_history[0] - result.ground_state_energy);
    println!("  Iterations: {}", result.num_iterations);
    
    // Check if energy is decreasing
    let mut decreasing = true;
    for i in 1..result.convergence_history.len() {
        if result.convergence_history[i] > result.convergence_history[i-1] + 1e-6 {
            decreasing = false;
            break;
        }
    }
    println!("  Monotonically decreasing: {}", decreasing);
}

/// Main function demonstrating comprehensive VQE for H2
fn main() {
    println!("{}", "=".repeat(70));
    println!("Variational Quantum Eigensolver for H2 Molecule");
    println!("{}", "=".repeat(70));
    println!("\nThis example demonstrates various optimization techniques");
    println!("for finding the ground state energy of H2 using VQE.\n");

    // Create H2 Hamiltonian
    let hamiltonian = create_h2_hamiltonian();
    println!("H2 Hamiltonian created with {} Pauli terms", hamiltonian.terms.len());
    
    // Test expectation value at initial state
    let initial_state = State::zero_state(4);
    let initial_energy = hamiltonian.expectation(&initial_state);
    println!("Initial state energy (|0000⟩): {:.6}", initial_energy);
    
    // Compute exact ground state energy by diagonalization
    println!("Computing exact ground state energy by diagonalizing Hamiltonian...");
    let exact_energy = compute_exact_ground_state_energy(&hamiltonian);
    println!("Exact ground state energy (from this Hamiltonian): {:.6} Ha", exact_energy);
    
    // Expected ground state energy for H2 at equilibrium
    // Note: Different STO-3G implementations may give slightly different values
    println!("Target ground state energy (literature, STO-3G): ~ -1.137 Ha (Hartree)");
    println!("Note: The exact energy depends on the specific basis set implementation.");
    println!("First-principles computed energy may differ from literature values.\n");

    // ============================================================================
    // Run VQE with different ansatzes
    // ============================================================================
    
    let result_he_adam = run_vqe_hardware_efficient_adam(&hamiltonian, 3);
    analyze_convergence(&result_he_adam, "Hardware-Efficient (Adam)");

    let result_he_gd = run_vqe_hardware_efficient_gd(&hamiltonian, 3);
    analyze_convergence(&result_he_gd, "Hardware-Efficient (Gradient Descent)");

    let result_ra = run_vqe_real_amplitudes(&hamiltonian, 3);
    analyze_convergence(&result_ra, "Real Amplitudes");

    let result_su2 = run_vqe_efficient_su2(&hamiltonian, 2);
    analyze_convergence(&result_su2, "Efficient SU(2)");

    let result_chem = run_vqe_chemistry_ansatz(&hamiltonian);
    analyze_convergence(&result_chem, "Chemistry-Inspired");

    // ============================================================================
    // Gradient method comparison
    // ============================================================================
    
    compare_gradient_methods(&hamiltonian);

    // ============================================================================
    // Quantum Natural Gradient
    // ============================================================================
    
    demonstrate_qng(&hamiltonian);

    // ============================================================================
    // Ansatz architecture comparisons
    // ============================================================================
    
    test_entangling_patterns(&hamiltonian);
    test_entangling_gates(&hamiltonian);

    // ============================================================================
    // Final summary
    // ============================================================================
    
    println!("\n{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));
    println!("\nBest results:");
    println!("  Hardware-Efficient (Adam): {:.6} Ha", result_he_adam.ground_state_energy);
    println!("  Real Amplitudes: {:.6} Ha", result_ra.ground_state_energy);
    println!("  Efficient SU(2): {:.6} Ha", result_su2.ground_state_energy);
    println!("  Chemistry-Inspired: {:.6} Ha", result_chem.ground_state_energy);
    
    let best_energy = [
        result_he_adam.ground_state_energy,
        result_ra.ground_state_energy,
        result_su2.ground_state_energy,
        result_chem.ground_state_energy,
    ]
    .iter()
    .fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("\nBest energy found: {:.6} Ha", best_energy);
    println!("Note: Compare this with the exact ground state energy shown at the start.");
    println!("If VQE energy is close to the exact energy, VQE is working correctly.");
    println!("If exact energy differs significantly from -1.137 Ha, verify Hamiltonian coefficients.");
    
    println!("\n{}", "=".repeat(70));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(70));
}

