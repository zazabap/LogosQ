//! Observable measurement for quantum states
//!
//! This module provides tools for computing expectation values of observables.

use crate::states::State;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait for quantum observables
pub trait Observable {
    /// Compute expectation value ⟨ψ|O|ψ⟩
    fn expectation(&self, state: &State) -> f64;

    /// Get the matrix representation (if available)
    fn matrix(&self) -> Option<Array2<Complex64>> {
        None
    }
}

/// Pauli operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// Get 2×2 matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Pauli::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
        }
    }
}

/// A single Pauli term like 0.5 * X₀Y₁Z₂
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Coefficient (weight)
    pub coefficient: f64,
    /// Pauli operators for each qubit
    pub paulis: Vec<Pauli>,
}

impl PauliTerm {
    pub fn new(coefficient: f64, paulis: Vec<Pauli>) -> Self {
        Self {
            coefficient,
            paulis,
        }
    }

    /// Compute expectation value for this term
    pub fn expectation(&self, state: &State) -> f64 {
        let n = state.num_qubits;
        assert_eq!(
            self.paulis.len(),
            n,
            "Pauli term must have operator for each qubit"
        );

        // For efficiency, we compute expectation directly
        let vector = state.vector.as_slice().unwrap();
        let dim = 1 << n;

        let mut result = Complex64::new(0.0, 0.0);

        for i in 0..dim {
            let mut amplitude = Complex64::new(1.0, 0.0);
            let mut j = i;

            // Apply Pauli operators
            for (qubit, pauli) in self.paulis.iter().enumerate() {
                let bit = (i >> (n - 1 - qubit)) & 1;

                match pauli {
                    Pauli::I => {
                        // Identity: no change
                    }
                    Pauli::X => {
                        // Flip bit
                        j ^= 1 << (n - 1 - qubit);
                    }
                    Pauli::Y => {
                        // Flip bit and add phase
                        j ^= 1 << (n - 1 - qubit);
                        amplitude *= if bit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    Pauli::Z => {
                        // Add phase based on bit
                        if bit == 1 {
                            amplitude *= Complex64::new(-1.0, 0.0);
                        }
                    }
                }
            }

            result += vector[i].conj() * amplitude * vector[j];
        }

        self.coefficient * result.re
    }
}

/// Observable as a sum of Pauli terms
#[derive(Debug, Clone)]
pub struct PauliObservable {
    pub terms: Vec<PauliTerm>,
    pub num_qubits: usize,
}

impl PauliObservable {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            terms: Vec::new(),
            num_qubits,
        }
    }

    /// Add a Pauli term
    pub fn add_term(&mut self, term: PauliTerm) {
        assert_eq!(
            term.paulis.len(),
            self.num_qubits,
            "Term must match observable qubit count"
        );
        self.terms.push(term);
    }

    /// Create a simple Z observable on a single qubit
    pub fn single_z(num_qubits: usize, target_qubit: usize) -> Self {
        assert!(
            target_qubit < num_qubits,
            "Qubit index out of bounds"
        );
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[target_qubit] = Pauli::Z;

        let mut obs = Self::new(num_qubits);
        obs.add_term(PauliTerm::new(1.0, paulis));
        obs
    }

    /// Create ZZ interaction term
    pub fn zz_interaction(
        num_qubits: usize,
        qubit1: usize,
        qubit2: usize,
        coefficient: f64,
    ) -> Self {
        assert!(qubit1 < num_qubits, "Qubit index out of bounds");
        assert!(qubit2 < num_qubits, "Qubit index out of bounds");
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[qubit1] = Pauli::Z;
        paulis[qubit2] = Pauli::Z;

        let mut obs = Self::new(num_qubits);
        obs.add_term(PauliTerm::new(coefficient, paulis));
        obs
    }

    /// Create Ising Hamiltonian: H = Σᵢ Jᵢ ZᵢZᵢ₊₁ + Σᵢ hᵢ Zᵢ
    pub fn ising_hamiltonian(num_qubits: usize, j_couplings: &[f64], h_fields: &[f64]) -> Self {
        let mut obs = Self::new(num_qubits);

        // ZZ interactions
        for i in 0..j_couplings.len() {
            assert!(i + 1 < num_qubits, "Qubit index out of bounds");
            let mut paulis = vec![Pauli::I; num_qubits];
            paulis[i] = Pauli::Z;
            paulis[i + 1] = Pauli::Z;
            obs.add_term(PauliTerm::new(j_couplings[i], paulis));
        }

        // Z fields
        for i in 0..h_fields.len() {
            let mut paulis = vec![Pauli::I; num_qubits];
            paulis[i] = Pauli::Z;
            obs.add_term(PauliTerm::new(h_fields[i], paulis));
        }

        obs
    }
}

impl Observable for PauliObservable {
    fn expectation(&self, state: &State) -> f64 {
        self.terms.iter().map(|term| term.expectation(state)).sum()
    }
}
