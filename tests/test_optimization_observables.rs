use logosq::circuits::Circuit;
use logosq::optimization::observable::*;
use logosq::states::State;

mod tests {
    use super::*;
    // ============================================================================
    // OBSERVABLE TESTS
    // ============================================================================

    #[test]
    fn test_pauli_matrices() {
        let x = Pauli::X.matrix();
        let y = Pauli::Y.matrix();
        let z = Pauli::Z.matrix();
        let i = Pauli::I.matrix();

        // Check dimensions
        assert_eq!(x.shape(), &[2, 2]);
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(z.shape(), &[2, 2]);
        assert_eq!(i.shape(), &[2, 2]);

        // Check specific values
        use num_complex::Complex64;
        assert_eq!(i[[0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(z[[0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(z[[1, 1]], Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn test_pauli_term_z_expectation() {
        // Test Z expectation on |0⟩ state (should be +1)
        let paulis = vec![Pauli::Z];
        let term = PauliTerm::new(1.0, paulis);

        let state = State::zero_state(1);
        let expectation = term.expectation(&state);

        assert!((expectation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_term_z_on_one_state() {
        // Test Z expectation on |1⟩ state (should be -1)
        let paulis = vec![Pauli::Z];
        let term = PauliTerm::new(1.0, paulis);

        let mut circuit = Circuit::new(1);
        circuit.x(0); // Flip to |1⟩

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let expectation = term.expectation(&state);
        assert!((expectation - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_term_x_expectation() {
        // Test X expectation on |+⟩ state (should be +1)
        let paulis = vec![Pauli::X];
        let term = PauliTerm::new(1.0, paulis);

        let mut circuit = Circuit::new(1);
        circuit.h(0); // Create |+⟩

        let mut state = State::zero_state(1);
        circuit.execute(&mut state).unwrap();

        let expectation = term.expectation(&state);
        assert!((expectation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_observable_single_z() {
        let obs = PauliObservable::single_z(2, 0);

        let state = State::zero_state(2);
        let expectation = obs.expectation(&state);

        assert!((expectation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_observable_zz_interaction() {
        let obs = PauliObservable::zz_interaction(2, 0, 1, 1.0);

        // |00⟩: Z₀Z₁ = (+1)(+1) = +1
        let state = State::zero_state(2);
        let expectation = obs.expectation(&state);
        assert!((expectation - 1.0).abs() < 1e-10);

        // |11⟩: Z₀Z₁ = (-1)(-1) = +1
        let mut circuit = Circuit::new(2);
        circuit.x(0);
        circuit.x(1);
        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();
        let expectation = obs.expectation(&state);
        assert!((expectation - 1.0).abs() < 1e-10);

        // |01⟩: Z₀Z₁ = (+1)(-1) = -1
        let mut circuit = Circuit::new(2);
        circuit.x(1);
        let mut state = State::zero_state(2);
        circuit.execute(&mut state).unwrap();
        let expectation = obs.expectation(&state);
        assert!((expectation - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ising_hamiltonian() {
        let j_couplings = vec![1.0]; // One ZZ interaction
        let h_fields = vec![0.5, 0.5]; // Two Z fields

        let hamiltonian = PauliObservable::ising_hamiltonian(2, &j_couplings, &h_fields);

        // H = Z₀Z₁ + 0.5Z₀ + 0.5Z₁
        // On |00⟩: (+1)(+1) + 0.5(+1) + 0.5(+1) = 1 + 0.5 + 0.5 = 2
        let state = State::zero_state(2);
        let energy = hamiltonian.expectation(&state);
        assert!((energy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_observable_multiple_terms() {
        let mut obs = PauliObservable::new(2);

        // Add Z₀ term
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::Z, Pauli::I]));
        // Add Z₁ term
        obs.add_term(PauliTerm::new(1.0, vec![Pauli::I, Pauli::Z]));

        // On |00⟩: Z₀ + Z₁ = 1 + 1 = 2
        let state = State::zero_state(2);
        let expectation = obs.expectation(&state);
        assert!((expectation - 2.0).abs() < 1e-10);
    }
}
