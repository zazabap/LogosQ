use logosq::gates::matrix::*;
use logosq::states::State;
use num_complex::Complex64;

#[test]
fn test_cnot_gate() {
    // Create CNOT gate
    let cnot = cnot_gate();

    // Test matrix representation
    assert_eq!(cnot.matrix.shape(), &[4, 4]);

    // Set up a 2-qubit circuit with CNOT
    let mut circuit = logosq::circuits::Circuit::new(2);
    circuit.add_matrix_gate(cnot.matrix.clone(), vec![0, 1], "CNOT").unwrap();

    // Test |00⟩ → |00⟩ (no flip when control is 0)
    let mut state = State::zero_state(2);
    circuit.execute(&mut state).unwrap();
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);

    // Test |10⟩ → |11⟩ (flip when control is 1)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.execute(&mut state).unwrap();

    circuit.execute(&mut state).unwrap();
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);

    // Test |+0⟩ → Bell state (1/√2)(|00⟩ + |11⟩)
    let mut state = State::zero_state(2);
    let h = h_gate();
    let mut h_circuit = logosq::circuits::Circuit::new(2);
    h_circuit.add_single_qubit_gate(h.matrix.clone(), 0, "H");
    h_circuit.execute(&mut state).unwrap();

    circuit.execute(&mut state).unwrap();
    assert!((state.probability(0) - 0.5).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 0.5).abs() < 1e-10);
}

#[test]
fn test_swap_gate() {
    // Create SWAP gate
    let swap = swap_gate();

    // Test matrix representation
    assert_eq!(swap.matrix.shape(), &[4, 4]);

    // Set up a 2-qubit circuit with SWAP
    let mut circuit = logosq::circuits::Circuit::new(2);
    circuit.add_matrix_gate(swap.matrix.clone(), vec![0, 1], "SWAP").unwrap();

    // Test |01⟩ → |10⟩ (swap states)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state).unwrap();

    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);

    circuit.execute(&mut state).unwrap();
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!((state.probability(2) - 1.0).abs() < 1e-10);
    assert!(state.probability(3) < 1e-10);
}

#[test]
fn test_cz_gate() {
    // Create CZ gate
    let cz = cz_gate();

    // Test matrix representation
    assert_eq!(cz.matrix.shape(), &[4, 4]);

    // Set up a 2-qubit circuit with CZ
    let mut circuit = logosq::circuits::Circuit::new(2);
    circuit.add_matrix_gate(cz.matrix.clone(), vec![0, 1], "CZ").unwrap();

    // Test |11⟩ → -|11⟩ (phase flip when both qubits are 1)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state).unwrap();

    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);

    circuit.execute(&mut state).unwrap();
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);
    assert!((state.vector()[3] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_toffoli_gate() {
    // Create Toffoli gate
    let toffoli = toffoli_gate();

    // Test matrix representation
    assert_eq!(toffoli.matrix.shape(), &[8, 8]);

    // Set up a 3-qubit circuit with Toffoli
    let mut circuit = logosq::circuits::Circuit::new(3);
    circuit.add_matrix_gate(toffoli.matrix.clone(), vec![0, 1, 2], "Toffoli").unwrap();

    // Test |110⟩ → |111⟩ (flip target when both controls are 1)
    let mut state = State::zero_state(3);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(3);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state).unwrap();

    // |110⟩ corresponds to binary 110 = decimal 6
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);
    assert!(state.probability(4) < 1e-10);
    assert!(state.probability(5) < 1e-10);
    assert!((state.probability(6) - 1.0).abs() < 1e-10);
    assert!(state.probability(7) < 1e-10);

    circuit.execute(&mut state).unwrap();

    // |111⟩ corresponds to binary 111 = decimal 7
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);
    assert!(state.probability(4) < 1e-10);
    assert!(state.probability(5) < 1e-10);
    assert!(state.probability(6) < 1e-10);
    assert!((state.probability(7) - 1.0).abs() < 1e-10);
}
