use num_complex::{Complex64};
use logosq::gates::*;
use logosq::states::State;
use std::f64::consts::{PI, SQRT_2};

#[test]
fn test_x_gate() {
    // Create X gate
    let x = x_gate();

    // Test matrix representation
    assert_eq!(x.matrix.shape(), &[2, 2]);
    assert!((x.matrix[[0, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((x.matrix[[0, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((x.matrix[[1, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((x.matrix[[1, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    
    // Test application to |0⟩ (should get |1⟩)
    let mut state = State::zero_state(1);
    x.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    
    // Test application to |1⟩ (should get |0⟩)
    let mut state = State::one_state(1);
    x.apply(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
}

#[test]
fn test_y_gate() {
    // Create Y gate
    let y = y_gate();
    
    // Test matrix representation
    assert_eq!(y.matrix.shape(), &[2, 2]);
    assert!((y.matrix[[0, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((y.matrix[[0, 1]] - Complex64::new(0.0, -1.0)).norm() < 1e-10);
    assert!((y.matrix[[1, 0]] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    assert!((y.matrix[[1, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    
    // Test application to |0⟩ (should get i|1⟩)
    let mut state = State::zero_state(1);
    y.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    
    // Test application to |1⟩ (should get -i|0⟩)
    let mut state = State::one_state(1);
    y.apply(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!((state.vector[0] - Complex64::new(0.0, -1.0)).norm() < 1e-10);
}

#[test]
fn test_z_gate() {
    // Create Z gate
    let z = z_gate();
    
    // Test matrix representation
    assert_eq!(z.matrix.shape(), &[2, 2]);
    assert!((z.matrix[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((z.matrix[[0, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((z.matrix[[1, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((z.matrix[[1, 1]] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    
    // Test application to |0⟩ (should remain |0⟩)
    let mut state = State::zero_state(1);
    z.apply(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    
    // Test application to |1⟩ (should get -|1⟩)
    let mut state = State::one_state(1);
    z.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    
    // Test application to |+⟩ (should get |−⟩)
    let mut state = State::plus_state(1);
    z.apply(&mut state);
    // |−⟩ has equal probability for |0⟩ and |1⟩
    assert!((state.probability(0) - 0.5).abs() < 1e-10);
    assert!((state.probability(1) - 0.5).abs() < 1e-10);
    // But opposite phases
    assert!((state.vector[0] - Complex64::new(1.0/SQRT_2, 0.0)).norm() < 1e-10);
    assert!((state.vector[1] - Complex64::new(-1.0/SQRT_2, 0.0)).norm() < 1e-10);
}

#[test]
fn test_hadamard_gate() {
    // Create H gate
    let h = h_gate();
    
    // Test matrix representation
    assert_eq!(h.matrix.shape(), &[2, 2]);
    assert!((h.matrix[[0, 0]] - Complex64::new(1.0/SQRT_2, 0.0)).norm() < 1e-10);
    assert!((h.matrix[[0, 1]] - Complex64::new(1.0/SQRT_2, 0.0)).norm() < 1e-10);
    assert!((h.matrix[[1, 0]] - Complex64::new(1.0/SQRT_2, 0.0)).norm() < 1e-10);
    assert!((h.matrix[[1, 1]] - Complex64::new(-1.0/SQRT_2, 0.0)).norm() < 1e-10);
    
    // Test application to |0⟩ (should get |+⟩)
    let mut state = State::zero_state(1);
    h.apply(&mut state);
    assert!((state.probability(0) - 0.5).abs() < 1e-10);
    assert!((state.probability(1) - 0.5).abs() < 1e-10);
    
    // Test application to |1⟩ (should get |−⟩)
    let mut state = State::one_state(1);
    h.apply(&mut state);
    assert!((state.probability(0) - 0.5).abs() < 1e-10);
    assert!((state.probability(1) - 0.5).abs() < 1e-10);
    
    // Test H·H = I (applying H twice should give back the original state)
    let mut state = State::zero_state(1);
    h.apply(&mut state);
    h.apply(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
}

#[test]
fn test_s_gate() {
    // Create S gate
    let s = s_gate();
    
    // Test matrix representation
    assert_eq!(s.matrix.shape(), &[2, 2]);
    assert!((s.matrix[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((s.matrix[[0, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((s.matrix[[1, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((s.matrix[[1, 1]] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    
    // Test application to |0⟩ (should remain |0⟩)
    let mut state = State::zero_state(1);
    s.apply(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    
    // Test application to |1⟩ (should get i|1⟩)
    let mut state = State::one_state(1);
    s.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    
    // Test S·S = Z
    let mut state = State::one_state(1);
    s.apply(&mut state);
    s.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_t_gate() {
    // Create T gate
    let t = t_gate();
    
    // Test matrix representation
    assert_eq!(t.matrix.shape(), &[2, 2]);
    assert!((t.matrix[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    
    // Test application to |1⟩ (phase should change)
    let mut state = State::one_state(1);
    t.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    
    // Test T·T·T·T = Z
    let mut state = State::one_state(1);
    t.apply(&mut state);
    t.apply(&mut state);
    t.apply(&mut state);
    t.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_rotation_gates() {
    // Test RX gate
    let rx_pi = rx_gate(PI);

    // RX(π) should be equivalent to X gate
    let mut state = State::zero_state(1);
    rx_pi.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    
    // Test RY gate
    let ry_pi = ry_gate(PI);
    
    // RY(π) should flip bits with a phase
    let mut state = State::zero_state(1);
    ry_pi.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    
    // Test RZ gate
    let rz_2pi = rz_gate(2.0 * PI);
    
    // RZ(2π) should be equivalent to Z gate
    let mut state = State::one_state(1);
    rz_2pi.apply(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!((state.vector[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_cnot_gate() {
    // Create CNOT gate
    let cnot = cnot_gate();
    
    // Test matrix representation
    assert_eq!(cnot.matrix.shape(), &[4, 4]);
    
    // Set up a 2-qubit circuit with CNOT
    let mut circuit = logosq::circuits::Circuit::new(2);
    circuit.add_matrix_gate(cnot.matrix.clone(), vec![0, 1], "CNOT");
    
    // Test |00⟩ → |00⟩ (no flip when control is 0)
    let mut state = State::zero_state(2);
    circuit.execute(&mut state);
    assert!((state.probability(0) - 1.0).abs() < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);
    
    // Test |10⟩ → |11⟩ (flip when control is 1)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.execute(&mut state);
    
    circuit.execute(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);
    
    // Test |+0⟩ → Bell state (1/√2)(|00⟩ + |11⟩)
    let mut state = State::zero_state(2);
    let h = h_gate();
    let mut h_circuit = logosq::circuits::Circuit::new(2);
    h_circuit.add_single_qubit_gate(h.matrix.clone(), 0, "H");
    h_circuit.execute(&mut state);
    
    circuit.execute(&mut state);
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
    circuit.add_matrix_gate(swap.matrix.clone(), vec![0, 1], "SWAP");
    
    // Test |01⟩ → |10⟩ (swap states)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state);
    
    assert!(state.probability(0) < 1e-10);
    assert!((state.probability(1) - 1.0).abs() < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);
    
    circuit.execute(&mut state);
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
    circuit.add_matrix_gate(cz.matrix.clone(), vec![0, 1], "CZ");
    
    // Test |11⟩ → -|11⟩ (phase flip when both qubits are 1)
    let mut state = State::zero_state(2);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(2);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state);
    
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);
    
    circuit.execute(&mut state);
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!((state.probability(3) - 1.0).abs() < 1e-10);
    assert!((state.vector[3] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_toffoli_gate() {
    // Create Toffoli gate
    let toffoli = toffoli_gate();
    
    // Test matrix representation
    assert_eq!(toffoli.matrix.shape(), &[8, 8]);
    
    // Set up a 3-qubit circuit with Toffoli
    let mut circuit = logosq::circuits::Circuit::new(3);
    circuit.add_matrix_gate(toffoli.matrix.clone(), vec![0, 1, 2], "Toffoli");
    
    // Test |110⟩ → |111⟩ (flip target when both controls are 1)
    let mut state = State::zero_state(3);
    let x = x_gate();
    let mut x_circuit = logosq::circuits::Circuit::new(3);
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 0, "X");
    x_circuit.add_single_qubit_gate(x.matrix.clone(), 1, "X");
    x_circuit.execute(&mut state);
    
    // |110⟩ corresponds to binary 110 = decimal 6
    assert!(state.probability(0) < 1e-10);
    assert!(state.probability(1) < 1e-10);
    assert!(state.probability(2) < 1e-10);
    assert!(state.probability(3) < 1e-10);
    assert!(state.probability(4) < 1e-10);
    assert!(state.probability(5) < 1e-10);
    assert!((state.probability(6) - 1.0).abs() < 1e-10);
    assert!(state.probability(7) < 1e-10);
    
    circuit.execute(&mut state);
    
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