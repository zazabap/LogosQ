use logosq::algorithms::xyz_heisenberg::{
    calculate_energy_efficient, create_circuit, HeisenbergParameters,
};
use logosq::prelude::*;

#[test]
fn test_mps_initial_state_expectation_matches_field_energy() {
    let params = HeisenbergParameters {
        jx: 0.0,
        jy: 0.0,
        jz: 0.0,
        external_field: 1.0,
        time_steps: 0,
        dt: 0.1,
    };

    let config = MpsConfig {
        max_bond_dim: 2,
        truncation_threshold: 1e-12,
    };

    let state = MpsState::one_state(4, config);
    let energy = calculate_energy_mps(&state, &params);

    // |1111⟩ has Z expectation -1 per qubit, so energy = -4 * field strength.
    assert!((energy + 4.0).abs() < 1e-10, "Expected -4.0, got {energy}");
}

#[test]
fn test_mps_state_vector_representation_matches_basis_state() {
    let config = MpsConfig::default();
    let state = MpsState::one_state(3, config);
    let vector = state.to_state_vector();

    assert_eq!(vector.len(), 8);
    for idx in 0..7 {
        assert!(
            vector[idx].norm() < 1e-12,
            "Unexpected amplitude at basis {idx}: {}",
            vector[idx]
        );
    }
    assert!(
        (vector[7] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
        "Last basis state should have amplitude 1, got {}",
        vector[7]
    );
}

#[test]
fn test_mps_single_qubit_gate_matches_dense_hadamard() {
    // Dense reference
    let mut dense_state = State::one_state(1);
    let mut dense_circuit = Circuit::new(1);
    dense_circuit.h(0);
    dense_circuit
        .execute(&mut dense_state)
        .expect("Dense Hadamard failed");
    let dense_vector = dense_state.vector().to_vec();

    // MPS application
    let config = MpsConfig::default();
    let mut mps_state = MpsState::one_state(1, config);
    let gate = h_gate();
    mps_state.apply_single_qubit(0, &gate.matrix);
    let mps_vector = mps_state.to_state_vector();

    for (dense_amp, mps_amp) in dense_vector.iter().zip(&mps_vector) {
        assert!(
            (*dense_amp - *mps_amp).norm() < 1e-12,
            "Single-qubit gate mismatch: dense={dense_amp}, mps={mps_amp}"
        );
    }
}

#[test]
fn test_mps_two_qubit_gate_matches_dense_cz() {
    // Dense reference with CZ
    let mut dense_state = State::one_state(2);
    let mut dense_circuit = Circuit::new(2);
    dense_circuit.cz(0, 1);
    dense_circuit
        .execute(&mut dense_state)
        .expect("Dense CZ failed");
    let dense_vector = dense_state.vector().to_vec();

    // Apply same CZ matrix through the MPS backend
    let mut mps_state = MpsState::one_state(
        2,
        MpsConfig {
            max_bond_dim: 4,
            truncation_threshold: 0.0,
        },
    );
    let cz_matrix = cz_gate();
    mps_state.apply_two_qubit(0, &cz_matrix.matrix);
    let mps_vector = mps_state.to_state_vector();

    for (idx, (dense_amp, mps_amp)) in dense_vector.iter().zip(&mps_vector).enumerate() {
        assert!(
            (*dense_amp - *mps_amp).norm() < 1e-12,
            "CZ mismatch at basis {idx}: dense={dense_amp}, mps={mps_amp}"
        );
    }
}

#[test]
fn test_mps_two_qubit_gate_on_entangled_state_matches_dense_sequence() {
    // Dense reference that generates an entangled intermediate state before a second two-qubit gate
    let mut dense_state = State::zero_state(2);
    let mut circuit = Circuit::new(2);
    circuit.h(0);
    circuit.rx(1, 0.37);
    circuit
        .add_operation(cnot_gate(), vec![0, 1], "CNOT")
        .expect("Failed to add CNOT");
    circuit.ry(0, -0.81);
    circuit
        .add_operation(cz_gate(), vec![0, 1], "CZ")
        .expect("Failed to add CZ");
    circuit
        .execute(&mut dense_state)
        .expect("Dense entangled sequence failed");
    let dense_vector = dense_state.vector().to_vec();

    // Apply the same sequence via the MPS backend
    let mut mps_state = MpsState::zero_state(
        2,
        MpsConfig {
            max_bond_dim: 8,
            truncation_threshold: 1e-12,
        },
    );
    let h = h_gate();
    mps_state.apply_single_qubit(0, &h.matrix);
    let rx = rx_gate(0.37);
    mps_state.apply_single_qubit(1, &rx.matrix);
    let cnot = cnot_gate();
    mps_state.apply_two_qubit(0, &cnot.matrix);
    let ry = ry_gate(-0.81);
    mps_state.apply_single_qubit(0, &ry.matrix);
    let cz = cz_gate();
    mps_state.apply_two_qubit(0, &cz.matrix);

    let mps_vector = mps_state.to_state_vector();
    for (idx, (dense_amp, mps_amp)) in dense_vector.iter().zip(&mps_vector).enumerate() {
        assert!(
            (*dense_amp - *mps_amp).norm() < 1e-10,
            "Entangled two-qubit sequence mismatch at basis {idx}: dense={dense_amp}, mps={mps_amp}"
        );
    }
}

#[test]
fn test_mps_field_only_evolution_matches_dense_result() {
    let params = HeisenbergParameters {
        jx: 0.0,
        jy: 0.0,
        jz: 0.0,
        external_field: 0.35,
        time_steps: 4,
        dt: 0.2,
    };

    // Dense evolution (only RZ rotations)
    let mut dense_state = State::one_state(3);
    let circuit = create_circuit(3, &params);
    circuit
        .execute(&mut dense_state)
        .expect("Dense field-only evolution failed");
    let dense_vector = dense_state.vector().to_vec();
    let dense_energy = calculate_energy_efficient(&dense_state, &params);

    // MPS evolution should match exactly because only single-qubit gates occur.
    let mut mps_state = MpsState::one_state(
        3,
        MpsConfig {
            max_bond_dim: 4,
            truncation_threshold: 1e-12,
        },
    );
    evolve_heisenberg_mps(&mut mps_state, &params);
    let mps_vector = mps_state.to_state_vector();
    let mps_energy = calculate_energy_mps(&mps_state, &params);

    for (idx, (dense_amp, mps_amp)) in dense_vector.iter().zip(&mps_vector).enumerate() {
        assert!(
            (*dense_amp - *mps_amp).norm() < 1e-10,
            "Field-only mismatch at basis {idx}: dense={dense_amp}, mps={mps_amp}"
        );
    }
    assert!(
        (dense_energy - mps_energy).abs() < 1e-10,
        "Field-only energy mismatch: dense={dense_energy}, mps={mps_energy}"
    );
}
