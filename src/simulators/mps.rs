use crate::algorithms::xyz_heisenberg::HeisenbergParameters;
use crate::error::Result;
use crate::states::{QuantumStateBackend, State};
use nalgebra::{DMatrix, SVD};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

const PHYSICAL_DIM: usize = 2;

/// Configuration parameters that control the tensor-network approximation.
#[derive(Clone, Copy, Debug)]
pub struct MpsConfig {
    /// Maximum allowed bond dimension after SVD truncation.
    pub max_bond_dim: usize,
    /// Singular values below this threshold are discarded.
    pub truncation_threshold: f64,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            truncation_threshold: 1e-8,
        }
    }
}

/// Simple Matrix Product State representation with open boundary conditions.
#[derive(Clone, Debug)]
pub struct MpsState {
    tensors: Vec<Array3<Complex64>>, // (left_bond, physical, right_bond)
    max_bond_dim: usize,
    truncation_threshold: f64,
}

impl MpsState {
    /// Creates an |111...1⟩ product state (all spins up in computational basis).
    pub fn one_state(num_qubits: usize, config: MpsConfig) -> Self {
        assert!(num_qubits > 0, "MPS backend requires at least one qubit");
        let mut tensors = Vec::with_capacity(num_qubits);

        for site in 0..num_qubits {
            let left_dim = if site == 0 { 1 } else { 1 };
            let right_dim = if site == num_qubits - 1 { 1 } else { 1 };
            let mut tensor = Array3::from_elem(
                (left_dim, PHYSICAL_DIM, right_dim),
                Complex64::new(0.0, 0.0),
            );

            // Populate |1⟩ amplitude (basis index 1)
            for left in 0..left_dim {
                for right in 0..right_dim {
                    tensor[[left, 1, right]] = Complex64::new(1.0, 0.0);
                }
            }
            tensors.push(tensor);
        }

        Self {
            tensors,
            max_bond_dim: config.max_bond_dim.max(1),
            truncation_threshold: config.truncation_threshold,
        }
    }

    /// Creates an |000...0⟩ product state.
    pub fn zero_state(num_qubits: usize, config: MpsConfig) -> Self {
        assert!(num_qubits > 0, "MPS backend requires at least one qubit");
        let mut tensors = Vec::with_capacity(num_qubits);

        for site in 0..num_qubits {
            let left_dim = if site == 0 { 1 } else { 1 };
            let right_dim = if site == num_qubits - 1 { 1 } else { 1 };
            let mut tensor = Array3::from_elem(
                (left_dim, PHYSICAL_DIM, right_dim),
                Complex64::new(0.0, 0.0),
            );

            for left in 0..left_dim {
                for right in 0..right_dim {
                    tensor[[left, 0, right]] = Complex64::new(1.0, 0.0);
                }
            }
            tensors.push(tensor);
        }

        Self {
            tensors,
            max_bond_dim: config.max_bond_dim.max(1),
            truncation_threshold: config.truncation_threshold,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.tensors.len()
    }

    /// Expands the MPS into a full state vector (useful for testing/comparison).
    pub fn to_state_vector(&self) -> Vec<Complex64> {
        let n = self.num_qubits();
        let dim = 1usize << n;
        let mut amplitudes = Vec::with_capacity(dim);

        for basis in 0..dim {
            let mut left_vec = vec![Complex64::new(1.0, 0.0)];

            for (site, tensor) in self.tensors.iter().enumerate() {
                let bit = (basis >> (n - 1 - site)) & 1;
                let (left_dim, _, right_dim) = tensor.dim();
                assert_eq!(
                    left_vec.len(),
                    left_dim,
                    "Bond dimension mismatch at site {site}"
                );

                let mut next = vec![Complex64::new(0.0, 0.0); right_dim];
                for l in 0..left_dim {
                    let left_amp = left_vec[l];
                    if left_amp == Complex64::new(0.0, 0.0) {
                        continue;
                    }
                    for r in 0..right_dim {
                        next[r] += left_amp * tensor[[l, bit, r]];
                    }
                }

                left_vec = next;
            }

            assert_eq!(
                left_vec.len(),
                1,
                "Final bond dimension must collapse to scalar amplitude"
            );
            amplitudes.push(left_vec[0]);
        }

        amplitudes
    }

    /// Applies a single-qubit gate (2x2 unitary) to the specified site.
    pub fn apply_single_qubit(&mut self, site: usize, gate: &Array2<Complex64>) {
        let tensor = &mut self.tensors[site];
        let (left_dim, _, right_dim) = tensor.dim();
        let mut updated = tensor.clone();

        for l in 0..left_dim {
            for r in 0..right_dim {
                for target_idx in 0..PHYSICAL_DIM {
                    let mut amplitude = Complex64::new(0.0, 0.0);
                    for source_idx in 0..PHYSICAL_DIM {
                        amplitude += gate[[target_idx, source_idx]] * tensor[[l, source_idx, r]];
                    }
                    updated[[l, target_idx, r]] = amplitude;
                }
            }
        }

        *tensor = updated;
    }

    /// Applies a nearest-neighbor two-qubit gate using TEBD-style SVD compression.
    pub fn apply_two_qubit(&mut self, site: usize, gate: &Array2<Complex64>) {
        let next = site + 1;
        let left_tensor = self.tensors[site].clone();
        let right_tensor = self.tensors[next].clone();

        let (left_dim, _, shared) = left_tensor.dim();
        let (_, _, right_dim) = right_tensor.dim();

        // Merge tensors along shared bond.
        let mut theta =
            vec![Complex64::new(0.0, 0.0); left_dim * PHYSICAL_DIM * PHYSICAL_DIM * right_dim];
        for l in 0..left_dim {
            for r in 0..right_dim {
                for s in 0..PHYSICAL_DIM {
                    for t in 0..PHYSICAL_DIM {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for mid in 0..shared {
                            acc += left_tensor[[l, s, mid]] * right_tensor[[mid, t, r]];
                        }
                        let idx = ((((l * PHYSICAL_DIM) + s) * PHYSICAL_DIM) + t) * right_dim + r;
                        theta[idx] = acc;
                    }
                }
            }
        }

        // Apply two-qubit gate to physical indices.
        for l in 0..left_dim {
            for r in 0..right_dim {
                let mut vec = [Complex64::new(0.0, 0.0); 4];
                for s in 0..PHYSICAL_DIM {
                    for t in 0..PHYSICAL_DIM {
                        let idx = s * PHYSICAL_DIM + t;
                        let flat = ((((l * PHYSICAL_DIM) + s) * PHYSICAL_DIM) + t) * right_dim + r;
                        vec[idx] = theta[flat];
                    }
                }

                let mut new_vec = [Complex64::new(0.0, 0.0); 4];
                for row in 0..4 {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for col in 0..4 {
                        acc += gate[[row, col]] * vec[col];
                    }
                    new_vec[row] = acc;
                }

                for s in 0..PHYSICAL_DIM {
                    for t in 0..PHYSICAL_DIM {
                        let idx = s * PHYSICAL_DIM + t;
                        let flat = ((((l * PHYSICAL_DIM) + s) * PHYSICAL_DIM) + t) * right_dim + r;
                        theta[flat] = new_vec[idx];
                    }
                }
            }
        }

        // Reshape theta into matrix (left_dim * PHYSICAL_DIM) x (PHYSICAL_DIM * right_dim)
        let rows = left_dim * PHYSICAL_DIM;
        let cols = PHYSICAL_DIM * right_dim;
        let dm = DMatrix::from_row_slice(rows, cols, &theta);
        let svd = SVD::new(dm, true, true);
        let u = svd.u.expect("SVD without U");
        let v_t = svd.v_t.expect("SVD without V");
        let singulars = svd.singular_values;

        let mut keep = 0usize;
        for (idx, sigma) in singulars.iter().enumerate() {
            if idx < self.max_bond_dim && *sigma > self.truncation_threshold {
                keep += 1;
            }
        }
        if keep == 0 {
            keep = 1.min(singulars.len());
        }

        let mut left_tensor_new =
            Array3::from_elem((left_dim, PHYSICAL_DIM, keep), Complex64::new(0.0, 0.0));
        let mut right_tensor_new =
            Array3::from_elem((keep, PHYSICAL_DIM, right_dim), Complex64::new(0.0, 0.0));

        for alpha in 0..keep {
            let sqrt_sigma = singulars[alpha].sqrt();
            for l in 0..left_dim {
                for s in 0..PHYSICAL_DIM {
                    let row = l * PHYSICAL_DIM + s;
                    left_tensor_new[[l, s, alpha]] = u[(row, alpha)] * sqrt_sigma;
                }
            }

            for r in 0..right_dim {
                for t in 0..PHYSICAL_DIM {
                    let col = t * right_dim + r;
                    // Directly use V† entries provided by nalgebra.
                    let v_entry = v_t[(alpha, col)];
                    right_tensor_new[[alpha, t, r]] = sqrt_sigma * v_entry;
                }
            }
        }

        self.tensors[site] = left_tensor_new;
        self.tensors[next] = right_tensor_new;
    }

    /// Apply a Pauli-X rotation to a single site.
    pub fn apply_pauli_x(&mut self, site: usize) {
        let gate = pauli_x();
        self.apply_single_qubit(site, &gate);
    }

    /// Apply a Hadamard gate to a single site.
    pub fn apply_hadamard(&mut self, site: usize) {
        let gate = hadamard_matrix();
        self.apply_single_qubit(site, &gate);
    }

    /// Apply an adjacent SWAP gate between site and site + 1.
    pub fn apply_swap_gate(&mut self, site: usize) {
        assert!(
            site + 1 < self.num_qubits(),
            "Swap gate requires a valid neighbor"
        );
        let gate = swap_unitary();
        self.apply_two_qubit(site, &gate);
    }

    /// Apply a controlled-phase rotation between any two qubits.
    ///
    /// For non-neighboring qubits, this routine inserts SWAP networks to bring the
    /// targets together, applies the gate, then restores the original ordering.
    pub fn apply_controlled_phase(&mut self, mut control: usize, mut target: usize, angle: f64) {
        assert!(control != target, "Control and target must differ");
        if control > target {
            std::mem::swap(&mut control, &mut target);
        }
        assert!(
            target < self.num_qubits(),
            "Target qubit out of range for controlled phase"
        );

        let mut swaps = Vec::new();
        let mut current = target;
        while current > control + 1 {
            let swap_site = current - 1;
            self.apply_swap_gate(swap_site);
            swaps.push(swap_site);
            current -= 1;
        }

        let gate = controlled_phase_unitary(angle);
        self.apply_two_qubit(control, &gate);

        for site in swaps.into_iter().rev() {
            self.apply_swap_gate(site);
        }
    }

    /// Convert the MPS into a dense `State`.
    pub fn to_dense_state(&self) -> State {
        let amplitudes = self.to_state_vector();
        let vector = Array1::from(amplitudes);
        State::new_unchecked(vector, self.num_qubits())
    }

    fn identity_env_left(&self, upto: usize) -> Array2<Complex64> {
        let mut env = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        let identity = identity_single();
        for site in 0..upto {
            env = propagate_left_env(&env, &self.tensors[site], &identity);
        }
        env
    }

    fn identity_env_right(&self, from_site: usize) -> Array2<Complex64> {
        let mut env = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        let identity = identity_single();
        for site in (from_site..self.tensors.len()).rev() {
            env = propagate_right_env(&env, &self.tensors[site], &identity);
        }
        env
    }

    pub fn expectation_one_site(&self, site: usize, operator: &Array2<Complex64>) -> f64 {
        let left_env = self.identity_env_left(site);
        let right_env = self.identity_env_right(site + 1);
        let tensor = &self.tensors[site];
        let (left_dim, _, right_dim) = tensor.dim();

        let mut value = Complex64::new(0.0, 0.0);
        for l in 0..left_dim {
            for lp in 0..left_dim {
                let left_val = left_env[[l, lp]];
                if left_val == Complex64::new(0.0, 0.0) {
                    continue;
                }
                for r in 0..right_dim {
                    for rp in 0..right_dim {
                        let right_val = right_env[[r, rp]];
                        if right_val == Complex64::new(0.0, 0.0) {
                            continue;
                        }
                        for s in 0..PHYSICAL_DIM {
                            for t in 0..PHYSICAL_DIM {
                                let op = operator[[s, t]];
                                if op == Complex64::new(0.0, 0.0) {
                                    continue;
                                }
                                value += left_val
                                    * tensor[[l, s, r]]
                                    * op
                                    * tensor[[lp, t, rp]].conj()
                                    * right_val;
                            }
                        }
                    }
                }
            }
        }

        value.re
    }

    pub fn expectation_two_site(&self, site: usize, operator: &Array2<Complex64>) -> f64 {
        let left_env = self.identity_env_left(site);
        let right_env = self.identity_env_right(site + 2);

        let tensor_a = &self.tensors[site];
        let tensor_b = &self.tensors[site + 1];
        let (left_dim, _, shared) = tensor_a.dim();
        let (_, _, right_dim) = tensor_b.dim();

        let mut value = Complex64::new(0.0, 0.0);
        for l in 0..left_dim {
            for lp in 0..left_dim {
                let left_val = left_env[[l, lp]];
                if left_val == Complex64::new(0.0, 0.0) {
                    continue;
                }
                for r in 0..right_dim {
                    for rp in 0..right_dim {
                        let right_val = right_env[[r, rp]];
                        if right_val == Complex64::new(0.0, 0.0) {
                            continue;
                        }

                        for beta in 0..shared {
                            for betap in 0..shared {
                                for s in 0..PHYSICAL_DIM {
                                    for t in 0..PHYSICAL_DIM {
                                        for u in 0..PHYSICAL_DIM {
                                            for v in 0..PHYSICAL_DIM {
                                                let idx_row = s * PHYSICAL_DIM + t;
                                                let idx_col = u * PHYSICAL_DIM + v;
                                                let op = operator[[idx_row, idx_col]];
                                                if op == Complex64::new(0.0, 0.0) {
                                                    continue;
                                                }
                                                value += left_val
                                                    * tensor_a[[l, s, beta]]
                                                    * tensor_b[[beta, t, r]]
                                                    * op
                                                    * tensor_a[[lp, u, betap]].conj()
                                                    * tensor_b[[betap, v, rp]].conj()
                                                    * right_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        value.re
    }
}

/// Evolves an MPS state under the XYZ-Heisenberg Hamiltonian using first-order TEBD.
pub fn evolve_heisenberg_mps(state: &mut MpsState, params: &HeisenbergParameters) {
    let num_qubits = state.num_qubits();

    for _ in 0..params.time_steps {
        let xx_gate = xx_unitary(params.jx * params.dt);
        let yy_gate = yy_unitary(params.jy * params.dt);
        let zz_gate = zz_unitary(params.jz * params.dt);

        for pair in 0..num_qubits - 1 {
            state.apply_two_qubit(pair, &xx_gate);
        }
        for pair in 0..num_qubits - 1 {
            state.apply_two_qubit(pair, &yy_gate);
        }
        for pair in 0..num_qubits - 1 {
            state.apply_two_qubit(pair, &zz_gate);
        }

        if params.external_field != 0.0 {
            let rz_gate = rz_unitary(params.external_field * params.dt);
            for site in 0..num_qubits {
                state.apply_single_qubit(site, &rz_gate);
            }
        }
    }
}

/// Calculates the energy expectation value ⟨ψ|H|ψ⟩ for an MPS state.
pub fn calculate_energy_mps(state: &MpsState, params: &HeisenbergParameters) -> f64 {
    let num_qubits = state.num_qubits();
    let mut energy = 0.0;

    for i in 0..num_qubits - 1 {
        energy += params.jx * state.expectation_two_site(i, &xx_operator());
        energy += params.jy * state.expectation_two_site(i, &yy_operator());
        energy += params.jz * state.expectation_two_site(i, &zz_operator());
    }

    if params.external_field != 0.0 {
        for i in 0..num_qubits {
            energy += params.external_field * state.expectation_one_site(i, &pauli_z());
        }
    }

    energy
}

fn identity_single() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (PHYSICAL_DIM, PHYSICAL_DIM),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_x() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (PHYSICAL_DIM, PHYSICAL_DIM),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_y() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (PHYSICAL_DIM, PHYSICAL_DIM),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
}

fn pauli_z() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (PHYSICAL_DIM, PHYSICAL_DIM),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap()
}

fn hadamard_matrix() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (PHYSICAL_DIM, PHYSICAL_DIM),
        vec![
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ],
    )
    .unwrap()
}

fn swap_unitary() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap()
}

fn controlled_phase_unitary(angle: f64) -> Array2<Complex64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, angle),
        ],
    )
    .unwrap()
}

fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let dim_a = a.dim().0;
    let dim_b = b.dim().0;
    let mut result = Array2::from_elem((dim_a * dim_b, dim_a * dim_b), Complex64::new(0.0, 0.0));

    for i in 0..dim_a {
        for j in 0..dim_a {
            for k in 0..dim_b {
                for l in 0..dim_b {
                    let row = i * dim_b + k;
                    let col = j * dim_b + l;
                    result[[row, col]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

fn xx_operator() -> Array2<Complex64> {
    kron(&pauli_x(), &pauli_x())
}

fn yy_operator() -> Array2<Complex64> {
    kron(&pauli_y(), &pauli_y())
}

fn zz_operator() -> Array2<Complex64> {
    kron(&pauli_z(), &pauli_z())
}

fn xx_unitary(theta: f64) -> Array2<Complex64> {
    rotate_two_qubit(&xx_operator(), theta)
}

fn yy_unitary(theta: f64) -> Array2<Complex64> {
    rotate_two_qubit(&yy_operator(), theta)
}

fn zz_unitary(theta: f64) -> Array2<Complex64> {
    rotate_two_qubit(&zz_operator(), theta)
}

fn rotate_two_qubit(pauli: &Array2<Complex64>, theta: f64) -> Array2<Complex64> {
    let ident = kron(&identity_single(), &identity_single());
    let mut unitary = ident.mapv(|v| v * theta.cos());
    let factor = Complex64::new(0.0, -theta.sin());
    let pauli_term = pauli.mapv(|v| v * factor);
    unitary += &pauli_term;
    unitary
}

fn rz_unitary(theta: f64) -> Array2<Complex64> {
    let mut mat = Array2::from_elem((PHYSICAL_DIM, PHYSICAL_DIM), Complex64::new(0.0, 0.0));
    mat[[0, 0]] = Complex64::from_polar(1.0, -theta / 2.0);
    mat[[1, 1]] = Complex64::from_polar(1.0, theta / 2.0);
    mat
}

fn propagate_left_env(
    env: &Array2<Complex64>,
    tensor: &Array3<Complex64>,
    operator: &Array2<Complex64>,
) -> Array2<Complex64> {
    let (left_dim, _, right_dim) = tensor.dim();
    let mut new_env = Array2::from_elem((right_dim, right_dim), Complex64::new(0.0, 0.0));

    for l in 0..left_dim {
        for lp in 0..left_dim {
            let left_val = env[[l, lp]];
            if left_val == Complex64::new(0.0, 0.0) {
                continue;
            }
            for r in 0..right_dim {
                for rp in 0..right_dim {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for s in 0..PHYSICAL_DIM {
                        for t in 0..PHYSICAL_DIM {
                            let op = operator[[s, t]];
                            if op == Complex64::new(0.0, 0.0) {
                                continue;
                            }
                            acc += tensor[[l, s, r]] * op * tensor[[lp, t, rp]].conj();
                        }
                    }
                    new_env[[r, rp]] += left_val * acc;
                }
            }
        }
    }

    new_env
}

fn propagate_right_env(
    env: &Array2<Complex64>,
    tensor: &Array3<Complex64>,
    operator: &Array2<Complex64>,
) -> Array2<Complex64> {
    let (left_dim, _, right_dim) = tensor.dim();
    let mut new_env = Array2::from_elem((left_dim, left_dim), Complex64::new(0.0, 0.0));

    for r in 0..right_dim {
        for rp in 0..right_dim {
            let right_val = env[[r, rp]];
            if right_val == Complex64::new(0.0, 0.0) {
                continue;
            }
            for l in 0..left_dim {
                for lp in 0..left_dim {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for s in 0..PHYSICAL_DIM {
                        for t in 0..PHYSICAL_DIM {
                            let op = operator[[s, t]];
                            if op == Complex64::new(0.0, 0.0) {
                                continue;
                            }
                            acc += tensor[[l, s, r]] * op * tensor[[lp, t, rp]].conj();
                        }
                    }
                    new_env[[l, lp]] += acc * right_val;
                }
            }
        }
    }

    new_env
}

impl QuantumStateBackend for MpsState {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }

    fn apply_single_qubit_matrix(&mut self, qubit: usize, matrix: &Array2<Complex64>) -> Result<()> {
        if qubit >= self.num_qubits() {
            return Err(crate::error::LogosQError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits(),
            });
        }
        self.apply_single_qubit(qubit, matrix);
        Ok(())
    }

    fn apply_two_qubit_matrix(
        &mut self,
        control: usize,
        target: usize,
        matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if control >= self.num_qubits() || target >= self.num_qubits() {
            return Err(crate::error::LogosQError::InvalidQubitIndex {
                index: control.max(target),
                num_qubits: self.num_qubits(),
            });
        }

        // MPS works best with nearest-neighbor gates
        // For non-adjacent qubits, we need to use SWAP networks
        if control.abs_diff(target) == 1 {
            // Adjacent qubits - direct application
            let min_qubit = control.min(target);
            self.apply_two_qubit(min_qubit, matrix);
        } else {
            // Non-adjacent - use controlled phase with SWAP network
            // For now, convert to controlled phase if it's a CP gate, otherwise use SWAP
            if is_controlled_phase(matrix) {
                let angle = extract_phase_angle(matrix);
                self.apply_controlled_phase(control, target, angle);
            } else {
                // Use SWAP network to bring qubits together
                let mut swaps = Vec::new();
                let mut current = target;
                while current > control + 1 {
                    let swap_site = current - 1;
                    self.apply_swap_gate(swap_site);
                    swaps.push(swap_site);
                    current -= 1;
                }

                // Apply gate on adjacent qubits
                self.apply_two_qubit(control, matrix);

                // Reverse SWAPs
                for site in swaps.into_iter().rev() {
                    self.apply_swap_gate(site);
                }
            }
        }
        Ok(())
    }

    fn apply_three_qubit_matrix(
        &mut self,
        _q1: usize,
        _q2: usize,
        _q3: usize,
        _matrix: &Array2<Complex64>,
    ) -> Result<()> {
        // Three-qubit gates are not directly supported in MPS
        // Would need to convert to dense state, apply, then convert back
        // For now, return an error suggesting conversion
        Err(crate::error::LogosQError::OptimizationError {
            message: "Three-qubit gates not directly supported in MPS backend. Convert to dense state first.".to_string(),
        })
    }

    fn apply_full_matrix(&mut self, matrix: &Array2<Complex64>) -> Result<()> {
        // Convert to dense state, apply, then reconstruct MPS
        let mut dense = self.to_dense_state();
        let new_vector = matrix.dot(dense.vector());
        *dense.vector_mut() = new_vector;
        dense.normalize();

        // Reconstruct MPS from dense state
        // For product states, we can reconstruct directly
        // For entangled states, we use a simplified approach that works for many cases
        let num_qubits = self.num_qubits();
        let config = MpsConfig {
            max_bond_dim: self.max_bond_dim,
            truncation_threshold: self.truncation_threshold,
        };

        let vector = dense.vector();
        
        // Check if the state is a product state (all zeros except one element)
        let mut non_zero_count = 0;
        let mut non_zero_idx = 0;
        for (i, &amp) in vector.iter().enumerate() {
            if amp.norm() > 1e-10 {
                non_zero_count += 1;
                non_zero_idx = i;
            }
        }

        if non_zero_count == 1 {
            // Product state - can create directly
            *self = MpsState::zero_state(num_qubits, config);
            // Apply X gates to set the correct bits
            for qubit in 0..num_qubits {
                let bit = (non_zero_idx >> (num_qubits - 1 - qubit)) & 1;
                if bit == 1 {
                    self.apply_pauli_x(qubit);
                }
            }
        } else {
            // For non-product states, try to reconstruct by applying gates
            // This is a simplified approach - start from zero and apply gates to match amplitudes
            // For proper implementation, would need full SVD-based MPS reconstruction
            // For now, we use a heuristic: if state is close to a product state, reconstruct it
            // Otherwise, return an error for truly entangled states
            
            // Check if it's a separable product state (can be written as |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ...)
            // For 2 qubits, check if it's separable
            if num_qubits == 2 {
                // For 2-qubit states, try to reconstruct by applying single-qubit gates
                // This works for product states but not for entangled states
                *self = MpsState::zero_state(num_qubits, config);
                
                // Try to apply gates to match the state
                // This is a simplified reconstruction - works for product states
                // For entangled states, we'd need proper SVD
                let v00 = vector[0];
                let v01 = vector[1];
                let v10 = vector[2];
                let v11 = vector[3];
                
                // If state is separable, we can reconstruct it
                // Check if |v00*v11 - v01*v10| is small (separable condition)
                let det = v00 * v11 - v01 * v10;
                if det.norm() < 1e-8 {
                    // Approximately separable - can reconstruct
                    // For now, use a simple approximation
                    // In practice, this should use proper SVD
                    return Err(crate::error::LogosQError::OptimizationError {
                        message: "Full matrix gates on entangled MPS states require proper SVD reconstruction. Use dense state or decompose gates.".to_string(),
                    });
                } else {
                    // Entangled state
                    return Err(crate::error::LogosQError::OptimizationError {
                        message: "Full matrix gates on entangled MPS states require proper SVD reconstruction. Use dense state or decompose gates.".to_string(),
                    });
                }
            } else {
                // For more qubits, reconstruction is more complex
                return Err(crate::error::LogosQError::OptimizationError {
                    message: "Full matrix gates on entangled MPS states require proper SVD reconstruction. Use dense state or decompose gates.".to_string(),
                });
            }
        }

        Ok(())
    }
}

// Helper to check if a matrix is a controlled phase gate
fn is_controlled_phase(matrix: &Array2<Complex64>) -> bool {
    // Check if matrix has the form:
    // [[1, 0, 0, 0],
    //  [0, 1, 0, 0],
    //  [0, 0, 1, 0],
    //  [0, 0, 0, exp(i*phi)]]
    if matrix.shape() != [4, 4] {
        return false;
    }
    let tol = 1e-10;
    (matrix[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < tol
        && (matrix[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < tol
        && (matrix[[2, 2]] - Complex64::new(1.0, 0.0)).norm() < tol
        && matrix[[0, 1]].norm() < tol
        && matrix[[0, 2]].norm() < tol
        && matrix[[0, 3]].norm() < tol
        && matrix[[1, 0]].norm() < tol
        && matrix[[1, 2]].norm() < tol
        && matrix[[1, 3]].norm() < tol
        && matrix[[2, 0]].norm() < tol
        && matrix[[2, 1]].norm() < tol
        && matrix[[2, 3]].norm() < tol
        && matrix[[3, 0]].norm() < tol
        && matrix[[3, 1]].norm() < tol
        && matrix[[3, 2]].norm() < tol
        && matrix[[3, 3]].norm() - 1.0 < tol
}

// Helper to extract phase angle from controlled phase gate
fn extract_phase_angle(matrix: &Array2<Complex64>) -> f64 {
    matrix[[3, 3]].arg()
}
