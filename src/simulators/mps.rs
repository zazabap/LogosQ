use crate::algorithms::xyz_heisenberg::HeisenbergParameters;
use crate::states::State;
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
        let dm = DMatrix::from_vec(rows, cols, theta);
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
                    // Convert V† back to V
                    let v_entry = v_t[(alpha, col)].conj();
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
