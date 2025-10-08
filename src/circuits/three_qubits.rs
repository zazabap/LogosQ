use super::Circuit;
use crate::gates::optimized::ToffoliGate;

impl Circuit {
    /// Adds a Toffoli gate to the circuit - OPTIMIZED
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        assert!(
            control1 < self.num_qubits && control2 < self.num_qubits && target < self.num_qubits,
            "Qubit indices out of range"
        );
        assert!(
            control1 != control2 && control1 != target && control2 != target,
            "Control and target qubits must be different"
        );

        let gate = ToffoliGate {
            control1,
            control2,
            target,
            num_qubits: self.num_qubits,
        };

        self.add_operation(gate, vec![control1, control2, target], "Toffoli");
        self
    }
}
