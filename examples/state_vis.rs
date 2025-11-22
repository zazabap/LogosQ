use logosq::states::State;
use logosq::vis::Visualizable;
use num_complex::Complex64;
use std::env;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

fn main() -> std::io::Result<()> {
    println!("LogosQ Quantum State Visualization Examples\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let show_all = args.len() <= 1 || args.contains(&String::from("all"));
    let show_basic = show_all || args.contains(&String::from("basic"));
    let show_superposition = show_all || args.contains(&String::from("super"));
    let show_phase = show_all || args.contains(&String::from("phase"));
    let show_arbitrary = show_all || args.contains(&String::from("arb"));
    let show_evolution = show_all || args.contains(&String::from("evolution"));

    if args.len() > 1
        && !show_basic
        && !show_superposition
        && !show_phase
        && !show_arbitrary
        && !show_evolution
    {
        println!("Usage: cargo run --example state_vis [all|basic|super|phase|arb|evolution]");
        println!("  all: Show all visualizations (default)");
        println!("  basic: Show basic states (|0⟩, |1⟩)");
        println!("  super: Show superposition states (|+⟩, |-⟩)");
        println!("  phase: Show phase states (|i⟩, |-i⟩)");
        println!("  arb: Show arbitrary states");
        println!("  evolution: Show state evolution under gates");
        return Ok(());
    }

    println!("Note: This example visualizes single-qubit states using the 3D Bloch sphere.\n");

    // Example 1: Single qubit basis states
    if show_basic {
        println!("=== Example 1: Single Qubit Basis States ===");

        println!("\n|0⟩ state (North Pole of Bloch Sphere):");
        let zero_state = State::zero_state(1);
        println!("{}", zero_state.visualize());
        zero_state.save_visualization("zero_state.svg")?;
        println!("Saved visualization to zero_state.svg");

        println!("\n|1⟩ state (South Pole of Bloch Sphere):");
        let one_state = State::one_state(1);
        println!("{}", one_state.visualize());
        one_state.save_visualization("one_state.svg")?;
        println!("Saved visualization to one_state.svg");
    }

    // Example 2: Single qubit superposition states
    if show_superposition {
        println!("\n=== Example 2: Single Qubit Superposition States ===");

        println!("\n|+⟩ state (Positive X-axis on Bloch Sphere):");
        let plus_state = State::plus_state(1);
        println!("{}", plus_state.visualize());
        plus_state.save_visualization("plus_state.svg")?;
        println!("Saved visualization to plus_state.svg");

        println!("\n|-⟩ state (Negative X-axis on Bloch Sphere):");
        let mut minus_state = State::zero_state(1);
        minus_state.vector_mut()[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        minus_state.vector_mut()[1] = Complex64::new(-FRAC_1_SQRT_2, 0.0);
        println!("{}", minus_state.visualize());
        minus_state.save_visualization("minus_state.svg")?;
        println!("Saved visualization to minus_state.svg");
    }

    // Example 3: Phase states
    if show_phase {
        println!("\n=== Example 3: Phase States on Bloch Sphere ===");

        println!("\n|i⟩ state (Positive Y-axis on Bloch Sphere):");
        let mut i_state = State::zero_state(1);
        i_state.vector_mut()[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        i_state.vector_mut()[1] = Complex64::new(0.0, FRAC_1_SQRT_2);
        println!("{}", i_state.visualize());
        i_state.save_visualization("i_state.svg")?;
        println!("Saved visualization to i_state.svg");

        println!("\n|-i⟩ state (Negative Y-axis on Bloch Sphere):");
        let mut neg_i_state = State::zero_state(1);
        neg_i_state.vector_mut()[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        neg_i_state.vector_mut()[1] = Complex64::new(0.0, -FRAC_1_SQRT_2);
        println!("{}", neg_i_state.visualize());
        neg_i_state.save_visualization("neg_i_state.svg")?;
        println!("Saved visualization to neg_i_state.svg");
    }

    // Example 4: Arbitrary single-qubit states
    if show_arbitrary {
        println!("\n=== Example 4: Arbitrary Single-Qubit States ===");

        // Create an arbitrary state
        println!("\nArbitrary state 1:");
        let mut arbitrary_state1 = State::zero_state(1);
        let theta = PI / 4.0; // 45 degrees from Z-axis
        let phi = PI / 3.0; // 60 degrees azimuthal angle

        arbitrary_state1.vector_mut()[0] = Complex64::new(theta.cos(), 0.0);
        arbitrary_state1.vector_mut()[1] =
            Complex64::new(theta.sin() * phi.cos(), theta.sin() * phi.sin());
        arbitrary_state1.normalize();
        println!("{}", arbitrary_state1.visualize());
        arbitrary_state1.save_visualization("arbitrary_state1.svg")?;
        println!("Saved visualization to arbitrary_state1.svg");

        // Create another arbitrary state
        println!("\nArbitrary state 2:");
        let mut arbitrary_state2 = State::zero_state(1);
        let theta = 2.0 * PI / 3.0; // 120 degrees from Z-axis
        let phi = 3.0 * PI / 4.0; // 135 degrees azimuthal angle

        arbitrary_state2.vector_mut()[0] = Complex64::new(theta.cos(), 0.0);
        arbitrary_state2.vector_mut()[1] =
            Complex64::new(theta.sin() * phi.cos(), theta.sin() * phi.sin());
        arbitrary_state2.normalize();
        println!("{}", arbitrary_state2.visualize());
        arbitrary_state2.save_visualization("arbitrary_state2.svg")?;
        println!("Saved visualization to arbitrary_state2.svg");
    }

    // Display information about interactive viewing
    println!("\n=== Interactive Viewing ===");
    println!("You can open any of the generated SVG files in your browser or SVG viewer.");
    println!("Alternatively, uncomment the view() method in the code to display directly.");

    // Note about multi-qubit states
    println!("\n=== Note on Multi-Qubit States ===");
    println!("The current visualization implementation only supports single-qubit states.");
    println!("Multi-qubit states can be viewed in text format but cannot be visualized on the Bloch sphere.");

    println!("\nAll examples completed successfully!");
    Ok(())
}
