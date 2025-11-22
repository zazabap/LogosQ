//! State visualization module with 3D Bloch sphere for quantum states.

use crate::states::State;
use crate::vis::Visualizable;
use plotters::prelude::*;
use std::f64::consts::PI;
use std::io::{Error, ErrorKind, Result};

impl Visualizable for State {
    fn visualize(&self) -> String {
        // Simple text-based visualization
        let num_qubits = self.num_qubits();
        let mut output = format!(
            "Quantum State: {} qubit{}\n\n",
            num_qubits,
            if num_qubits != 1 { "s" } else { "" }
        );

        // Show only states with non-negligible probability
        for (i, amplitude) in self.vector().iter().enumerate() {
            let probability = amplitude.norm_sqr();
            if probability > 1e-10 {
                let basis = format!(
                    "{}{}{}",
                    "|",
                    format!("{:0width$b}", i, width = num_qubits),
                    ">"
                );
                output.push_str(&format!(
                    "{}: {:.4}{:+.4}i (p={:.4})\n",
                    basis, amplitude.re, amplitude.im, probability
                ));
            }
        }

        // Add Bloch coordinates for single-qubit states
        if num_qubits == 1 {
            let (x, y, z) = self.calculate_bloch_coordinates();
            output.push_str(&format!(
                "\nBloch coordinates: (x={:.4}, y={:.4}, z={:.4})\n",
                x, y, z
            ));
        }

        output
    }

    fn visualize_as_svg(&self) -> String {
        if self.num_qubits() != 1 {
            return "3D Bloch sphere visualization is only available for single-qubit states."
                .to_string();
        }

        "SVG visualization requires saving to a file.".to_string()
    }

    fn save_visualization(&self, filename: &str) -> Result<()> {
        if self.num_qubits() == 1 {
            self.plot_bloch_sphere_3d(filename)
        } else {
            // Return an error for multi-qubit states
            Err(Error::new(
                ErrorKind::InvalidInput,
                "3D Bloch sphere visualization is only available for single-qubit states.",
            ))
        }
    }

    fn view(&self) -> Result<()> {
        let temp_file = "/tmp/quantum_state.svg";

        // Handle the visualization error explicitly
        if let Err(e) = self.save_visualization(temp_file) {
            return Err(e);
        }

        // Open with system's default viewer
        #[cfg(target_os = "linux")]
        std::process::Command::new("xdg-open")
            .arg(temp_file)
            .spawn()?;

        #[cfg(target_os = "macos")]
        std::process::Command::new("open").arg(temp_file).spawn()?;

        #[cfg(target_os = "windows")]
        std::process::Command::new("cmd")
            .args(&["/C", "start", temp_file])
            .spawn()?;

        Ok(())
    }
}

impl State {
    // 3D Bloch sphere visualization for single qubit states
    fn plot_bloch_sphere_3d(&self, filename: &str) -> Result<()> {
        // Create drawing area
        let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();

        if let Err(e) = root.fill(&WHITE) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Drawing error: {:?}", e),
            ));
        }

        // Calculate Bloch coordinates
        let (x, y, z) = self.calculate_bloch_coordinates();

        // Create 3D chart
        let mut chart = match ChartBuilder::on(&root)
            .caption("3D Bloch Sphere", ("sans-serif", 30).into_font())
            .margin(40)
            .build_cartesian_3d(-1.2..1.2, -1.2..1.2, -1.2..1.2)
        {
            Ok(chart) => chart,
            Err(e) => {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Chart error: {:?}", e),
                ))
            }
        };

        // Configure axes (view_angle is not available in plotters)
        if let Err(e) = chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.1))
            .max_light_lines(3)
            .draw()
        {
            return Err(Error::new(ErrorKind::Other, format!("Axes error: {:?}", e)));
        }

        // Draw Bloch sphere wireframe - generate and draw each line separately
        // Longitude lines (lines of constant φ)
        for i in 0..12 {
            let phi = 2.0 * PI * (i as f64) / 12.0;
            let line_points: Vec<(f64, f64, f64)> = (0..=12)
                .map(|j| {
                    let theta = PI * (j as f64) / 12.0;
                    let x = 0.98 * theta.sin() * phi.cos();
                    let y = 0.98 * theta.sin() * phi.sin();
                    let z = 0.98 * theta.cos();
                    (x, y, z)
                })
                .collect();

            if let Err(e) =
                chart.draw_series(LineSeries::new(line_points, BLACK.mix(0.2).stroke_width(1)))
            {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Longitude line error: {:?}", e),
                ));
            }
        }

        // Latitude lines (lines of constant θ)
        for j in 1..12 {
            let theta = PI * (j as f64) / 12.0;
            let line_points: Vec<(f64, f64, f64)> = (0..=16)
                .map(|i| {
                    let phi = 2.0 * PI * (i as f64) / 16.0;
                    let x = 0.98 * theta.sin() * phi.cos();
                    let y = 0.98 * theta.sin() * phi.sin();
                    let z = 0.98 * theta.cos();
                    (x, y, z)
                })
                .collect();

            if let Err(e) =
                chart.draw_series(LineSeries::new(line_points, BLACK.mix(0.2).stroke_width(1)))
            {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Latitude line error: {:?}", e),
                ));
            }
        }

        // Draw coordinate axes
        if let Err(e) = chart.draw_series(LineSeries::new(
            vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            BLACK.stroke_width(2),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("X-axis drawing error: {:?}", e),
            ));
        }

        if let Err(e) = chart.draw_series(LineSeries::new(
            vec![(0.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
            BLACK.stroke_width(2),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Y-axis drawing error: {:?}", e),
            ));
        }

        if let Err(e) = chart.draw_series(LineSeries::new(
            vec![(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)],
            BLACK.stroke_width(2),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Z-axis drawing error: {:?}", e),
            ));
        }

        // Draw axis labels
        if let Err(e) = chart.draw_series(PointSeries::of_element(
            vec![(1.1, 0.0, 0.0)],
            0,
            BLACK,
            &|coord, _size, _style| Text::new("X", coord, ("sans-serif", 20).into_font()),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("X label error: {:?}", e),
            ));
        }

        if let Err(e) = chart.draw_series(PointSeries::of_element(
            vec![(0.0, 1.1, 0.0)],
            0,
            BLACK,
            &|coord, _size, _style| Text::new("Y", coord, ("sans-serif", 20).into_font()),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Y label error: {:?}", e),
            ));
        }

        if let Err(e) = chart.draw_series(PointSeries::of_element(
            vec![(0.0, 0.0, 1.1)],
            0,
            BLACK,
            &|coord, _size, _style| Text::new("Z", coord, ("sans-serif", 20).into_font()),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Z label error: {:?}", e),
            ));
        }

        // Add reference points for |0⟩, |1⟩, |+⟩, etc. to help with orientation
        let reference_points = [
            ((0.0, 0.0, 1.0), "|0⟩"),   // |0⟩ state (North pole)
            ((0.0, 0.0, -1.0), "|1⟩"),  // |1⟩ state (South pole)
            ((1.0, 0.0, 0.0), "|+⟩"),   // |+⟩ state (+X axis)
            ((-1.0, 0.0, 0.0), "|-⟩"),  // |-⟩ state (-X axis)
            ((0.0, 1.0, 0.0), "|i⟩"),   // |i⟩ state (+Y axis)
            ((0.0, -1.0, 0.0), "|-i⟩"), // |-i⟩ state (-Y axis)
        ];

        for (point, label) in reference_points.iter() {
            // Draw small points at reference states
            if let Err(e) = chart.draw_series(PointSeries::of_element(
                vec![*point],
                3,
                BLACK.mix(0.5),
                &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
            )) {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Reference point error: {:?}", e),
                ));
            }

            // Add labels
            if let Err(e) = chart.draw_series(PointSeries::of_element(
                vec![(point.0 * 1.1, point.1 * 1.1, point.2 * 1.1)],
                0,
                BLACK,
                &|coord, _size, _style| {
                    Text::new(label.to_string(), coord, ("sans-serif", 15).into_font())
                },
            )) {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("Reference label error: {:?}", e),
                ));
            }
        }

        // Normalize coordinates to ensure they're exactly on the sphere surface
        let norm = (x * x + y * y + z * z).sqrt();
        let (x_norm, y_norm, z_norm) = if norm > 1e-10 {
            (x / norm, y / norm, z / norm)
        } else {
            (0.0, 0.0, 1.0) // Default to |0⟩ if state is undefined
        };

        // Draw the state vector using normalized coordinates
        if let Err(e) = chart.draw_series(LineSeries::new(
            vec![
                (0.0, 0.0, 0.0),
                (x_norm * 0.98, y_norm * 0.98, z_norm * 0.98),
            ],
            RED.stroke_width(3),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Vector drawing error: {:?}", e),
            ));
        }

        // Draw the state point using normalized coordinates
        if let Err(e) = chart.draw_series(PointSeries::of_element(
            vec![(x_norm * 0.98, y_norm * 0.98, z_norm * 0.98)],
            6,
            RED.filled(),
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        )) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Point drawing error: {:?}", e),
            ));
        }

        // Show both original and normalized coordinates
        let coords_text = format!(
            "Bloch coordinates: (x={:.3}, y={:.3}, z={:.3}) [norm={:.3}]",
            x, y, z, norm
        );
        if let Err(e) = root.draw_text(
            &coords_text,
            &TextStyle::from(("sans-serif", 20).into_font()).color(&BLACK),
            (50, 550),
        ) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Coordinates text error: {:?}", e),
            ));
        }

        // Add state info
        let state_vec = self.vector();
        let alpha = state_vec[0];
        let beta = state_vec[1];
        let state_text = format!(
            "|ψ⟩ = {:.3}{:+.3}i|0⟩ + {:.3}{:+.3}i|1⟩",
            alpha.re, alpha.im, beta.re, beta.im
        );

        if let Err(e) = root.draw_text(
            &state_text,
            &TextStyle::from(("sans-serif", 20).into_font()).color(&BLACK),
            (50, 50),
        ) {
            return Err(Error::new(
                ErrorKind::Other,
                format!("State text error: {:?}", e),
            ));
        }

        if let Err(e) = root.present() {
            return Err(Error::new(
                ErrorKind::Other,
                format!("SVG rendering error: {:?}", e),
            ));
        }

        Ok(())
    }

    // Helper method to calculate Bloch coordinates
    fn calculate_bloch_coordinates(&self) -> (f64, f64, f64) {
        assert_eq!(
            self.num_qubits(), 1,
            "Bloch coordinates only defined for 1 qubit"
        );

        let state_vec = self.vector();
        let alpha = state_vec[0];
        let beta = state_vec[1];

        let x = 2.0 * (alpha.conj() * beta + alpha * beta.conj()).re;
        let y = 2.0 * (alpha.conj() * beta - alpha * beta.conj()).im;
        let z = alpha.norm_sqr() - beta.norm_sqr();

        (x, y, z)
    }
}
