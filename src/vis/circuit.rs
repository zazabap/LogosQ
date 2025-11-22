//! Circuit visualization module for quantum circuits.

use crate::circuits::Circuit;
use std::collections::HashMap;
use std::io::Result;

/// Generates a text-based visualization of the circuit
pub fn text_diagram(circuit: &Circuit) -> String {
    let num_qubits = circuit.num_qubits();
    let num_ops = circuit.num_operations();

    // Create empty grid
    let mut grid: Vec<Vec<char>> = vec![vec![' '; num_ops * 4 + 2]; num_qubits * 2 + 1];

    // Draw qubit lines
    for q in 0..num_qubits {
        let line_idx = q * 2 + 1;
        for col in 0..grid[0].len() {
            grid[line_idx][col] = '─';
        }

        // Add qubit labels
        let label = format!("q{}", q);
        for (i, c) in label.chars().enumerate() {
            if i < 3 {
                // Prevent overflowing
                grid[line_idx][i] = c;
            }
        }
    }

    // Draw gates
    for (op_idx, op) in circuit.operations().iter().enumerate() {
        let col = op_idx * 4 + 4; // Position of the gate

        match op.qubits().len() {
            // Single-qubit gate
            1 => {
                let qubit = op.qubits()[0];
                let row = qubit * 2 + 1;

                // Gate symbol (use first character of name)
                let symbol = op.name().chars().next().unwrap_or('?');
                grid[row][col] = symbol;

                // Box around symbol
                grid[row - 1][col - 1] = '┌';
                grid[row - 1][col] = '─';
                grid[row - 1][col + 1] = '┐';
                grid[row][col - 1] = '│';
                grid[row][col + 1] = '│';
                grid[row + 1][col - 1] = '└';
                grid[row + 1][col] = '─';
                grid[row + 1][col + 1] = '┘';
            }

            // Two-qubit gate
            2 => {
                let control = op.qubits()[0];
                let target = op.qubits()[1];
                let control_row = control * 2 + 1;
                let target_row = target * 2 + 1;

                // Handle CNOT specially
                if op.name().contains("CNOT") || op.name().contains("CX") {
                    // Draw control point
                    grid[control_row][col] = '●';

                    // Draw vertical line connecting control and target
                    let (min_row, max_row) = if control_row < target_row {
                        (control_row, target_row)
                    } else {
                        (target_row, control_row)
                    };

                    for row in min_row + 1..max_row {
                        grid[row][col] = '│';
                    }

                    // Draw target (X gate)
                    grid[target_row][col] = '⊕';
                } else {
                    // For other two-qubit gates
                    let first_letter = op.name().chars().next().unwrap_or('?');

                    // Draw control points
                    grid[control_row][col] = '●';
                    grid[target_row][col] = '●';

                    // Draw vertical line
                    let (min_row, max_row) = if control_row < target_row {
                        (control_row, target_row)
                    } else {
                        (target_row, control_row)
                    };

                    for row in min_row + 1..max_row {
                        grid[row][col] = '│';
                    }

                    // Add gate name
                    if target_row + 2 < grid.len() {
                        let label = format!("{}", first_letter);
                        for (i, c) in label.chars().enumerate() {
                            if col + i < grid[0].len() {
                                grid[max_row + 1][col + i] = c;
                            }
                        }
                    }
                }
            }

            // Multi-qubit gates (like Toffoli)
            _ => {
                // Sort qubits by index
                let mut qubits = op.qubits().to_vec();
                qubits.sort();

                // The last qubit is usually the target
                let target = *op.qubits().last().unwrap();
                let target_row = target * 2 + 1;

                // Draw target
                grid[target_row][col] = 'T'; // T for Toffoli or multiple controls

                // Draw controls and connecting lines
                let min_row = qubits[0] * 2 + 1;
                let max_row = qubits[qubits.len() - 1] * 2 + 1;

                for &q in &qubits[..qubits.len() - 1] {
                    let q_row = q * 2 + 1;
                    grid[q_row][col] = '●';
                }

                // Draw vertical line
                for row in min_row + 1..max_row {
                    if row % 2 == 0 {
                        // Only draw on spacer rows
                        grid[row][col] = '│';
                    }
                }
            }
        }
    }

    // Convert grid to string
    let mut result = String::new();
    if let Some(name) = circuit.name() {
        result.push_str(&format!("Circuit: {}\n", name));
    }

    for row in grid {
        result.push_str(&row.iter().collect::<String>());
        result.push('\n');
    }

    result
}

/// Generates an SVG visualization of the circuit
pub fn svg_diagram(circuit: &Circuit) -> String {
    let num_qubits = circuit.num_qubits();
    let num_ops = circuit.num_operations();

    // SVG constants
    let cell_width = 60;
    let cell_height = 40;
    let margin = 50;
    let width = num_ops * cell_width + 2 * margin;
    let height = num_qubits * cell_height + 2 * margin;

    // Start SVG
    let mut svg = format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .qubit-line {{ stroke: #000; stroke-width: 1; }}
            .gate-box {{ fill: white; stroke: #000; stroke-width: 1; }}
            .gate-text {{ font-family: sans-serif; font-size: 14px; text-anchor: middle; dominant-baseline: middle; }}
            .control-point {{ fill: #000; stroke: none; }}
            .connector-line {{ stroke: #000; stroke-width: 1; }}
            .qubit-label {{ font-family: sans-serif; font-size: 14px; text-anchor: start; dominant-baseline: middle; }}
        </style>
        "#,
        width, height
    );

    // Add circuit name if available
    if let Some(name) = circuit.name() {
        svg.push_str(&format!(
            r#"<text x="{}" y="20" font-family="sans-serif" font-size="16px" text-anchor="middle">{}</text>"#,
            width / 2, name
        ));
    }

    // Draw qubit lines
    for q in 0..num_qubits {
        let y = q * cell_height + margin;

        // Qubit label
        svg.push_str(&format!(
            r#"<text x="10" y="{}" class="qubit-label">q{}</text>"#,
            y, q
        ));

        // Qubit line
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="qubit-line" />"#,
            margin,
            y,
            width - margin,
            y
        ));
    }

    // Define gate colors
    let mut gate_colors = HashMap::new();
    gate_colors.insert("H", "#E6F0FF"); // Light blue for Hadamard
    gate_colors.insert("X", "#FFE6E6"); // Light red for X
    gate_colors.insert("Y", "#FFE6FF"); // Light purple for Y
    gate_colors.insert("Z", "#E6FFE6"); // Light green for Z
    gate_colors.insert("RX", "#FFE6CC"); // Light orange for RX
    gate_colors.insert("RY", "#FFFFCC"); // Light yellow for RY
    gate_colors.insert("RZ", "#CCE6FF"); // Light blue for RZ
    gate_colors.insert("S", "#E6FFFF"); // Light cyan for S
    gate_colors.insert("T", "#FFE6F0"); // Light pink for T

    // Draw gates
    for (op_idx, op) in circuit.operations().iter().enumerate() {
        let x = op_idx * cell_width + margin + cell_width / 2;

        match op.qubits().len() {
            // Single-qubit gate
            1 => {
                let qubit = op.qubits()[0];
                let y = qubit * cell_height + margin;

                // Determine gate color
                let gate_prefix = op.name().split('(').next().unwrap_or("").trim();
                let fill_color = gate_colors
                    .get(gate_prefix)
                    .unwrap_or(&"#FFFFFF")
                    .to_string();

                // Gate box
                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" class="gate-box" fill="{}" />"#,
                    x - 15,
                    y - 15,
                    30,
                    30,
                    fill_color
                ));

                // Gate label
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" class="gate-text">{}</text>"#,
                    x, y, op.name()
                ));
            }

            // Two-qubit gate
            2 => {
                let control = op.qubits()[0];
                let target = op.qubits()[1];
                let control_y = control * cell_height + margin;
                let target_y = target * cell_height + margin;

                // Vertical connector line
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line" />"#,
                    x, control_y, x, target_y
                ));

                if op.name().contains("CNOT") || op.name().contains("CX") {
                    // Control point
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="5" class="control-point" />"#,
                        x, control_y
                    ));

                    // Target (X gate)
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="10" class="gate-box" fill="none" />"#,
                        x, target_y
                    ));
                    svg.push_str(&format!(
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line" />"#,
                        x - 10,
                        target_y,
                        x + 10,
                        target_y
                    ));
                    svg.push_str(&format!(
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line" />"#,
                        x,
                        target_y - 10,
                        x,
                        target_y + 10
                    ));
                } else if op.name().contains("CZ") {
                    // Control points
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="5" class="control-point" />"#,
                        x, control_y
                    ));

                    // Z gate
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="15" class="gate-box" fill="none" />"#,
                        x, target_y
                    ));
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" class="gate-text">Z</text>"#,
                        x, target_y
                    ));
                } else {
                    // Generic two-qubit gate
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="5" class="control-point" />"#,
                        x, control_y
                    ));

                    // Target gate label
                    svg.push_str(&format!(
                        r#"<rect x="{}" y="{}" width="{}" height="{}" class="gate-box" />"#,
                        x - 15,
                        target_y - 15,
                        30,
                        30
                    ));
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" class="gate-text">{}</text>"#,
                        x, target_y, op.name()
                    ));
                }
            }

            // Multi-qubit gates
            _ => {
                // Sort qubits by index
                let mut sorted_qubits = op.qubits().to_vec();
                sorted_qubits.sort();

                let min_y = sorted_qubits[0] * cell_height + margin;
                let max_y = sorted_qubits[sorted_qubits.len() - 1] * cell_height + margin;

                // Vertical connector
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line" />"#,
                    x, min_y, x, max_y
                ));

                // Control points
                for &q in &sorted_qubits[..sorted_qubits.len() - 1] {
                    let q_y = q * cell_height + margin;
                    svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="5" class="control-point" />"#,
                        x, q_y
                    ));
                }

                // Target gate
                let target = sorted_qubits[sorted_qubits.len() - 1];
                let target_y = target * cell_height + margin;

                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" class="gate-box" />"#,
                    x - 15,
                    target_y - 15,
                    30,
                    30
                ));
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" class="gate-text">{}</text>"#,
                    x,
                    target_y,
                    op.name().chars().next().unwrap_or('?')
                ));
            }
        }
    }

    // Close SVG tag
    svg.push_str("</svg>");
    svg
}

/// Save the SVG circuit diagram to a file
pub fn save_svg_diagram(circuit: &Circuit, filename: &str) -> Result<()> {
    let svg = svg_diagram(circuit);
    std::fs::write(filename, svg)
}

/// Generate and open an HTML visualization of the circuit
pub fn view_circuit(circuit: &Circuit) -> Result<()> {
    let svg = svg_diagram(circuit);

    let html = format!(
        r#"<!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Circuit Visualization</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .circuit-container {{ overflow: auto; border: 1px solid #ccc; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Quantum Circuit Visualization</h1>
            <div class="circuit-container">
                {}
            </div>
        </body>
        </html>"#,
        svg
    );

    // Save HTML to temporary file
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("quantum_circuit.html");
    std::fs::write(&file_path, html)?;

    // Open the HTML file in the default browser
    if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(&["/C", "start", file_path.to_str().unwrap()])
            .spawn()?;
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(file_path).spawn()?;
    } else {
        std::process::Command::new("xdg-open")
            .arg(file_path)
            .spawn()?;
    }

    Ok(())
}

// Implement Visualizable trait for Circuit
impl crate::vis::Visualizable for Circuit {
    fn visualize(&self) -> String {
        text_diagram(self)
    }

    fn visualize_as_svg(&self) -> String {
        svg_diagram(self)
    }

    fn save_visualization(&self, filename: &str) -> Result<()> {
        save_svg_diagram(self, filename)
    }

    fn view(&self) -> Result<()> {
        view_circuit(self)
    }
}
