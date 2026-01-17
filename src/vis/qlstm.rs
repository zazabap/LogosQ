//! QLSTM visualization module
//!
//! This module provides visualization tools for Quantum Long Short-Term Memory (QLSTM)
//! cells and layers, including SVG diagram generation.

use crate::qml::{QLSTMConfig, VQCType, VariationalQuantumCircuit, QLSTM};
use std::io::Result;

/// Generate a text-based QLSTM architecture diagram
pub fn text_diagram(config: &QLSTMConfig) -> String {
    let mut result = String::new();

    result.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    result.push_str("│                    QLSTM Cell Architecture                  │\n");
    result.push_str("├─────────────────────────────────────────────────────────────┤\n");
    result.push_str("│                                                             │\n");
    result.push_str("│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │\n");
    result.push_str("│  │  VQC_f  │    │  VQC_i  │    │  VQC_c  │    │  VQC_o  │  │\n");
    result.push_str("│  │ (Forget)│    │ (Input) │    │ (Cell)  │    │(Output) │  │\n");
    result.push_str("│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘  │\n");
    result.push_str("│       │σ            │σ            │tanh          │σ       │\n");
    result.push_str("│       ▼              ▼              ▼              ▼        │\n");
    result.push_str("│       ○──────────────○──────────────○──────────────○        │\n");
    result.push_str("│       │              │              │              │        │\n");
    result.push_str("│       ▼              ▼              ▼              ▼        │\n");
    result.push_str("│   ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐      │\n");
    result.push_str("│   │  f_t  │     │  i_t  │     │  c̃_t │     │  o_t  │      │\n");
    result.push_str("│   └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘      │\n");
    result.push_str("│       │             │             │             │          │\n");
    result.push_str("│       └──────┬──────┴──────┬──────┘             │          │\n");
    result.push_str("│              ▼             ▼                    │          │\n");
    result.push_str("│         ┌─────────┐   ┌─────────┐               │          │\n");
    result.push_str("│  c_{t-1}│   ⊙     │   │   ⊙     │               │          │\n");
    result.push_str("│    ────►│  f_t    │──►│  i_t    │               │          │\n");
    result.push_str("│         └────┬────┘   └────┬────┘               │          │\n");
    result.push_str("│              └──────┬──────┘                    │          │\n");
    result.push_str("│                     ▼                           │          │\n");
    result.push_str("│                ┌─────────┐                      │          │\n");
    result.push_str("│                │   c_t   │──────────┐           │          │\n");
    result.push_str("│                └────┬────┘          │           │          │\n");
    result.push_str("│                     │              tanh         │          │\n");
    result.push_str("│                     ▼               ▼           ▼          │\n");
    result.push_str("│                ┌─────────┐     ┌─────────┐                 │\n");
    result.push_str("│                │  c_t    │────►│   ⊙     │────► h_t        │\n");
    result.push_str("│                └─────────┘     └─────────┘                 │\n");
    result.push_str("│                                                            │\n");
    result.push_str("├────────────────────────────────────────────────────────────┤\n");
    result.push_str(&format!(
        "│  Input size: {}    Hidden size: {}    Qubits: {}    Layers: {}    │\n",
        config.input_size, config.hidden_size, config.num_qubits, config.num_layers
    ));
    result.push_str("└────────────────────────────────────────────────────────────┘\n");

    result
}

/// Generate a text diagram showing VQC structure
pub fn vqc_text_diagram(vqc: &VariationalQuantumCircuit) -> String {
    let mut result = String::new();

    let vqc_type_name = match vqc.vqc_type {
        VQCType::Simple => "Simple",
        VQCType::BasicEntangling => "BasicEntangling",
        VQCType::StronglyEntangling => "StronglyEntangling",
        VQCType::Custom => "Custom",
    };

    result.push_str("┌─────────────────────────────────────────┐\n");
    result.push_str(&format!(
        "│  VQC: {} ({} qubits, {} layers)  \n",
        vqc_type_name, vqc.num_qubits, vqc.num_layers
    ));
    result.push_str("├─────────────────────────────────────────┤\n");

    // Show input encoding
    if vqc.include_input_encoding {
        result.push_str("│  Input Encoding:                        │\n");
        for q in 0..vqc.num_qubits {
            result.push_str(&format!("│    q{}: ──RY(πx{})──                    │\n", q, q));
        }
        result.push_str("├─────────────────────────────────────────┤\n");
    }

    // Show variational layers
    for layer in 0..vqc.num_layers {
        result.push_str(&format!("│  Layer {}:                              │\n", layer));
        match vqc.vqc_type {
            VQCType::Simple => {
                for q in 0..vqc.num_qubits {
                    result.push_str(&format!("│    q{}: ──RY(θ)──                       │\n", q));
                }
            }
            VQCType::BasicEntangling | VQCType::Custom => {
                for q in 0..vqc.num_qubits {
                    result.push_str(&format!(
                        "│    q{}: ──RY(θ)──RZ(θ)──                │\n",
                        q
                    ));
                }
                result.push_str("│    Entangling: CNOT ladder              │\n");
            }
            VQCType::StronglyEntangling => {
                for q in 0..vqc.num_qubits {
                    result.push_str(&format!(
                        "│    q{}: ──RZ(θ)──RY(θ)──RZ(θ)──        │\n",
                        q
                    ));
                }
                result.push_str("│    Entangling: CNOT chain (rotated)    │\n");
            }
        }
        if layer < vqc.num_layers - 1 {
            result.push_str("├─────────────────────────────────────────┤\n");
        }
    }

    result.push_str("├─────────────────────────────────────────┤\n");
    result.push_str("│  Measurement: Z expectation per qubit   │\n");
    result.push_str(&format!(
        "│  Parameters: {}                          │\n",
        vqc.num_variational_params()
    ));
    result.push_str("└─────────────────────────────────────────┘\n");

    result
}

/// Generate an SVG visualization of the QLSTM cell architecture
/// Based on Figure 5 from "Quantum Long Short-Term Memory" (arXiv:2009.01783)
pub fn svg_diagram(config: &QLSTMConfig) -> String {
    let width = 750;
    let height = 580;

    let vqc_type_name = match config.vqc_type {
        VQCType::Simple => "Simple",
        VQCType::BasicEntangling => "BasicEntangling",
        VQCType::StronglyEntangling => "StronglyEntangling",
        VQCType::Custom => "Custom",
    };

    let mut svg = format!(
        r##"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#333"/>
            </marker>
        </defs>
        <style>
            .title {{ font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; }}
            .vqc-box {{ fill: #4FC3F7; stroke: #0288D1; stroke-width: 2; }}
            .input-box {{ fill: #81C784; stroke: #388E3C; stroke-width: 2; }}
            .output-box {{ fill: #81C784; stroke: #388E3C; stroke-width: 2; }}
            .postproc-box {{ fill: #4FC3F7; stroke: #0288D1; stroke-width: 2; }}
            .op-circle {{ fill: #F8BBD9; stroke: #C2185B; stroke-width: 2; }}
            .activation {{ fill: #F8BBD9; stroke: #C2185B; stroke-width: 1.5; }}
            .label {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; dominant-baseline: middle; }}
            .label-bold {{ font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; dominant-baseline: middle; }}
            .label-italic {{ font-family: Arial, sans-serif; font-size: 11px; font-style: italic; text-anchor: middle; dominant-baseline: middle; }}
            .small-label {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; dominant-baseline: middle; }}
            .arrow {{ stroke: #333; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }}
            .line {{ stroke: #333; stroke-width: 1.5; fill: none; }}
            .cell-boundary {{ fill: #FFEB3B; fill-opacity: 0.4; stroke: #333; stroke-width: 2; stroke-dasharray: 8,4; }}
        </style>
        "##,
        width, height
    );

    // Background
    svg.push_str(r##"<rect x="0" y="0" width="750" height="580" fill="#FFFFFF"/>"##);

    // Title
    svg.push_str(&format!(
        r##"<text x="{}" y="20" class="title">QLSTM Cell (arXiv:2009.01783)</text>"##,
        width / 2
    ));

    // ============ YELLOW DASHED CELL BOUNDARY ============
    svg.push_str(r##"<rect x="100" y="120" width="520" height="390" rx="10" class="cell-boundary"/>"##);

    // ============ LEFT SIDE: Inputs (Green boxes) ============
    // c_{t-1} input (top left)
    svg.push_str(r##"<rect x="20" y="175" width="55" height="30" rx="8" class="input-box"/>"##);
    svg.push_str(r##"<text x="47" y="190" class="label-italic">c_{t-1}</text>"##);

    // h_{t-1} input (middle left)  
    svg.push_str(r##"<rect x="20" y="380" width="55" height="30" rx="8" class="input-box"/>"##);
    svg.push_str(r##"<text x="47" y="395" class="label-italic">h_{t-1}</text>"##);

    // x_t input (bottom left)
    svg.push_str(r##"<rect x="20" y="455" width="55" height="30" rx="8" class="input-box"/>"##);
    svg.push_str(r##"<text x="47" y="470" class="label-italic">x_t</text>"##);

    // ============ RIGHT SIDE: Outputs (Green boxes) ============
    // c_t output
    svg.push_str(r##"<rect x="645" y="175" width="55" height="30" rx="8" class="output-box"/>"##);
    svg.push_str(r##"<text x="672" y="190" class="label-italic">c_t</text>"##);

    // h_t output
    svg.push_str(r##"<rect x="645" y="310" width="55" height="30" rx="8" class="output-box"/>"##);
    svg.push_str(r##"<text x="672" y="325" class="label-italic">h_t</text>"##);

    // y_t output
    svg.push_str(r##"<rect x="645" y="70" width="55" height="30" rx="8" class="output-box"/>"##);
    svg.push_str(r##"<text x="672" y="85" class="label-italic">y_t</text>"##);

    // ============ VQC BLOCKS (Blue boxes) ============
    // VQC₁ - Forget gate
    svg.push_str(r##"<rect x="133" y="420" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="160" y="435" class="label-bold">VQC₁</text>"##);

    // VQC₂ - Input gate
    svg.push_str(r##"<rect x="213" y="420" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="240" y="435" class="label-bold">VQC₂</text>"##);

    // VQC₃ - Cell gate
    svg.push_str(r##"<rect x="293" y="420" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="320" y="435" class="label-bold">VQC₃</text>"##);

    // VQC₄ - Output gate
    svg.push_str(r##"<rect x="373" y="420" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="400" y="435" class="label-bold">VQC₄</text>"##);

    // VQC₅ - h_t processing (positioned to receive from third ⊗)
    svg.push_str(r##"<rect x="533" y="310" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="560" y="325" class="label-bold">VQC₅</text>"##);

    // VQC₆ - y_t processing
    svg.push_str(r##"<rect x="533" y="130" width="55" height="30" rx="5" class="vqc-box"/>"##);
    svg.push_str(r##"<text x="560" y="145" class="label-bold">VQC₆</text>"##);

    // ============ ACTIVATION FUNCTIONS (Pink circles) ============
    // σ after VQC₁ (Forget)
    svg.push_str(r##"<circle cx="160" cy="375" r="15" class="activation"/>"##);
    svg.push_str(r##"<text x="160" y="375" class="label">σ</text>"##);
    svg.push_str(r##"<line x1="160" y1="420" x2="160" y2="390" class="arrow"/>"##);

    // σ after VQC₂ (Input)
    svg.push_str(r##"<circle cx="240" cy="375" r="15" class="activation"/>"##);
    svg.push_str(r##"<text x="240" y="375" class="label">σ</text>"##);
    svg.push_str(r##"<line x1="240" y1="420" x2="240" y2="390" class="arrow"/>"##);

    // tanh after VQC₃ (Cell)
    svg.push_str(r##"<circle cx="320" cy="375" r="15" class="activation"/>"##);
    svg.push_str(r##"<text x="320" y="375" class="small-label">tanh</text>"##);
    svg.push_str(r##"<line x1="320" y1="420" x2="320" y2="390" class="arrow"/>"##);

    // σ after VQC₄ (Output)
    svg.push_str(r##"<circle cx="400" cy="375" r="15" class="activation"/>"##);
    svg.push_str(r##"<text x="400" y="375" class="label">σ</text>"##);
    svg.push_str(r##"<line x1="400" y1="420" x2="400" y2="390" class="arrow"/>"##);

    // tanh for c_t → tanh(c_t)
    svg.push_str(r##"<circle cx="450" cy="250" r="12" class="activation"/>"##);
    svg.push_str(r##"<text x="450" y="250" class="small-label">tanh</text>"##);

    // ============ MULTIPLICATION OPERATIONS (⊗) ============
    // First ⊗: c_{t-1} × f_t (forget gate operation)
    svg.push_str(r##"<circle cx="160" cy="190" r="15" class="op-circle"/>"##);
    svg.push_str(r##"<text x="160" y="190" class="label-bold">⊗</text>"##);
    
    // c_{t-1} to first ⊗
    svg.push_str(r##"<line x1="75" y1="190" x2="145" y2="190" class="arrow"/>"##);
    // f_t (σ output) to first ⊗
    svg.push_str(r##"<line x1="160" y1="360" x2="160" y2="205" class="arrow"/>"##);

    // Second ⊗: i_t × c̃_t (input × cell candidate)
    svg.push_str(r##"<circle cx="280" cy="310" r="15" class="op-circle"/>"##);
    svg.push_str(r##"<text x="280" y="310" class="label-bold">⊗</text>"##);
    
    // i_t to second ⊗
    svg.push_str(r##"<path d="M 240 360 L 240 310 L 265 310" class="arrow"/>"##);
    // c̃_t to second ⊗
    svg.push_str(r##"<path d="M 320 360 L 320 310 L 295 310" class="arrow"/>"##);

    // Third ⊗: o_t × tanh(c_t) (output gate operation)
    svg.push_str(r##"<circle cx="480" cy="325" r="15" class="op-circle"/>"##);
    svg.push_str(r##"<text x="480" y="325" class="label-bold">⊗</text>"##);
    
    // o_t to third ⊗
    svg.push_str(r##"<path d="M 400 360 L 400 325 L 465 325" class="arrow"/>"##);
    // tanh(c_t) to third ⊗
    svg.push_str(r##"<line x1="462" y1="250" x2="480" y2="310" class="arrow"/>"##);

    // ============ ADDITION OPERATION (⊕) ============
    svg.push_str(r##"<circle cx="220" cy="190" r="15" class="op-circle"/>"##);
    svg.push_str(r##"<text x="220" y="190" class="label-bold">⊕</text>"##);
    
    // From first ⊗ to ⊕
    svg.push_str(r##"<line x1="175" y1="190" x2="205" y2="190" class="arrow"/>"##);
    // From second ⊗ to ⊕
    svg.push_str(r##"<path d="M 280 295 L 280 250 L 220 250 L 220 205" class="arrow"/>"##);

    // ============ CONNECTIONS ============
    // ⊕ output to c_t
    svg.push_str(r##"<line x1="235" y1="190" x2="645" y2="190" class="arrow"/>"##);
    
    // Branch from c_t line to tanh(c_t)
    svg.push_str(r##"<path d="M 380 190 L 380 250 L 438 250" class="arrow"/>"##);

    // Third ⊗ output to VQC₅
    svg.push_str(r##"<line x1="495" y1="325" x2="533" y2="325" class="arrow"/>"##);

    // VQC₅ output to h_t
    svg.push_str(r##"<line x1="588" y1="325" x2="645" y2="325" class="arrow"/>"##);

    // h_t feedback to VQC₆ (goes up and left)
    svg.push_str(r##"<path d="M 672 310 L 672 220 L 560 220 L 560 160" class="arrow"/>"##);

    // VQC₆ to y_t
    svg.push_str(r##"<line x1="560" y1="130" x2="560" y2="100" class="line"/>"##);
    svg.push_str(r##"<line x1="560" y1="100" x2="645" y2="85" class="arrow"/>"##);

    // ============ INPUT CONNECTIONS TO VQCs ============
    // Combine x_t and h_{t-1} at bottom, then feed to all 4 VQCs
    // Horizontal line at y=490 connecting all inputs
    svg.push_str(r##"<path d="M 47 485 L 47 490 L 400 490" class="line"/>"##);
    svg.push_str(r##"<path d="M 47 410 L 47 490" class="line"/>"##);
    
    // Vertical arrows up to each VQC
    svg.push_str(r##"<line x1="160" y1="490" x2="160" y2="450" class="arrow"/>"##);
    svg.push_str(r##"<line x1="240" y1="490" x2="240" y2="450" class="arrow"/>"##);
    svg.push_str(r##"<line x1="320" y1="490" x2="320" y2="450" class="arrow"/>"##);
    svg.push_str(r##"<line x1="400" y1="490" x2="400" y2="450" class="arrow"/>"##);

    // ============ CLASSICAL POST-PROCESSING (Optional) ============
    svg.push_str(r##"<rect x="430" y="40" width="90" height="40" rx="5" class="postproc-box"/>"##);
    svg.push_str(r##"<text x="475" y="55" class="small-label">Classical</text>"##);
    svg.push_str(r##"<text x="475" y="68" class="small-label">Post-processing</text>"##);
    svg.push_str(r##"<text x="475" y="90" class="small-label" fill="#666">(Optional)</text>"##);
    
    // Arrow from post-processing to y_t
    svg.push_str(r##"<line x1="520" y1="60" x2="645" y2="75" class="arrow"/>"##);

    // ============ CONFIGURATION INFO ============
    svg.push_str(&format!(
        r##"<text x="375" y="555" class="small-label" fill="#666">Config: input={}, hidden={}, qubits={}, layers={}, type={}</text>"##,
        config.input_size, config.hidden_size, config.num_qubits, config.num_layers, vqc_type_name
    ));

    svg.push_str("</svg>");
    svg
}

/// Generate an SVG visualization of a single VQC
pub fn vqc_svg_diagram(vqc: &VariationalQuantumCircuit) -> String {
    let cell_width = 50;
    let cell_height = 40;
    let margin = 60;
    let num_cols = vqc.num_layers * 3 + (if vqc.include_input_encoding { 2 } else { 1 });
    let width = num_cols * cell_width + 2 * margin;
    let height = vqc.num_qubits * cell_height + 2 * margin + 40;

    let vqc_type_name = match vqc.vqc_type {
        VQCType::Simple => "Simple",
        VQCType::BasicEntangling => "BasicEntangling",
        VQCType::StronglyEntangling => "StronglyEntangling",
        VQCType::Custom => "Custom",
    };

    let mut svg = format!(
        r##"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .title {{ font-family: 'Courier New', monospace; font-size: 14px; font-weight: bold; text-anchor: middle; }}
            .qubit-line {{ stroke: #000; stroke-width: 1; }}
            .gate-box {{ fill: #E3F2FD; stroke: #1976D2; stroke-width: 1.5; rx: 3; }}
            .entangle-box {{ fill: #E8F5E9; stroke: #4CAF50; stroke-width: 1.5; rx: 3; }}
            .gate-text {{ font-family: 'Courier New', monospace; font-size: 10px; text-anchor: middle; dominant-baseline: middle; }}
            .qubit-label {{ font-family: 'Courier New', monospace; font-size: 12px; text-anchor: end; dominant-baseline: middle; }}
            .control-point {{ fill: #000; }}
            .connector-line {{ stroke: #000; stroke-width: 1; }}
        </style>
        "##,
        width, height
    );

    // Title
    svg.push_str(&format!(
        r##"<text x="{}" y="20" class="title">VQC: {} ({} qubits, {} layers)</text>"##,
        width / 2, vqc_type_name, vqc.num_qubits, vqc.num_layers
    ));

    // Draw qubit lines
    for q in 0..vqc.num_qubits {
        let y = q * cell_height + margin;

        // Qubit label
        svg.push_str(&format!(
            r##"<text x="{}" y="{}" class="qubit-label">q{}</text>"##,
            margin - 10, y, q
        ));

        // Qubit line
        svg.push_str(&format!(
            r##"<line x1="{}" y1="{}" x2="{}" y2="{}" class="qubit-line"/>"##,
            margin, y, width - margin, y
        ));
    }

    let mut col = 0;

    // Input encoding
    if vqc.include_input_encoding {
        for q in 0..vqc.num_qubits {
            let x = col * cell_width + margin + cell_width / 2;
            let y = q * cell_height + margin;

            svg.push_str(&format!(
                r##"<rect x="{}" y="{}" width="40" height="25" class="gate-box"/>"##,
                x - 20, y - 12
            ));
            svg.push_str(&format!(
                r##"<text x="{}" y="{}" class="gate-text">RY(x{})</text>"##,
                x, y, q
            ));
        }
        col += 1;
    }

    // Variational layers
    for layer in 0..vqc.num_layers {
        // Rotation gates
        match vqc.vqc_type {
            VQCType::Simple => {
                let x = col * cell_width + margin + cell_width / 2;
                for q in 0..vqc.num_qubits {
                    let y = q * cell_height + margin;
                    svg.push_str(&format!(
                        r##"<rect x="{}" y="{}" width="30" height="20" class="gate-box"/>"##,
                        x - 15, y - 10
                    ));
                    svg.push_str(&format!(
                        r##"<text x="{}" y="{}" class="gate-text">RY</text>"##,
                        x, y
                    ));
                }
                col += 1;
            }
            VQCType::BasicEntangling | VQCType::Custom => {
                // RY gates
                let x = col * cell_width + margin + cell_width / 2;
                for q in 0..vqc.num_qubits {
                    let y = q * cell_height + margin;
                    svg.push_str(&format!(
                        r##"<rect x="{}" y="{}" width="30" height="20" class="gate-box"/>"##,
                        x - 15, y - 10
                    ));
                    svg.push_str(&format!(
                        r##"<text x="{}" y="{}" class="gate-text">RY</text>"##,
                        x, y
                    ));
                }
                col += 1;

                // RZ gates
                let x = col * cell_width + margin + cell_width / 2;
                for q in 0..vqc.num_qubits {
                    let y = q * cell_height + margin;
                    svg.push_str(&format!(
                        r##"<rect x="{}" y="{}" width="30" height="20" class="gate-box"/>"##,
                        x - 15, y - 10
                    ));
                    svg.push_str(&format!(
                        r##"<text x="{}" y="{}" class="gate-text">RZ</text>"##,
                        x, y
                    ));
                }
                col += 1;

                // CNOT ladder
                let x = col * cell_width + margin + cell_width / 2;
                for q in 0..vqc.num_qubits.saturating_sub(1) {
                    let control_y = q * cell_height + margin;
                    let target_y = (q + 1) * cell_height + margin;

                    svg.push_str(&format!(
                        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line"/>"##,
                        x, control_y, x, target_y
                    ));
                    svg.push_str(&format!(
                        r##"<circle cx="{}" cy="{}" r="4" class="control-point"/>"##,
                        x, control_y
                    ));
                    svg.push_str(&format!(
                        r##"<circle cx="{}" cy="{}" r="8" fill="none" stroke="#000" stroke-width="1.5"/>"##,
                        x, target_y
                    ));
                    svg.push_str(&format!(
                        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line"/>"##,
                        x - 8, target_y, x + 8, target_y
                    ));
                    svg.push_str(&format!(
                        r##"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line"/>"##,
                        x, target_y - 8, x, target_y + 8
                    ));
                }
                col += 1;
            }
            VQCType::StronglyEntangling => {
                // RZ-RY-RZ gates
                for gate in ["RZ", "RY", "RZ"] {
                    let x = col * cell_width + margin + cell_width / 2;
                    for q in 0..vqc.num_qubits {
                        let y = q * cell_height + margin;
                        svg.push_str(&format!(
                            r##"<rect x="{}" y="{}" width="30" height="20" class="gate-box"/>"##,
                            x - 15, y - 10
                        ));
                        svg.push_str(&format!(
                            r##"<text x="{}" y="{}" class="gate-text">{}</text>"##,
                            x, y, gate
                        ));
                    }
                    col += 1;
                }

                // CNOT chain with offset
                let x = col * cell_width + margin + cell_width / 2;
                let offset = layer % vqc.num_qubits;
                for i in 0..vqc.num_qubits.saturating_sub(1) {
                    let control = (i + offset) % vqc.num_qubits;
                    let target = (i + 1 + offset) % vqc.num_qubits;
                    if control != target {
                        let control_y = control * cell_height + margin;
                        let target_y = target * cell_height + margin;
                        let (min_y, max_y) = if control_y < target_y {
                            (control_y, target_y)
                        } else {
                            (target_y, control_y)
                        };

                        svg.push_str(&format!(
                            r##"<line x1="{}" y1="{}" x2="{}" y2="{}" class="connector-line"/>"##,
                            x, min_y, x, max_y
                        ));
                        svg.push_str(&format!(
                            r##"<circle cx="{}" cy="{}" r="4" class="control-point"/>"##,
                            x, control_y
                        ));
                        svg.push_str(&format!(
                            r##"<circle cx="{}" cy="{}" r="8" fill="none" stroke="#000" stroke-width="1.5"/>"##,
                            x, target_y
                        ));
                    }
                }
                col += 1;
            }
        }
    }

    // Measurement symbols
    let x = col * cell_width + margin + cell_width / 2;
    for q in 0..vqc.num_qubits {
        let y = q * cell_height + margin;
        svg.push_str(&format!(
            r##"<rect x="{}" y="{}" width="30" height="20" fill="#FFF3E0" stroke="#FF9800" stroke-width="1.5" rx="3"/>"##,
            x - 15, y - 10
        ));
        svg.push_str(&format!(
            r##"<text x="{}" y="{}" class="gate-text">⟨Z⟩</text>"##,
            x, y
        ));
    }

    // Info text
    svg.push_str(&format!(
        r##"<text x="{}" y="{}" class="gate-text" style="font-size: 11px;">Parameters: {}</text>"##,
        width / 2, height - 10, vqc.num_variational_params()
    ));

    svg.push_str("</svg>");
    svg
}

/// Save QLSTM diagram to SVG file
pub fn save_svg_diagram(config: &QLSTMConfig, filename: &str) -> Result<()> {
    let svg = svg_diagram(config);
    std::fs::write(filename, svg)
}

/// Save VQC diagram to SVG file
pub fn save_vqc_svg_diagram(vqc: &VariationalQuantumCircuit, filename: &str) -> Result<()> {
    let svg = vqc_svg_diagram(vqc);
    std::fs::write(filename, svg)
}

/// View QLSTM diagram in browser
pub fn view_qlstm(config: &QLSTMConfig) -> Result<()> {
    let svg = svg_diagram(config);

    let html = format!(
        r##"<!DOCTYPE html>
        <html>
        <head>
            <title>QLSTM Visualization</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                pre {{ background: #f0f0f0; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>QLSTM Cell Visualization</h1>
                {}
                <h2>Text Representation</h2>
                <pre>{}</pre>
            </div>
        </body>
        </html>"##,
        svg,
        text_diagram(config).replace('<', "&lt;").replace('>', "&gt;")
    );

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("qlstm_visualization.html");
    std::fs::write(&file_path, html)?;

    if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/C", "start", file_path.to_str().unwrap()])
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

// Implement Visualizable trait for QLSTM
impl crate::vis::Visualizable for QLSTM {
    fn visualize(&self) -> String {
        text_diagram(self.config())
    }

    fn visualize_as_svg(&self) -> String {
        svg_diagram(self.config())
    }

    fn save_visualization(&self, filename: &str) -> Result<()> {
        save_svg_diagram(self.config(), filename)
    }

    fn view(&self) -> Result<()> {
        view_qlstm(self.config())
    }
}

// Implement Visualizable trait for VariationalQuantumCircuit
impl crate::vis::Visualizable for VariationalQuantumCircuit {
    fn visualize(&self) -> String {
        vqc_text_diagram(self)
    }

    fn visualize_as_svg(&self) -> String {
        vqc_svg_diagram(self)
    }

    fn save_visualization(&self, filename: &str) -> Result<()> {
        save_vqc_svg_diagram(self, filename)
    }

    fn view(&self) -> Result<()> {
        let svg = vqc_svg_diagram(self);
        let html = format!(
            r##"<!DOCTYPE html>
            <html>
            <head><title>VQC Visualization</title></head>
            <body style="font-family: sans-serif; margin: 20px;">
                <h1>Variational Quantum Circuit</h1>
                {}
            </body>
            </html>"##,
            svg
        );

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("vqc_visualization.html");
        std::fs::write(&file_path, html)?;

        if cfg!(target_os = "windows") {
            std::process::Command::new("cmd")
                .args(["/C", "start", file_path.to_str().unwrap()])
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
}
