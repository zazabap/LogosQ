//! Visualization module for quantum components including circuits, states, gates and noise models.

pub mod circuit;

// Placeholder modules for future implementation
pub mod gate;
pub mod state;
// pub mod noise;

// Re-export main visualization functions
pub use circuit::{
    save_svg_diagram as save_circuit_svg, svg_diagram as circuit_svg, text_diagram as circuit_text,
    view_circuit,
};

// Main visualization function that determines the type and renders accordingly
pub fn visualize<T: Visualizable>(item: &T) -> String {
    item.visualize()
}

// Trait for types that can be visualized
pub trait Visualizable {
    fn visualize(&self) -> String;
    fn visualize_as_svg(&self) -> String;
    fn save_visualization(&self, filename: &str) -> std::io::Result<()>;
    fn view(&self) -> std::io::Result<()>;
}
