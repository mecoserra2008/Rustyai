//! Neural network layers and modules

use crate::error::Result;

/// Dense (fully connected) layer
pub struct Dense {
    // To be implemented
}

/// Activation functions
pub mod activation {
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }
}
