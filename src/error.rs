//! Error types for RustyAI

use thiserror::Error;

/// Result type alias for RustyAI operations
pub type Result<T> = std::result::Result<T, RustyAIError>;

/// Main error type for RustyAI
#[derive(Error, Debug)]
pub enum RustyAIError {
    /// Invalid shape for an operation
    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid parameter
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Model not fitted
    #[error("Model has not been fitted yet. Call fit() first.")]
    NotFitted,

    /// Convergence error
    #[error("Algorithm failed to converge after {iterations} iterations")]
    ConvergenceError { iterations: usize },

    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Singular matrix error
    #[error("Matrix is singular and cannot be inverted")]
    SingularMatrix,

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid data error
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Empty data error
    #[error("Cannot operate on empty data")]
    EmptyData,

    /// Index out of bounds
    #[error("Index {index} out of bounds for dimension of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Feature mismatch
    #[error("Feature count mismatch: expected {expected}, got {got}")]
    FeatureMismatch { expected: usize, got: usize },

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<String> for RustyAIError {
    fn from(s: String) -> Self {
        RustyAIError::Other(s)
    }
}

impl From<&str> for RustyAIError {
    fn from(s: &str) -> Self {
        RustyAIError::Other(s.to_string())
    }
}

impl From<ndarray::ShapeError> for RustyAIError {
    fn from(e: ndarray::ShapeError) -> Self {
        RustyAIError::InvalidShape {
            expected: "valid shape".to_string(),
            got: e.to_string(),
        }
    }
}
