//! Matrix decomposition and dimensionality reduction
//!
//! Comprehensive dimensionality reduction methods:
//! - PCA (Principal Component Analysis)
//! - t-SNE (t-Distributed Stochastic Neighbor Embedding)
//! - UMAP (Uniform Manifold Approximation and Projection)

pub mod advanced;

pub use advanced::{PCA, TSNE, UMAP};
