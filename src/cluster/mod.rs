//! Clustering algorithms
//!
//! Comprehensive clustering methods:
//! - K-Means: Partitioning clustering
//! - DBSCAN: Density-based clustering
//! - Hierarchical: Bottom-up/top-down clustering
//! - Spectral: Graph-based clustering

pub mod kmeans;
pub mod advanced;

pub use kmeans::KMeans;
pub use advanced::{
    dbscan, hierarchical_clustering, spectral_clustering, mean_shift,
    DBSCANResult, HierarchicalResult, Linkage,
};
