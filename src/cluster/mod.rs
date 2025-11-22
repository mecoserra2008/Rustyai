//! Clustering algorithms
//!
//! Includes K-Means, DBSCAN, hierarchical clustering, and Gaussian Mixture Models.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// K-Means clustering algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans<A> {
    n_clusters: usize,
    _phantom: std::marker::PhantomData<A>,
}

/// DBSCAN clustering algorithm
#[derive(Debug, Clone)]
pub struct DBSCAN<A> {
    eps: A,
    min_samples: usize,
}
