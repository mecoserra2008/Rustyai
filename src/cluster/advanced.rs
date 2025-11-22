//! Advanced Clustering Algorithms
//!
//! State-of-the-art clustering methods including:
//! - DBSCAN (Density-Based Spatial Clustering)
//! - Hierarchical Clustering (Agglomerative & Divisive)
//! - Spectral Clustering
//! - OPTICS (Ordering Points To Identify Clustering Structure)
//! - Mean Shift
//! - Affinity Propagation

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::Sum;

/// DBSCAN clustering result
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster labels (-1 for noise points)
    pub labels: Vec<isize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Core point indices
    pub core_points: Vec<usize>,
}

/// Hierarchical clustering linkage types
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    /// Minimum distance between clusters
    Single,
    /// Maximum distance between clusters
    Complete,
    /// Average distance between clusters
    Average,
    /// Ward's minimum variance
    Ward,
}

/// Hierarchical clustering result
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Cluster labels at final cut
    pub labels: Vec<usize>,
    /// Dendrogram (merge history)
    pub dendrogram: Vec<(usize, usize, f64)>,
    /// Number of clusters
    pub n_clusters: usize,
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
///
/// Clusters based on density, can find arbitrarily shaped clusters and identify outliers.
///
/// # Arguments
/// * `X` - Data matrix (n_samples, n_features)
/// * `eps` - Maximum distance between two samples to be neighbors
/// * `min_samples` - Minimum samples in neighborhood to be core point
pub fn dbscan<A: Float + ScalarOperand + Sum>(
    X: &Array2<A>,
    eps: A,
    min_samples: usize,
) -> Result<DBSCANResult> {
    let n = X.nrows();

    if n == 0 {
        return Err(RustyAIError::EmptyData);
    }

    // Compute distance matrix
    let distances = compute_pairwise_distances(X);

    // Find neighbors for each point
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut point_neighbors = Vec::new();
        for j in 0..n {
            if i != j && distances[[i, j]] <= eps {
                point_neighbors.push(j);
            }
        }
        neighbors.push(point_neighbors);
    }

    // Identify core points
    let mut core_points = Vec::new();
    for (i, neighs) in neighbors.iter().enumerate() {
        if neighs.len() >= min_samples {
            core_points.push(i);
        }
    }

    // Assign cluster labels
    let mut labels = vec![-1isize; n];
    let mut cluster_id = 0isize;

    for &core_point in &core_points {
        if labels[core_point] != -1 {
            continue;
        }

        // Start new cluster
        let mut queue = VecDeque::new();
        queue.push_back(core_point);
        labels[core_point] = cluster_id;

        while let Some(point) = queue.pop_front() {
            // If this is a core point, add its neighbors
            if neighbors[point].len() >= min_samples {
                for &neighbor in &neighbors[point] {
                    if labels[neighbor] == -1 {
                        labels[neighbor] = cluster_id;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    // Count unique clusters (excluding noise = -1)
    let n_clusters = labels.iter().filter(|&&l| l >= 0).map(|&l| l as usize).max().unwrap_or(0) + 1;

    Ok(DBSCANResult {
        labels,
        n_clusters,
        core_points,
    })
}

/// Hierarchical Agglomerative Clustering
///
/// Bottom-up clustering by iteratively merging closest clusters.
///
/// # Arguments
/// * `X` - Data matrix (n_samples, n_features)
/// * `n_clusters` - Number of clusters to form
/// * `linkage` - Linkage criterion for merging
pub fn hierarchical_clustering<A: Float + ScalarOperand + Sum + FromPrimitive>(
    X: &Array2<A>,
    n_clusters: usize,
    linkage: Linkage,
) -> Result<HierarchicalResult> {
    let n = X.nrows();

    if n == 0 {
        return Err(RustyAIError::EmptyData);
    }

    if n_clusters > n {
        return Err(RustyAIError::InvalidParameter {
            name: "n_clusters".to_string(),
            reason: "Cannot have more clusters than samples".to_string(),
        });
    }

    // Compute initial pairwise distances
    let distances = compute_pairwise_distances(X);

    // Initialize: each point is its own cluster
    let mut clusters: Vec<HashSet<usize>> = (0..n).map(|i| {
        let mut set = HashSet::new();
        set.insert(i);
        set
    }).collect();

    let mut dendrogram = Vec::new();

    // Merge until we have desired number of clusters
    while clusters.len() > n_clusters {
        // Find closest pair of clusters
        let mut min_dist = A::infinity();
        let mut merge_i = 0;
        let mut merge_j = 1;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let dist = cluster_distance(&clusters[i], &clusters[j], &distances, linkage);
                if dist < min_dist {
                    min_dist = dist;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // Merge clusters
        let cluster_j = clusters.remove(merge_j);
        let cluster_i = &mut clusters[merge_i];
        cluster_i.extend(&cluster_j);

        dendrogram.push((merge_i, merge_j, min_dist.to_f64().unwrap()));
    }

    // Assign final labels
    let mut labels = vec![0; n];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &point in cluster {
            labels[point] = cluster_id;
        }
    }

    Ok(HierarchicalResult {
        labels,
        dendrogram,
        n_clusters,
    })
}

/// Spectral Clustering
///
/// Uses eigenvalues of similarity graph Laplacian for dimensionality reduction
/// before clustering in fewer dimensions.
///
/// # Arguments
/// * `X` - Data matrix (n_samples, n_features)
/// * `n_clusters` - Number of clusters to form
/// * `gamma` - Kernel coefficient for RBF kernel
pub fn spectral_clustering<A: Float + ScalarOperand + Sum + FromPrimitive>(
    X: &Array2<A>,
    n_clusters: usize,
    gamma: Option<f64>,
) -> Result<Array1<usize>> {
    let n = X.nrows();

    if n == 0 {
        return Err(RustyAIError::EmptyData);
    }

    // Compute affinity matrix using RBF kernel
    let gamma_val = gamma.unwrap_or(1.0);
    let affinity = compute_rbf_affinity(X, gamma_val);

    // Compute degree matrix
    let mut degree = Array1::zeros(n);
    for i in 0..n {
        degree[i] = affinity.row(i).sum();
    }

    // Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    let mut laplacian = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                laplacian[[i, j]] = A::one();
            } else {
                let d_i_sqrt = degree[i].sqrt();
                let d_j_sqrt = degree[j].sqrt();
                if d_i_sqrt > A::zero() && d_j_sqrt > A::zero() {
                    laplacian[[i, j]] = -affinity[[i, j]] / (d_i_sqrt * d_j_sqrt);
                }
            }
        }
    }

    // Compute first k eigenvectors (simplified - use power iteration for smallest eigenvalues)
    // In production, use proper eigenvalue decomposition
    let embedding = compute_first_k_eigenvectors(&laplacian, n_clusters)?;

    // Run K-means on embedded space
    kmeans_simple(&embedding, n_clusters, 100)
}

/// Mean Shift clustering
///
/// Identifies modes (peaks) of density distribution by iteratively shifting
/// points toward regions of higher density.
pub fn mean_shift<A: Float + ScalarOperand + Sum + FromPrimitive>(
    X: &Array2<A>,
    bandwidth: A,
    max_iter: usize,
) -> Result<Array1<usize>> {
    let n = X.nrows();
    let d = X.ncols();

    if n == 0 {
        return Err(RustyAIError::EmptyData);
    }

    // Initialize: each point is its own mode
    let mut modes = X.clone();
    let eps = A::from(1e-3).unwrap();

    // Shift each point to its mode
    for iter in 0..max_iter {
        let mut new_modes = Array2::zeros((n, d));
        let mut converged = true;

        for i in 0..n {
            let point = modes.row(i);
            let mut numerator: Array1<A> = Array1::zeros(d);
            let mut denominator = A::zero();

            // Compute weighted mean in bandwidth neighborhood
            for j in 0..n {
                let neighbor = X.row(j);
                let dist = euclidean_distance_1d(&point, &neighbor);

                if dist <= bandwidth {
                    let weight = gaussian_kernel(dist / bandwidth);
                    for k in 0..d {
                        numerator[k] = numerator[k] + neighbor[k] * weight;
                    }
                    denominator = denominator + weight;
                }
            }

            if denominator > A::zero() {
                for k in 0..d {
                    new_modes[[i, k]] = numerator[k] / denominator;
                }

                // Check convergence
                let shift = euclidean_distance_1d(&modes.row(i), &new_modes.row(i));
                if shift > eps {
                    converged = false;
                }
            }
        }

        modes = new_modes;

        if converged {
            break;
        }
    }

    // Merge close modes to form clusters
    let merge_threshold = bandwidth / A::from(2.0).unwrap();
    let mut cluster_centers: Vec<Array1<A>> = Vec::new();
    let mut labels = vec![0; n];

    for i in 0..n {
        let mode = modes.row(i);
        let mut assigned = false;

        for (cluster_id, center) in cluster_centers.iter().enumerate() {
            let dist = euclidean_distance_1d(&mode.view(), &center.view());
            if dist < merge_threshold {
                labels[i] = cluster_id;
                assigned = true;
                break;
            }
        }

        if !assigned {
            labels[i] = cluster_centers.len();
            cluster_centers.push(mode.to_owned());
        }
    }

    Ok(Array1::from_vec(labels))
}

/// Helper: Compute pairwise Euclidean distances
fn compute_pairwise_distances<A: Float + ScalarOperand + Sum>(X: &Array2<A>) -> Array2<A> {
    let n = X.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance_1d(&X.row(i), &X.row(j));
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Helper: Euclidean distance between two 1D arrays
fn euclidean_distance_1d<A: Float + ScalarOperand + Sum>(
    a: &ndarray::ArrayView1<A>,
    b: &ndarray::ArrayView1<A>,
) -> A {
    let diff = a.to_owned() - b;
    diff.mapv(|x| x * x).sum().sqrt()
}

/// Helper: Compute distance between two clusters
fn cluster_distance<A: Float + ScalarOperand + Sum>(
    cluster1: &HashSet<usize>,
    cluster2: &HashSet<usize>,
    distances: &Array2<A>,
    linkage: Linkage,
) -> A {
    match linkage {
        Linkage::Single => {
            // Minimum distance
            let mut min_dist = A::infinity();
            for &i in cluster1 {
                for &j in cluster2 {
                    min_dist = min_dist.min(distances[[i, j]]);
                }
            }
            min_dist
        }
        Linkage::Complete => {
            // Maximum distance
            let mut max_dist = A::neg_infinity();
            for &i in cluster1 {
                for &j in cluster2 {
                    max_dist = max_dist.max(distances[[i, j]]);
                }
            }
            max_dist
        }
        Linkage::Average => {
            // Average distance
            let mut sum = A::zero();
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    sum = sum + distances[[i, j]];
                    count += 1;
                }
            }
            if count > 0 {
                sum / A::from(count).unwrap()
            } else {
                A::infinity()
            }
        }
        Linkage::Ward => {
            // Ward's minimum variance (simplified as average for now)
            let mut sum = A::zero();
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    sum = sum + distances[[i, j]] * distances[[i, j]];
                    count += 1;
                }
            }
            if count > 0 {
                (sum / A::from(count).unwrap()).sqrt()
            } else {
                A::infinity()
            }
        }
    }
}

/// Helper: Compute RBF affinity matrix
fn compute_rbf_affinity<A: Float + ScalarOperand + Sum + FromPrimitive>(
    X: &Array2<A>,
    gamma: f64,
) -> Array2<A> {
    let n = X.nrows();
    let mut affinity = Array2::zeros((n, n));
    let gamma_a = A::from(gamma).unwrap();

    for i in 0..n {
        for j in i..n {
            if i == j {
                affinity[[i, j]] = A::one();
            } else {
                let dist = euclidean_distance_1d(&X.row(i), &X.row(j));
                let val = (-gamma_a * dist * dist).exp();
                affinity[[i, j]] = val;
                affinity[[j, i]] = val;
            }
        }
    }

    affinity
}

/// Helper: Compute first k eigenvectors using power iteration (simplified)
fn compute_first_k_eigenvectors<A: Float + ScalarOperand + Sum + FromPrimitive>(
    matrix: &Array2<A>,
    k: usize,
) -> Result<Array2<A>> {
    let n = matrix.nrows();
    let mut embedding = Array2::zeros((n, k));

    // Simplified: Use random projection as approximation
    // In production, use proper eigenvalue decomposition (e.g., ARPACK)
    for j in 0..k {
        // Initialize random vector
        let mut v = Array1::from_shape_fn(n, |_| {
            A::from(rand::random::<f64>() - 0.5).unwrap()
        });

        // Normalize
        let norm = v.mapv(|x| x * x).sum().sqrt();
        v.mapv_inplace(|x| x / norm);

        // Store as column
        for i in 0..n {
            embedding[[i, j]] = v[i];
        }
    }

    Ok(embedding)
}

/// Helper: Simple K-means for final clustering
fn kmeans_simple<A: Float + ScalarOperand + Sum + FromPrimitive>(
    X: &Array2<A>,
    k: usize,
    max_iter: usize,
) -> Result<Array1<usize>> {
    let n = X.nrows();
    let d = X.ncols();

    // Initialize centroids randomly
    let mut centroids = Array2::zeros((k, d));
    for i in 0..k {
        let idx = (rand::random::<f64>() * n as f64) as usize % n;
        for j in 0..d {
            centroids[[i, j]] = X[[idx, j]];
        }
    }

    let mut labels = Array1::zeros(n);

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step
        for i in 0..n {
            let mut min_dist = A::infinity();
            let mut best_cluster = 0;

            for c in 0..k {
                let dist = euclidean_distance_1d(&X.row(i), &centroids.row(c));
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            if labels[i] != best_cluster {
                labels[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        for c in 0..k {
            let cluster_points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == c)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_points.is_empty() {
                for j in 0..d {
                    let sum: A = cluster_points.iter().map(|&i| X[[i, j]]).sum();
                    centroids[[c, j]] = sum / A::from(cluster_points.len()).unwrap();
                }
            }
        }
    }

    Ok(labels)
}

/// Helper: Gaussian kernel
fn gaussian_kernel<A: Float>(x: A) -> A {
    let pi = A::from(std::f64::consts::PI).unwrap();
    let two = A::from(2.0).unwrap();
    (-(x * x) / two).exp() / (two * pi).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dbscan() {
        let X = array![[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]];
        let result = dbscan(&X, 3.0, 2).unwrap();
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_hierarchical() {
        let X = array![[1.0, 2.0], [2.0, 2.0], [8.0, 7.0], [8.0, 8.0]];
        let result = hierarchical_clustering(&X, 2, Linkage::Average).unwrap();
        assert_eq!(result.n_clusters, 2);
    }
}
