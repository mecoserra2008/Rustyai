//! K-Means Clustering
//!
//! The most fundamental clustering algorithm that partitions data
//! into K clusters based on distance to cluster centroids.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// K-Means clustering algorithm
///
/// Partitions n observations into k clusters where each observation
/// belongs to the cluster with the nearest mean (centroid).
///
/// # Algorithm
/// 1. Initialize K centroids (using K-Means++)
/// 2. Assignment: Assign each point to nearest centroid
/// 3. Update: Recalculate centroids as mean of assigned points
/// 4. Repeat 2-3 until convergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans<A: Float> {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: A,
    /// Number of initializations (to avoid local minima)
    pub n_init: usize,
    /// Cluster centroids (k x n_features)
    pub cluster_centers: Option<Array2<A>>,
    /// Labels for training data
    pub labels: Option<Array1<usize>>,
    /// Inertia (sum of squared distances to closest centroid)
    pub inertia: Option<A>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> KMeans<A> {
    /// Create a new K-Means model
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: A::from(1e-4).unwrap(),
            n_init: 10,
            cluster_centers: None,
            labels: None,
            inertia: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: A) -> Self {
        self.tol = tol;
        self
    }

    /// Set number of initializations
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Fit K-Means on training data
    pub fn fit(&mut self, X: &Array2<A>) -> Result<()> {
        let (n_samples, _n_features) = X.dim();

        if n_samples < self.n_clusters {
            return Err(RustyAIError::InvalidParameter {
                name: "X".to_string(),
                reason: format!("n_samples={} should be >= n_clusters={}", n_samples, self.n_clusters),
            });
        }

        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_inertia = A::infinity();

        // Multiple initializations to avoid local minima
        for _ in 0..self.n_init {
            let (centers, labels, inertia) = self.fit_once(X)?;

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels);
            }
        }

        self.cluster_centers = best_centers;
        self.labels = best_labels;
        self.inertia = Some(best_inertia);

        Ok(())
    }

    /// Single K-Means run
    fn fit_once(&self, X: &Array2<A>) -> Result<(Array2<A>, Array1<usize>, A)> {
        let (n_samples, n_features) = X.dim();

        // Initialize centroids using K-Means++
        let mut centers = self.kmeans_plus_plus_init(X)?;
        let mut labels = Array1::zeros(n_samples);
        let mut prev_inertia = A::infinity();

        for _iteration in 0..self.max_iter {
            // Assignment step: assign each point to nearest centroid
            let mut inertia = A::zero();

            for i in 0..n_samples {
                let point = X.row(i);
                let (cluster, dist_sq) = self.nearest_centroid(&point, &centers);
                labels[i] = cluster;
                inertia = inertia + dist_sq;
            }

            // Check convergence
            if (prev_inertia - inertia).abs() < self.tol {
                break;
            }
            prev_inertia = inertia;

            // Update step: recompute centroids
            for k in 0..self.n_clusters {
                let cluster_points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == k)
                    .map(|(idx, _)| idx)
                    .collect();

                if !cluster_points.is_empty() {
                    for j in 0..n_features {
                        let sum: A = cluster_points.iter().map(|&idx| X[[idx, j]]).sum();
                        centers[[k, j]] = sum / A::from(cluster_points.len()).unwrap();
                    }
                }
            }
        }

        let final_inertia = self.compute_inertia(X, &labels, &centers);
        Ok((centers, labels, final_inertia))
    }

    /// K-Means++ initialization (smart centroid initialization)
    ///
    /// Chooses initial centroids to be far from each other,
    /// significantly improving convergence and final quality.
    fn kmeans_plus_plus_init(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let (n_samples, n_features) = X.dim();
        let mut centers = Array2::zeros((self.n_clusters, n_features));

        // Choose first center uniformly at random
        let first_idx = (rand::random::<f64>() * n_samples as f64) as usize % n_samples;
        for j in 0..n_features {
            centers[[0, j]] = X[[first_idx, j]];
        }

        // Choose remaining centers with probability proportional to distance²
        for k in 1..self.n_clusters {
            let mut distances = Vec::with_capacity(n_samples);
            let mut total_dist = A::zero();

            for i in 0..n_samples {
                let point = X.row(i);
                let current_centers = centers.slice(ndarray::s![..k, ..]).to_owned();
                let (_, min_dist_sq) = self.nearest_centroid(&point, &current_centers);
                distances.push(min_dist_sq);
                total_dist = total_dist + min_dist_sq;
            }

            // Weighted random sampling
            let mut cumsum = A::zero();
            let target = A::from(rand::random::<f64>()).unwrap() * total_dist;

            let mut chosen_idx = 0;
            for (i, &dist) in distances.iter().enumerate() {
                cumsum = cumsum + dist;
                if cumsum >= target {
                    chosen_idx = i;
                    break;
                }
            }

            for j in 0..n_features {
                centers[[k, j]] = X[[chosen_idx, j]];
            }
        }

        Ok(centers)
    }

    /// Find nearest centroid for a point
    fn nearest_centroid(&self, point: &ndarray::ArrayView1<A>, centers: &Array2<A>) -> (usize, A) {
        let mut min_dist_sq = A::infinity();
        let mut nearest = 0;

        for k in 0..centers.nrows() {
            let centroid = centers.row(k);
            let dist_sq: A = point.iter().zip(centroid.iter())
                .map(|(&p, &c)| (p - c) * (p - c))
                .sum();

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                nearest = k;
            }
        }

        (nearest, min_dist_sq)
    }

    /// Compute inertia (sum of squared distances to nearest centroids)
    fn compute_inertia(&self, X: &Array2<A>, labels: &Array1<usize>, centers: &Array2<A>) -> A {
        let mut inertia = A::zero();

        for (i, &label) in labels.iter().enumerate() {
            let point = X.row(i);
            let centroid = centers.row(label);
            let dist_sq: A = point.iter().zip(centroid.iter())
                .map(|(&p, &c)| (p - c) * (p - c))
                .sum();
            inertia = inertia + dist_sq;
        }

        inertia
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<usize>> {
        let centers = self.cluster_centers.as_ref()
            .ok_or_else(|| RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = X.row(i);
            let (cluster, _) = self.nearest_centroid(&point, centers);
            labels[i] = cluster;
        }

        Ok(labels)
    }

    /// Fit and predict in one step
    pub fn fit_predict(&mut self, X: &Array2<A>) -> Result<Array1<usize>> {
        self.fit(X)?;
        Ok(self.labels.clone().unwrap())
    }

    /// Transform X to cluster-distance space
    ///
    /// Returns distances to each cluster center.
    pub fn transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let centers = self.cluster_centers.as_ref()
            .ok_or_else(|| RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let mut distances = Array2::zeros((n_samples, self.n_clusters));

        for i in 0..n_samples {
            let point = X.row(i);
            for k in 0..self.n_clusters {
                let centroid = centers.row(k);
                let dist: A = point.iter().zip(centroid.iter())
                    .map(|(&p, &c)| (p - c) * (p - c))
                    .sum::<A>()
                    .sqrt();
                distances[[i, k]] = dist;
            }
        }

        Ok(distances)
    }

    /// Get cluster centers
    pub fn get_cluster_centers(&self) -> Option<&Array2<A>> {
        self.cluster_centers.as_ref()
    }

    /// Get inertia
    pub fn get_inertia(&self) -> Option<A> {
        self.inertia
    }
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> Default for KMeans<A> {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmeans_basic() {
        // Create simple 2D dataset with clear clusters
        let X = array![
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
        ];

        let mut kmeans = KMeans::new(2).with_n_init(3).with_max_iter(100);
        kmeans.fit(&X).unwrap();

        assert_eq!(kmeans.cluster_centers.as_ref().unwrap().shape(), &[2, 2]);
        assert_eq!(kmeans.labels.as_ref().unwrap().len(), 6);
        assert!(kmeans.inertia.unwrap() > 0.0);
    }

    #[test]
    fn test_kmeans_predict() {
        let X_train = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ];

        let X_test = array![
            [0.5, 0.5],
            [10.5, 10.5],
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&X_train).unwrap();

        let predictions = kmeans.predict(&X_test).unwrap();
        assert_eq!(predictions.len(), 2);
        // Points should be in different clusters
        assert_ne!(predictions[0], predictions[1]);
    }
}
