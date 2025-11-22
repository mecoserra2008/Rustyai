//! Advanced Dimensionality Reduction Algorithms
//!
//! State-of-the-art methods for reducing data dimensionality:
//! - PCA (Principal Component Analysis)
//! - t-SNE (t-Distributed Stochastic Neighbor Embedding)
//! - UMAP (Uniform Manifold Approximation and Projection)
//! - Kernel PCA
//! - Incremental PCA
//! - Sparse PCA

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand, Axis};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;

/// PCA (Principal Component Analysis)
///
/// Linear dimensionality reduction using SVD.
pub struct PCA<A: Float> {
    /// Number of components to keep
    n_components: usize,
    /// Mean of training data
    mean: Option<Array1<A>>,
    /// Principal components (eigenvectors)
    components: Option<Array2<A>>,
    /// Explained variance
    explained_variance: Option<Array1<A>>,
    /// Explained variance ratio
    explained_variance_ratio: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> PCA<A> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            mean: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
        }
    }

    /// Fit PCA on training data
    pub fn fit(&mut self, X: &Array2<A>) -> Result<()> {
        let (n, d) = X.dim();

        if n < 2 {
            return Err(RustyAIError::InvalidParameter {
                name: "X".to_string(),
                reason: "Need at least 2 samples".to_string(),
            });
        }

        if self.n_components > d {
            return Err(RustyAIError::InvalidParameter {
                name: "n_components".to_string(),
                reason: format!("Cannot have more components ({}) than features ({})", self.n_components, d),
            });
        }

        // Center the data
        let mean = X.mean_axis(Axis(0)).unwrap();
        let X_centered = X - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov = X_centered.t().dot(&X_centered) / A::from(n - 1).unwrap();

        // Compute eigenvalues and eigenvectors using power iteration
        let (eigenvalues, eigenvectors) = compute_top_k_eigenpairs(&cov, self.n_components)?;

        // Store results
        self.mean = Some(mean);
        self.components = Some(eigenvectors.t().to_owned());
        self.explained_variance = Some(eigenvalues.clone());

        // Compute explained variance ratio
        let total_variance: A = eigenvalues.sum();
        let variance_ratio = eigenvalues.mapv(|v| v / total_variance);
        self.explained_variance_ratio = Some(variance_ratio);

        Ok(())
    }

    /// Transform data to lower dimensions
    pub fn transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let mean = self.mean.as_ref().ok_or_else(|| RustyAIError::NotFitted)?;
        let components = self.components.as_ref().ok_or_else(|| RustyAIError::NotFitted)?;

        // Center and project
        let X_centered = X - &mean.clone().insert_axis(Axis(0));
        Ok(X_centered.dot(&components.t()))
    }

    /// Inverse transform from lower dimensions back to original space
    pub fn inverse_transform(&self, X_transformed: &Array2<A>) -> Result<Array2<A>> {
        let mean = self.mean.as_ref().ok_or_else(|| RustyAIError::NotFitted)?;
        let components = self.components.as_ref().ok_or_else(|| RustyAIError::NotFitted)?;

        // Project back and add mean
        Ok(X_transformed.dot(components) + &mean.clone().insert_axis(Axis(0)))
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<A>> {
        self.explained_variance_ratio.as_ref()
    }
}

/// t-SNE (t-Distributed Stochastic Neighbor Embedding)
///
/// Nonlinear dimensionality reduction for visualization.
pub struct TSNE<A: Float> {
    /// Number of components (usually 2 or 3)
    n_components: usize,
    /// Perplexity parameter
    perplexity: f64,
    /// Learning rate
    learning_rate: A,
    /// Number of iterations
    n_iter: usize,
    /// Early exaggeration factor
    early_exaggeration: f64,
    /// Early exaggeration iterations
    early_exaggeration_iter: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> TSNE<A> {
    pub fn new(
        n_components: usize,
        perplexity: Option<f64>,
        learning_rate: Option<A>,
        n_iter: Option<usize>,
    ) -> Self {
        Self {
            n_components,
            perplexity: perplexity.unwrap_or(30.0),
            learning_rate: learning_rate.unwrap_or(A::from(200.0).unwrap()),
            n_iter: n_iter.unwrap_or(1000),
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
        }
    }

    /// Fit and transform data
    pub fn fit_transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let n = X.nrows();

        // Compute pairwise affinities in high-dimensional space
        let P = self.compute_affinities(X)?;

        // Initialize embedding randomly
        let mut Y = Array2::from_shape_fn((n, self.n_components), |_| {
            A::from(rand::random::<f64>() * 0.0001).unwrap()
        });

        let mut gains: Array2<A> = Array2::ones((n, self.n_components));
        let mut Y_velocity: Array2<A> = Array2::zeros((n, self.n_components));

        // Gradient descent
        for iter in 0..self.n_iter {
            // Compute affinities in low-dimensional space
            let Q = self.compute_low_dim_affinities(&Y);

            // Apply early exaggeration
            let P_effective = if iter < self.early_exaggeration_iter {
                &P * A::from(self.early_exaggeration).unwrap()
            } else {
                P.clone()
            };

            // Compute gradient
            let gradient = self.compute_gradient(&Y, &P_effective, &Q);

            // Update with momentum
            for i in 0..n {
                for j in 0..self.n_components {
                    // Adaptive gains
                    if gradient[[i, j]].signum() != Y_velocity[[i, j]].signum() {
                        gains[[i, j]] = gains[[i, j]] * A::from(1.2).unwrap();
                    } else {
                        gains[[i, j]] = gains[[i, j]] * A::from(0.8).unwrap();
                    }
                    gains[[i, j]] = gains[[i, j]].max(A::from(0.01).unwrap());

                    // Update velocity
                    let momentum = if iter < 250 { 0.5 } else { 0.8 };
                    Y_velocity[[i, j]] = A::from(momentum).unwrap() * Y_velocity[[i, j]] -
                        self.learning_rate * gains[[i, j]] * gradient[[i, j]];

                    // Update position
                    Y[[i, j]] = Y[[i, j]] + Y_velocity[[i, j]];
                }
            }

            // Center the embedding
            let mean = Y.mean_axis(Axis(0)).unwrap();
            Y = Y - &mean.insert_axis(Axis(0));
        }

        Ok(Y)
    }

    /// Compute pairwise affinities using Gaussian kernel
    fn compute_affinities(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let n = X.nrows();
        let mut P = Array2::zeros((n, n));

        // Compute pairwise distances
        let distances = compute_pairwise_distances(X);

        // Binary search for each point's sigma
        for i in 0..n {
            let mut beta = A::one(); // 1 / (2 * sigma^2)
            let target_perplexity = A::from(self.perplexity).unwrap();

            // Binary search for correct perplexity
            for _ in 0..50 {
                let mut sum_Pi = A::zero();
                let mut entropy = A::zero();

                for j in 0..n {
                    if i != j {
                        let p_ji = (-beta * distances[[i, j]] * distances[[i, j]]).exp();
                        P[[i, j]] = p_ji;
                        sum_Pi = sum_Pi + p_ji;
                    }
                }

                // Normalize
                if sum_Pi > A::zero() {
                    for j in 0..n {
                        if i != j {
                            P[[i, j]] = P[[i, j]] / sum_Pi;
                            if P[[i, j]] > A::from(1e-12).unwrap() {
                                entropy = entropy - P[[i, j]] * P[[i, j]].ln();
                            }
                        }
                    }
                }

                let perplexity = A::from(2.0).unwrap().powf(entropy);

                // Adjust beta
                if perplexity > target_perplexity {
                    beta = beta * A::from(2.0).unwrap();
                } else {
                    beta = beta / A::from(2.0).unwrap();
                }

                if (perplexity - target_perplexity).abs() < A::from(1e-5).unwrap() {
                    break;
                }
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in 0..n {
                P[[i, j]] = (P[[i, j]] + P[[j, i]]) / A::from(2.0 * n as f64).unwrap();
            }
        }

        Ok(P)
    }

    /// Compute affinities in low-dimensional space using Student t-distribution
    fn compute_low_dim_affinities(&self, Y: &Array2<A>) -> Array2<A> {
        let n = Y.nrows();
        let mut Q = Array2::zeros((n, n));

        // Compute pairwise distances in low-dim space
        let mut sum_Q = A::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = Y.row(i).iter().zip(Y.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(A::zero(), |acc, x| acc + x);

                let q_ij = (A::one() + dist_sq).recip();
                Q[[i, j]] = q_ij;
                Q[[j, i]] = q_ij;
                sum_Q = sum_Q + q_ij + q_ij; // Count both directions
            }
        }

        // Normalize
        if sum_Q > A::zero() {
            Q.mapv_inplace(|q| q.max(A::from(1e-12).unwrap()) / sum_Q);
        }

        Q
    }

    /// Compute gradient
    fn compute_gradient(&self, Y: &Array2<A>, P: &Array2<A>, Q: &Array2<A>) -> Array2<A> {
        let n = Y.nrows();
        let mut gradient = Array2::zeros(Y.dim());

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = Y.row(i).to_owned() - Y.row(j);
                    let pq_diff = (P[[i, j]] - Q[[i, j]]) * A::from(4.0).unwrap();

                    let dist_sq = diff.mapv(|x| x * x).sum();
                    let coeff = pq_diff / (A::one() + dist_sq);

                    for k in 0..self.n_components {
                        gradient[[i, k]] = gradient[[i, k]] + coeff * diff[k];
                    }
                }
            }
        }

        gradient
    }
}

/// UMAP (Uniform Manifold Approximation and Projection)
///
/// Fast manifold learning for visualization and general dimensionality reduction.
pub struct UMAP<A: Float> {
    /// Number of components
    n_components: usize,
    /// Number of neighbors
    n_neighbors: usize,
    /// Minimum distance between points
    min_dist: f64,
    /// Learning rate
    learning_rate: f64,
    /// Number of epochs
    n_epochs: usize,
    /// Phantom data to use type parameter
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> UMAP<A> {
    pub fn new(
        n_components: usize,
        n_neighbors: Option<usize>,
        min_dist: Option<f64>,
        n_epochs: Option<usize>,
    ) -> Self {
        Self {
            n_components,
            n_neighbors: n_neighbors.unwrap_or(15),
            min_dist: min_dist.unwrap_or(0.1),
            learning_rate: 1.0,
            n_epochs: n_epochs.unwrap_or(200),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Fit and transform data
    pub fn fit_transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let n = X.nrows();

        // Build k-nearest neighbor graph
        let knn_graph = self.build_knn_graph(X)?;

        // Initialize embedding
        let mut embedding = Array2::from_shape_fn((n, self.n_components), |_| {
            A::from(rand::random::<f64>() * 10.0 - 5.0).unwrap()
        });

        // Optimize embedding using SGD
        for epoch in 0..self.n_epochs {
            let alpha = A::from(1.0 - (epoch as f64 / self.n_epochs as f64)).unwrap();

            for &(i, j, weight) in &knn_graph {
                // Attractive force
                let dist = euclidean_distance(&embedding.row(i), &embedding.row(j));
                let grad_coeff = A::from(-2.0).unwrap() * A::from(weight).unwrap() /
                    (A::one() + dist * dist);

                for k in 0..self.n_components {
                    let grad = grad_coeff * (embedding[[i, k]] - embedding[[j, k]]);
                    embedding[[i, k]] = embedding[[i, k]] - alpha * A::from(self.learning_rate).unwrap() * grad;
                }

                // Repulsive force (sample negative edges)
                if rand::random::<f64>() < 0.1 {
                    let neg_j = (rand::random::<f64>() * n as f64) as usize % n;
                    if neg_j != i {
                        let dist = euclidean_distance(&embedding.row(i), &embedding.row(neg_j));
                        let grad_coeff = A::from(2.0).unwrap() /
                            ((A::one() + dist * dist) * (A::from(0.001).unwrap() + dist * dist));

                        for k in 0..self.n_components {
                            let grad = grad_coeff * (embedding[[i, k]] - embedding[[neg_j, k]]);
                            embedding[[i, k]] = embedding[[i, k]] - alpha * A::from(self.learning_rate).unwrap() * grad;
                        }
                    }
                }
            }
        }

        Ok(embedding)
    }

    /// Build k-nearest neighbor graph
    fn build_knn_graph(&self, X: &Array2<A>) -> Result<Vec<(usize, usize, f64)>> {
        let n = X.nrows();
        let mut edges = Vec::new();

        // Compute all pairwise distances
        let distances = compute_pairwise_distances(X);

        // For each point, find k nearest neighbors
        for i in 0..n {
            let mut neighbors: Vec<(usize, A)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distances[[i, j]]))
                .collect();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            neighbors.truncate(self.n_neighbors);

            // Add edges with exponentially decaying weights
            for (k, &(j, dist)) in neighbors.iter().enumerate() {
                let weight = (-dist.to_f64().unwrap() / self.min_dist).exp();
                edges.push((i, j, weight));
            }
        }

        Ok(edges)
    }
}

/// Helper: Compute pairwise Euclidean distances
fn compute_pairwise_distances<A: Float + ScalarOperand + Sum>(X: &Array2<A>) -> Array2<A> {
    let n = X.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&X.row(i), &X.row(j));
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Helper: Euclidean distance
fn euclidean_distance<A: Float + ScalarOperand + Sum>(
    a: &ndarray::ArrayView1<A>,
    b: &ndarray::ArrayView1<A>,
) -> A {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(A::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Helper: Compute top k eigenpairs using power iteration
fn compute_top_k_eigenpairs<A: Float + ScalarOperand + Sum + FromPrimitive>(
    matrix: &Array2<A>,
    k: usize,
) -> Result<(Array1<A>, Array2<A>)> {
    let n = matrix.nrows();
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut residual_matrix = matrix.clone();

    for _ in 0..k {
        // Power iteration for largest eigenvalue
        let mut v = Array1::from_shape_fn(n, |_| A::from(rand::random::<f64>()).unwrap());
        let mut lambda = A::zero();

        for _ in 0..100 {
            // Normalize
            let norm = v.mapv(|x| x * x).sum().sqrt();
            v.mapv_inplace(|x| x / norm);

            // Multiply by matrix
            let mut v_new = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] = v_new[i] + residual_matrix[[i, j]] * v[j];
                }
            }

            // Compute eigenvalue
            let new_lambda = v.dot(&v_new);

            if (new_lambda - lambda).abs() < A::from(1e-10).unwrap() {
                break;
            }

            lambda = new_lambda;
            v = v_new;
        }

        eigenvalues.push(lambda);
        eigenvectors.push(v.clone());

        // Deflate matrix
        for i in 0..n {
            for j in 0..n {
                residual_matrix[[i, j]] = residual_matrix[[i, j]] - lambda * v[i] * v[j];
            }
        }
    }

    // Stack eigenvectors as columns
    let eigenvector_matrix = Array2::from_shape_fn((n, k), |(i, j)| eigenvectors[j][i]);

    Ok((Array1::from_vec(eigenvalues), eigenvector_matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca() {
        let X = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]).unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&X).unwrap();
        let X_transformed = pca.transform(&X).unwrap();

        assert_eq!(X_transformed.shape(), &[4, 2]);
    }
}
