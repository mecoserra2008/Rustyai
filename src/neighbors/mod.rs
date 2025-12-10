//! Nearest neighbors algorithms
//!
//! K-Nearest Neighbors for classification and regression.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::iter::Sum;

/// Distance metric for K-NN
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski { p: f64 },
}

/// K-Nearest Neighbors classifier
pub struct KNeighborsClassifier<A: Float> {
    /// Number of neighbors
    k: usize,
    /// Distance metric
    metric: Metric,
    /// Training data
    X_train_: Option<Array2<A>>,
    /// Training labels
    y_train_: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> KNeighborsClassifier<A> {
    /// Create a new K-Nearest Neighbors classifier
    ///
    /// # Arguments
    /// * `k` - Number of neighbors to use
    /// * `metric` - Distance metric (default: Euclidean)
    pub fn new(k: usize, metric: Option<Metric>) -> Self {
        Self {
            k,
            metric: metric.unwrap_or(Metric::Euclidean),
            X_train_: None,
            y_train_: None,
        }
    }

    /// Fit the K-NN classifier
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.X_train_ = Some(X.clone());
        self.y_train_ = Some(y.clone());

        Ok(())
    }

    /// Compute distance between two samples
    fn distance(&self, x1: &ndarray::ArrayView1<A>, x2: &ndarray::ArrayView1<A>) -> A {
        match self.metric {
            Metric::Euclidean => {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(A::zero(), |acc, x| acc + x)
                    .sqrt()
            }
            Metric::Manhattan => {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(A::zero(), |acc, x| acc + x)
            }
            Metric::Minkowski { p } => {
                let sum: A = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b).abs().powf(A::from(p).unwrap()))
                    .fold(A::zero(), |acc, x| acc + x);
                sum.powf(A::one() / A::from(p).unwrap())
            }
        }
    }

    /// Predict class labels
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let X_train = self.X_train_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let y_train = self.y_train_.as_ref().ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            // Compute distances to all training samples
            let x = X.row(i);
            let mut distances: Vec<(usize, A)> = (0..X_train.nrows())
                .map(|j| (j, self.distance(&x, &X_train.row(j))))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Get k nearest neighbors
            let k_neighbors = &distances[..self.k.min(distances.len())];

            // Vote for class (majority voting)
            let mut class_votes: HashMap<i32, usize> = HashMap::new();
            for &(idx, _) in k_neighbors {
                let class = y_train[idx].to_i32().unwrap_or(0);
                *class_votes.entry(class).or_insert(0) += 1;
            }

            // Get class with most votes
            let best_class = class_votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(0);

            predictions[i] = A::from(best_class).unwrap();
        }

        Ok(predictions)
    }

    /// Compute accuracy score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_y)| (pred - true_y).abs() < A::from(1e-10).unwrap())
            .count();
        Ok(A::from(correct).unwrap() / A::from(y.len()).unwrap())
    }
}

/// K-Nearest Neighbors regressor
pub struct KNeighborsRegressor<A: Float> {
    /// Number of neighbors
    k: usize,
    /// Distance metric
    metric: Metric,
    /// Weighting function (uniform or distance-weighted)
    weights: Weights,
    /// Training data
    X_train_: Option<Array2<A>>,
    /// Training labels
    y_train_: Option<Array1<A>>,
}

/// Weighting strategy for K-NN regression
#[derive(Debug, Clone, Copy)]
pub enum Weights {
    /// Uniform weights (all neighbors weighted equally)
    Uniform,
    /// Distance-based weights (closer neighbors weighted more heavily)
    Distance,
}

impl<A: Float + ScalarOperand + Sum> KNeighborsRegressor<A> {
    /// Create a new K-Nearest Neighbors regressor
    ///
    /// # Arguments
    /// * `k` - Number of neighbors to use
    /// * `metric` - Distance metric (default: Euclidean)
    /// * `weights` - Weighting strategy (default: Uniform)
    pub fn new(k: usize, metric: Option<Metric>, weights: Option<Weights>) -> Self {
        Self {
            k,
            metric: metric.unwrap_or(Metric::Euclidean),
            weights: weights.unwrap_or(Weights::Uniform),
            X_train_: None,
            y_train_: None,
        }
    }

    /// Fit the K-NN regressor
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.X_train_ = Some(X.clone());
        self.y_train_ = Some(y.clone());

        Ok(())
    }

    /// Compute distance between two samples
    fn distance(&self, x1: &ndarray::ArrayView1<A>, x2: &ndarray::ArrayView1<A>) -> A {
        match self.metric {
            Metric::Euclidean => {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(A::zero(), |acc, x| acc + x)
                    .sqrt()
            }
            Metric::Manhattan => {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(A::zero(), |acc, x| acc + x)
            }
            Metric::Minkowski { p } => {
                let sum: A = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b).abs().powf(A::from(p).unwrap()))
                    .fold(A::zero(), |acc, x| acc + x);
                sum.powf(A::one() / A::from(p).unwrap())
            }
        }
    }

    /// Predict target values
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let X_train = self.X_train_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let y_train = self.y_train_.as_ref().ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            // Compute distances to all training samples
            let x = X.row(i);
            let mut distances: Vec<(usize, A)> = (0..X_train.nrows())
                .map(|j| (j, self.distance(&x, &X_train.row(j))))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Get k nearest neighbors
            let k_neighbors = &distances[..self.k.min(distances.len())];

            // Compute weighted average
            let prediction = match self.weights {
                Weights::Uniform => {
                    // Simple average
                    let sum: A = k_neighbors.iter().map(|&(idx, _)| y_train[idx]).sum();
                    sum / A::from(k_neighbors.len()).unwrap()
                }
                Weights::Distance => {
                    // Distance-weighted average
                    let mut weighted_sum = A::zero();
                    let mut weight_sum = A::zero();

                    for &(idx, dist) in k_neighbors {
                        let weight = if dist < A::from(1e-10).unwrap() {
                            // If distance is zero, use very large weight
                            A::from(1e10).unwrap()
                        } else {
                            A::one() / dist
                        };

                        weighted_sum = weighted_sum + weight * y_train[idx];
                        weight_sum = weight_sum + weight;
                    }

                    if weight_sum > A::zero() {
                        weighted_sum / weight_sum
                    } else {
                        A::zero()
                    }
                }
            };

            predictions[i] = prediction;
        }

        Ok(predictions)
    }

    /// Compute R² score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;

        let y_mean = y.sum() / A::from(y.len()).unwrap();
        let ss_res: A = y
            .iter()
            .zip(predictions.iter())
            .map(|(&true_y, &pred_y)| {
                let diff = true_y - pred_y;
                diff * diff
            })
            .sum();

        let ss_tot: A = y
            .iter()
            .map(|&yi| {
                let diff = yi - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == A::zero() {
            return Ok(A::one());
        }

        Ok(A::one() - ss_res / ss_tot)
    }
}
