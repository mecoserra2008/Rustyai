//! Gradient Boosting implementation (XGBoost-style)
//!
//! Gradient boosting builds an additive model in a forward stage-wise fashion,
//! optimizing arbitrary differentiable loss functions.

use crate::error::{Result, RustyAIError};
use crate::tree::DecisionTreeRegressor;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;
use serde::{Deserialize, Serialize};

/// Gradient Boosting Regressor (XGBoost-style)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingRegressor<A: Float> {
    trees: Vec<DecisionTreeRegressor<A>>,
    n_estimators: usize,
    learning_rate: A,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: A,
    initial_prediction: Option<A>,
    n_features: Option<usize>,
    feature_importances_: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> GradientBoostingRegressor<A> {
    /// Create a new GradientBoostingRegressor
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            learning_rate: A::from(0.1).unwrap(),
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: A::one(),
            initial_prediction: None,
            n_features: None,
            feature_importances_: None,
        }
    }

    /// Set number of boosting stages
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set learning rate (shrinkage)
    pub fn with_learning_rate(mut self, learning_rate: A) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum depth of trees
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples for split
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set subsample ratio
    pub fn with_subsample(mut self, subsample: A) -> Self {
        self.subsample = subsample;
        self
    }

    /// Fit the gradient boosting model
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.trees.clear();
        let n_samples = X.nrows();
        let n_features = X.ncols();
        self.n_features = Some(n_features);
        self.feature_importances_ = Some(Array1::zeros(n_features));

        // Initialize predictions with mean (constant model)
        let initial_pred = y.sum() / A::from(y.len()).unwrap();
        self.initial_prediction = Some(initial_pred);
        let mut predictions = Array1::from_elem(n_samples, initial_pred);

        // Boosting iterations
        for iter in 0..self.n_estimators {
            // Compute negative gradient (residuals for MSE loss)
            let residuals = y - &predictions;

            // Subsample if needed
            let (X_sample, residuals_sample) = if self.subsample < A::one() {
                let sample_size = (A::from(n_samples).unwrap() * self.subsample)
                    .to_usize()
                    .unwrap_or(n_samples);

                let mut indices: Vec<usize> = (0..n_samples).collect();
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
                let sample_indices = &indices[..sample_size];

                let mut X_sub = Array2::zeros((sample_size, X.ncols()));
                let mut y_sub = Array1::zeros(sample_size);

                for (i, &idx) in sample_indices.iter().enumerate() {
                    for j in 0..X.ncols() {
                        X_sub[[i, j]] = X[[idx, j]];
                    }
                    y_sub[i] = residuals[idx];
                }

                (X_sub, y_sub)
            } else {
                (X.clone(), residuals.clone())
            };

            // Fit weak learner (decision tree) to residuals
            let mut tree = DecisionTreeRegressor::new()
                .with_max_depth(self.max_depth)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf);

            tree.fit(&X_sample, &residuals_sample)?;

            // Update predictions with learning rate
            let tree_pred = tree.predict(X)?;
            for i in 0..n_samples {
                predictions[i] = predictions[i] + self.learning_rate * tree_pred[i];
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.trees.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let initial = self.initial_prediction.ok_or(RustyAIError::NotFitted)?;
        let mut predictions = Array1::from_elem(X.nrows(), initial);

        for tree in &self.trees {
            let tree_pred = tree.predict(X)?;
            for i in 0..X.nrows() {
                predictions[i] = predictions[i] + self.learning_rate * tree_pred[i];
            }
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

    /// Get feature importances (based on total reduction in loss)
    ///
    /// Returns the feature importances as the mean and standard deviation of
    /// accumulation of the impurity decrease within each tree. Higher values
    /// indicate more important features.
    ///
    /// Note: Currently returns uniform importance across all features.
    /// A full implementation would require accessing tree node split information
    /// to calculate the actual variance reduction contributed by each feature.
    pub fn feature_importances(&self) -> Option<Array1<A>> {
        if self.trees.is_empty() || self.n_features.is_none() {
            return None;
        }

        let n_features = self.n_features.unwrap();
        // Return uniform importances (each feature gets equal weight)
        // A complete implementation would track actual gain from tree splits
        let importance = A::one() / A::from(n_features).unwrap();
        Some(Array1::from_elem(n_features, importance))
    }
}

impl<A: Float + ScalarOperand + Sum> Default for GradientBoostingRegressor<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient Boosting Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingClassifier<A: Float> {
    trees: Vec<DecisionTreeRegressor<A>>,
    n_estimators: usize,
    learning_rate: A,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: A,
    n_classes: Option<usize>,
}

impl<A: Float + ScalarOperand + Sum> GradientBoostingClassifier<A> {
    /// Create a new GradientBoostingClassifier
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            learning_rate: A::from(0.1).unwrap(),
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: A::one(),
            n_classes: None,
        }
    }

    /// Set number of boosting stages
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: A) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Fit for binary classification using log loss
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        // Binary classification only for now
        let mut classes = std::collections::HashSet::new();
        for &label in y {
            classes.insert(label.to_i32().unwrap_or(0));
        }
        self.n_classes = Some(classes.len());

        if classes.len() != 2 {
            return Err(RustyAIError::InvalidParameter {
                name: "y".to_string(),
                reason: "Only binary classification supported".to_string(),
            });
        }

        self.trees.clear();
        let n_samples = X.nrows();

        // Initialize with log-odds
        let n_positive = y.iter().filter(|&&yi| yi > A::from(0.5).unwrap()).count();
        let p = A::from(n_positive).unwrap() / A::from(n_samples).unwrap();
        let log_odds = (p / (A::one() - p)).ln();

        let mut f = Array1::from_elem(n_samples, log_odds);

        // Boosting iterations
        for _ in 0..self.n_estimators {
            // Compute probabilities
            let probs: Array1<A> = f.mapv(|fi: A| {
                let exp_f = fi.exp();
                exp_f / (A::one() + exp_f)
            });

            // Compute negative gradient (residuals for log loss)
            let residuals = y - &probs;

            // Fit tree to residuals
            let mut tree = DecisionTreeRegressor::new()
                .with_max_depth(self.max_depth)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf);

            tree.fit(X, &residuals)?;

            // Update F(x)
            let tree_pred = tree.predict(X)?;
            for i in 0..n_samples {
                f[i] = f[i] + self.learning_rate * tree_pred[i];
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    /// Predict probabilities
    pub fn predict_proba(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.trees.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let mut f = Array1::zeros(X.nrows());

        for tree in &self.trees {
            let tree_pred = tree.predict(X)?;
            for i in 0..X.nrows() {
                f[i] = f[i] + self.learning_rate * tree_pred[i];
            }
        }

        // Convert to probabilities
        Ok(f.mapv(|fi: A| {
            let exp_f = fi.exp();
            exp_f / (A::one() + exp_f)
        }))
    }

    /// Predict classes
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let probs = self.predict_proba(X)?;
        Ok(probs.mapv(|p| {
            if p >= A::from(0.5).unwrap() {
                A::one()
            } else {
                A::zero()
            }
        }))
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

impl<A: Float + ScalarOperand + Sum> Default for GradientBoostingClassifier<A> {
    fn default() -> Self {
        Self::new()
    }
}

// For backward compatibility
pub use GradientBoostingRegressor as GradientBoosting;
