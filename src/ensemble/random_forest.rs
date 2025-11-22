//! Random Forest implementation
//!
//! Ensemble learning method that constructs multiple decision trees
//! and outputs the mode (classification) or mean (regression) of individual trees.

use crate::error::{Result, RustyAIError};
use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor, Criterion};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::iter::Sum;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Random Forest Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier<A: Float> {
    trees: Vec<DecisionTreeClassifier<A>>,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    criterion: Criterion,
    bootstrap: bool,
}

impl<A: Float + ScalarOperand + Sum> RandomForestClassifier<A> {
    /// Create a new RandomForestClassifier
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: Criterion::Gini,
            bootstrap: true,
        }
    }

    /// Set number of estimators
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
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

    /// Set criterion
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Fit the random forest
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.trees.clear();
        let mut rng = rand::thread_rng();
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Determine max_features if not set
        let max_features = self.max_features.unwrap_or_else(|| {
            // Default: sqrt(n_features) for classification
            (n_features as f64).sqrt() as usize
        });

        for _ in 0..self.n_estimators {
            // Bootstrap sample
            let mut bootstrap_indices = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                bootstrap_indices.push(rng.gen_range(0..n_samples));
            }

            // Select features (feature bagging)
            let mut selected_features: Vec<usize> = (0..n_features).collect();
            use rand::seq::SliceRandom;
            selected_features.shuffle(&mut rng);
            let selected_features = &selected_features[..max_features.min(n_features)];

            // Create bootstrap sample with selected features
            let mut X_boot = Array2::zeros((n_samples, selected_features.len()));
            let mut y_boot = Array1::zeros(n_samples);

            for (i, &idx) in bootstrap_indices.iter().enumerate() {
                for (j, &feat) in selected_features.iter().enumerate() {
                    X_boot[[i, j]] = X[[idx, feat]];
                }
                y_boot[i] = y[idx];
            }

            // Train tree
            let mut tree = DecisionTreeClassifier::new()
                .with_criterion(self.criterion)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf);

            if let Some(max_d) = self.max_depth {
                tree = tree.with_max_depth(max_d);
            }

            tree.fit(&X_boot, &y_boot)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.trees.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Collect predictions from all trees
        for i in 0..n_samples {
            let row = X.row(i).to_owned();
            let mut votes: HashMap<i32, usize> = HashMap::new();

            for tree in &self.trees {
                let row_2d = row.clone().insert_axis(ndarray::Axis(0));
                let pred = tree.predict(&row_2d)?;
                let pred_class = pred[0].to_i32().unwrap_or(0);
                *votes.entry(pred_class).or_insert(0) += 1;
            }

            // Majority vote
            let majority_class = votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| A::from(class).unwrap())
                .unwrap_or(A::zero());

            predictions[i] = majority_class;
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

impl<A: Float + ScalarOperand + Sum> Default for RandomForestClassifier<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Random Forest Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor<A: Float> {
    trees: Vec<DecisionTreeRegressor<A>>,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    bootstrap: bool,
}

impl<A: Float + ScalarOperand + Sum> RandomForestRegressor<A> {
    /// Create a new RandomForestRegressor
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
        }
    }

    /// Set number of estimators
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
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

    /// Fit the random forest
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.trees.clear();
        let mut rng = rand::thread_rng();
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Determine max_features if not set
        let max_features = self.max_features.unwrap_or_else(|| {
            // Default: n_features/3 for regression
            (n_features / 3).max(1)
        });

        for _ in 0..self.n_estimators {
            // Bootstrap sample
            let mut bootstrap_indices = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                bootstrap_indices.push(rng.gen_range(0..n_samples));
            }

            // Select features (feature bagging)
            let mut selected_features: Vec<usize> = (0..n_features).collect();
            use rand::seq::SliceRandom;
            selected_features.shuffle(&mut rng);
            let selected_features = &selected_features[..max_features.min(n_features)];

            // Create bootstrap sample with selected features
            let mut X_boot = Array2::zeros((n_samples, selected_features.len()));
            let mut y_boot = Array1::zeros(n_samples);

            for (i, &idx) in bootstrap_indices.iter().enumerate() {
                for (j, &feat) in selected_features.iter().enumerate() {
                    X_boot[[i, j]] = X[[idx, feat]];
                }
                y_boot[i] = y[idx];
            }

            // Train tree
            let mut tree = DecisionTreeRegressor::new()
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf);

            if let Some(max_d) = self.max_depth {
                tree = tree.with_max_depth(max_d);
            }

            tree.fit(&X_boot, &y_boot)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    /// Make predictions (average predictions from all trees)
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.trees.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let mut predictions = Array1::zeros(X.nrows());

        for tree in &self.trees {
            let tree_pred = tree.predict(X)?;
            predictions = predictions + tree_pred;
        }

        let n_trees = A::from(self.trees.len()).unwrap();
        Ok(predictions.mapv(|p: A| p / n_trees))
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

impl<A: Float + ScalarOperand + Sum> Default for RandomForestRegressor<A> {
    fn default() -> Self {
        Self::new()
    }
}

// For backward compatibility, export RandomForest as RandomForestRegressor
pub use RandomForestRegressor as RandomForest;
